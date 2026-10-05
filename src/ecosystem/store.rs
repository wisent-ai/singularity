mod accounting;
mod fleet;
mod initialization;
mod queries;
mod records;
use super::model::{Initiative, Issue, Policy};
use crate::AppError;
use chrono::Utc;
use fleet::{metadata, record};
use sea_orm::sea_query::{Expr, OnConflict};
use sea_orm::{
    ActiveValue::NotSet, ActiveValue::Set, ColumnTrait, ConnectionTrait, EntityTrait,
    PaginatorTrait, QueryFilter, QueryOrder, QuerySelect, TransactionTrait,
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type Result<T, E = AppError> = std::result::Result<T, E>;
fn sql(error: sea_orm::DbErr) -> AppError {
    AppError::State(format!("ecosystem database: {error}"))
}

/// One being's ecosystem state in the fleet database: every row carries the
/// being's agent id, so beings share the database and never each other's rows.
pub struct Store {
    db: fleet::Db,
    being: String,
}

async fn meta_text(db: &impl ConnectionTrait, being: &str, key: &str) -> Result<Option<String>> {
    Ok(
        metadata::Entity::find_by_id((being.to_owned(), key.to_owned()))
            .one(db)
            .await
            .map_err(sql)?
            .map(|row| row.value),
    )
}

async fn set_meta_text(
    db: &impl ConnectionTrait,
    being: &str,
    key: &str,
    value: &str,
) -> Result<()> {
    metadata::Entity::insert(metadata::ActiveModel {
        being: Set(being.to_owned()),
        key: Set(key.to_owned()),
        value: Set(value.to_owned()),
    })
    .on_conflict(
        OnConflict::columns([metadata::Column::Being, metadata::Column::Key])
            .update_column(metadata::Column::Value)
            .to_owned(),
    )
    .exec_without_returning(db)
    .await
    .map_err(sql)?;
    Ok(())
}

fn parsed<T: DeserializeOwned>(text: Option<String>) -> Result<Option<T>> {
    text.map(|v| serde_json::from_str(&v).map_err(AppError::from))
        .transpose()
}

fn datas<T: DeserializeOwned>(rows: Vec<record::Model>) -> Result<Vec<T>> {
    rows.into_iter()
        .map(|row| serde_json::from_str(&row.data).map_err(AppError::from))
        .collect()
}

/// A condition on a top-level string field of a record's JSON.
fn field_is(field: &'static str, value: String) -> sea_orm::sea_query::SimpleExpr {
    Expr::cust_with_values(format!("data::jsonb->>'{field}' = $1"), [value])
}

/// `being`'s records of `kind`, oldest first.
fn of_kind(being: String, kind: String) -> sea_orm::Select<record::Entity> {
    record::Entity::find()
        .filter(record::Column::Being.eq(being))
        .filter(record::Column::Kind.eq(kind))
        .order_by_asc(record::Column::CreatedAt)
        .order_by_asc(record::Column::Id)
}

impl Store {
    pub fn meta<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let (being, key) = (self.being.clone(), key.to_owned());
        parsed(
            self.db
                .run(move |db| async move { meta_text(&db, &being, &key).await })?,
        )
    }
    pub fn set_meta(&self, key: &str, value: &impl Serialize) -> Result<()> {
        let (being, key, value) = (
            self.being.clone(),
            key.to_owned(),
            serde_json::to_string(value)?,
        );
        self.db
            .run(move |db| async move { set_meta_text(&db, &being, &key, &value).await })
    }
    pub fn clear_meta(&self, key: &str) -> Result<()> {
        let (being, key) = (self.being.clone(), key.to_owned());
        self.db.run(move |db| async move {
            metadata::Entity::delete_by_id((being, key))
                .exec(&db)
                .await
                .map_err(sql)?;
            Ok(())
        })
    }
    pub fn get<T: DeserializeOwned>(&self, kind: &str, id: &str) -> Result<Option<T>> {
        let key = (self.being.clone(), kind.to_owned(), id.to_owned());
        parsed(self.db.run(move |db| async move {
            Ok(record::Entity::find_by_id(key)
                .one(&db)
                .await
                .map_err(sql)?
                .map(|row| row.data))
        })?)
    }
    pub fn list<T: DeserializeOwned>(&self, kind: &str) -> Result<Vec<T>> {
        let (being, kind) = (self.being.clone(), kind.to_owned());
        datas(
            self.db
                .run(move |db| async move { of_kind(being, kind).all(&db).await.map_err(sql) })?,
        )
    }
    pub fn put(
        &self,
        kind: &str,
        id: &str,
        value: &impl Serialize,
        initiative: Option<&str>,
        detail: &str,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let data = serde_json::to_string(value)?;
        let content_sha256 = format!("{:x}", Sha256::digest(data.as_bytes()));
        let event_id = format!("event-{}", uuid::Uuid::new_v4());
        let event = json!({"id":event_id,"initiative_id":initiative,"kind":kind,"record_id":id,"record_sha256":content_sha256,"detail":detail,"created_at":now}).to_string();
        let event_sha256 = format!("{:x}", Sha256::digest(event.as_bytes()));
        let row = |kind: String, id: String, data: String, sha: String| record::ActiveModel {
            seq: NotSet,
            being: Set(self.being.clone()),
            kind: Set(kind),
            id: Set(id),
            data: Set(data),
            created_at: Set(now.clone()),
            updated_at: Set(now.clone()),
            content_sha256: Set(sha),
        };
        let written = row(kind.to_owned(), id.to_owned(), data, content_sha256);
        let event = row("event".into(), event_id, event, event_sha256);
        let (being, progress) = (self.being.clone(), json!(now).to_string());
        self.db.run(move |db| async move {
            let tx = db.begin().await.map_err(sql)?;
            record::Entity::insert(written)
                .on_conflict(
                    OnConflict::columns([
                        record::Column::Being,
                        record::Column::Kind,
                        record::Column::Id,
                    ])
                    .update_columns([
                        record::Column::Data,
                        record::Column::UpdatedAt,
                        record::Column::ContentSha256,
                    ])
                    .to_owned(),
                )
                .exec_without_returning(&tx)
                .await
                .map_err(sql)?;
            record::Entity::insert(event)
                .exec_without_returning(&tx)
                .await
                .map_err(sql)?;
            set_meta_text(&tx, &being, "last_progress_at", &progress).await?;
            tx.commit().await.map_err(sql)
        })
    }
    pub fn issue(&self, issue: &Issue) -> Result<()> {
        self.put("issue", &issue.operation, issue, None, &issue.message)
    }
    pub fn clear_issue(&self, operation: &str) -> Result<()> {
        let key = (self.being.clone(), "issue".to_owned(), operation.to_owned());
        self.db.run(move |db| async move {
            record::Entity::delete_by_id(key)
                .exec(&db)
                .await
                .map_err(sql)?;
            Ok(())
        })
    }
    pub fn paused(&self) -> Result<bool> {
        self.meta("paused")?
            .ok_or_else(|| AppError::State("missing pause state".into()))
    }
    fn count(&self, kind: &str) -> Result<u64> {
        let (being, kind) = (self.being.clone(), kind.to_owned());
        self.db
            .run(move |db| async move { of_kind(being, kind).count(&db).await.map_err(sql) })
    }
    pub fn status(&self, policy: &Policy) -> Result<Value> {
        let (spent, reserved) = self.balance()?;
        Ok(json!({"paused":self.paused()?,"updated_at":Utc::now(),
            "las_catalog_ready":self.meta::<bool>("las_catalog_ready")?.unwrap_or(false),
            "owner":self.meta::<Value>("owner")?,
            "runtime":{"version":env!("CARGO_PKG_VERSION"),"source_revision":option_env!("WISENT_SOURCE_COMMIT")},
            "last_progress_at":self.meta::<String>("last_progress_at")?,
            "active_count":self.active_count()?,
            "observation_count":self.count("observation")?,"opportunity_count":self.count("opportunity")?,"initiative_count":self.count("initiative")?,
            "spent_usd":spent.to_string(),"reserved_usd":reserved.to_string(),"budget_usd":policy.budget_usd.to_string(),
            "issues":self.list::<Issue>("issue")?}))
    }
    pub fn explain(&self, id: &str) -> Result<Value> {
        let initiative: Initiative = self
            .get("initiative", id)?
            .ok_or_else(|| AppError::State(format!("unknown initiative {id}")))?;
        let (being, initiative_id) = (self.being.clone(), id.to_owned());
        let related = self.db.run(move |db| async move {
            Ok(record::Entity::find()
                .select_only()
                .column(record::Column::Kind)
                .column_as(record::Column::Id.count(), "count")
                .filter(record::Column::Being.eq(being))
                .filter(field_is("initiative_id", initiative_id))
                .group_by(record::Column::Kind)
                .order_by_asc(record::Column::Kind)
                .into_tuple::<(String, i64)>()
                .all(&db)
                .await
                .map_err(sql)?
                .into_iter()
                .map(|(kind, count)| json!({"kind":kind,"count":count}))
                .collect::<Vec<_>>())
        })?;
        Ok(
            json!({"opportunity":self.get::<Value>("opportunity",&initiative.opportunity_id)?,
                "review":self.get::<Value>("review",&initiative.opportunity_id)?,"initiative":initiative,
                "related":related,"history":{"method":"records","params":{"initiative_id":id}}}),
        )
    }
}
