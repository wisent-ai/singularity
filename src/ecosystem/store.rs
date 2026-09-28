mod accounting;
mod fleet;
mod initialization;
mod queries;
mod records;
use super::model::{Initiative, Issue, Policy};
use crate::AppError;
use chrono::Utc;
use postgres::GenericClient;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type Result<T, E = AppError> = std::result::Result<T, E>;
fn sql(error: postgres::Error) -> AppError {
    AppError::State(format!("ecosystem database: {error}"))
}

/// One being's ecosystem state in the fleet database: every row carries the
/// being's agent id, so beings share the database and never each other's rows.
pub struct Store {
    db: fleet::Db,
    being: String,
}

fn meta_text(client: &mut impl GenericClient, being: &str, key: &str) -> Result<Option<String>> {
    Ok(client
        .query_opt(
            "SELECT value FROM ecosystem_metadata WHERE being=$1 AND key=$2",
            &[&being, &key],
        )
        .map_err(sql)?
        .map(|row| row.get(0)))
}

fn set_meta_text(
    client: &mut impl GenericClient,
    being: &str,
    key: &str,
    value: &str,
) -> Result<()> {
    client
        .execute(
            "INSERT INTO ecosystem_metadata(being,key,value) VALUES ($1,$2,$3)
             ON CONFLICT(being,key) DO UPDATE SET value=excluded.value",
            &[&being, &key, &value],
        )
        .map_err(sql)?;
    Ok(())
}

fn parsed<T: DeserializeOwned>(text: Option<String>) -> Result<Option<T>> {
    text.map(|v| serde_json::from_str(&v).map_err(AppError::from))
        .transpose()
}

fn datas<T: DeserializeOwned>(rows: Vec<postgres::Row>) -> Result<Vec<T>> {
    rows.into_iter()
        .map(|row| serde_json::from_str(row.get::<_, &str>(0)).map_err(AppError::from))
        .collect()
}

impl Store {
    pub fn meta<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let (being, key) = (self.being.clone(), key.to_owned());
        parsed(self.db.run(move |client| meta_text(client, &being, &key))?)
    }
    pub fn set_meta(&self, key: &str, value: &impl Serialize) -> Result<()> {
        let (being, key, value) = (
            self.being.clone(),
            key.to_owned(),
            serde_json::to_string(value)?,
        );
        self.db
            .run(move |client| set_meta_text(client, &being, &key, &value))
    }
    pub fn clear_meta(&self, key: &str) -> Result<()> {
        let (being, key) = (self.being.clone(), key.to_owned());
        self.db.run(move |client| {
            client
                .execute(
                    "DELETE FROM ecosystem_metadata WHERE being=$1 AND key=$2",
                    &[&being, &key],
                )
                .map_err(sql)?;
            Ok(())
        })
    }
    pub fn get<T: DeserializeOwned>(&self, kind: &str, id: &str) -> Result<Option<T>> {
        let (being, kind, id) = (self.being.clone(), kind.to_owned(), id.to_owned());
        parsed(self.db.run(move |client| {
            Ok(client
                .query_opt(
                    "SELECT data FROM ecosystem_records WHERE being=$1 AND kind=$2 AND id=$3",
                    &[&being, &kind, &id],
                )
                .map_err(sql)?
                .map(|row| row.get::<_, String>(0)))
        })?)
    }
    pub fn list<T: DeserializeOwned>(&self, kind: &str) -> Result<Vec<T>> {
        let (being, kind) = (self.being.clone(), kind.to_owned());
        datas(self.db.run(move |client| {
            client
                .query(
                    "SELECT data FROM ecosystem_records WHERE being=$1 AND kind=$2 ORDER BY created_at,id",
                    &[&being, &kind],
                )
                .map_err(sql)
        })?)
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
        let (being, kind, id) = (self.being.clone(), kind.to_owned(), id.to_owned());
        self.db.run(move |client| {
            let mut tx = client.transaction().map_err(sql)?;
            tx.execute(
                "INSERT INTO ecosystem_records(being,kind,id,data,created_at,updated_at,content_sha256)
                 VALUES ($1,$2,$3,$4,$5,$5,$6)
                 ON CONFLICT(being,kind,id) DO UPDATE SET data=excluded.data,updated_at=excluded.updated_at,content_sha256=excluded.content_sha256",
                &[&being, &kind, &id, &data, &now, &content_sha256],
            )
            .map_err(sql)?;
            tx.execute(
                "INSERT INTO ecosystem_records(being,kind,id,data,created_at,updated_at,content_sha256)
                 VALUES ($1,'event',$2,$3,$4,$4,$5)",
                &[&being, &event_id, &event, &now, &event_sha256],
            )
            .map_err(sql)?;
            set_meta_text(&mut tx, &being, "last_progress_at", &json!(now).to_string())?;
            tx.commit().map_err(sql)
        })
    }
    pub fn issue(&self, issue: &Issue) -> Result<()> {
        self.put("issue", &issue.operation, issue, None, &issue.message)
    }
    pub fn clear_issue(&self, operation: &str) -> Result<()> {
        let (being, operation) = (self.being.clone(), operation.to_owned());
        self.db.run(move |client| {
            client
                .execute(
                    "DELETE FROM ecosystem_records WHERE being=$1 AND kind='issue' AND id=$2",
                    &[&being, &operation],
                )
                .map_err(sql)?;
            Ok(())
        })
    }
    pub fn paused(&self) -> Result<bool> {
        self.meta("paused")?
            .ok_or_else(|| AppError::State("missing pause state".into()))
    }
    fn count(&self, kind: &str) -> Result<i64> {
        let (being, kind) = (self.being.clone(), kind.to_owned());
        self.db.run(move |client| {
            Ok(client
                .query_one(
                    "SELECT count(*) FROM ecosystem_records WHERE being=$1 AND kind=$2",
                    &[&being, &kind],
                )
                .map_err(sql)?
                .get(0))
        })
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
        let related = self.db.run(move |client| {
            Ok(client
                .query(
                    "SELECT kind,count(*) FROM ecosystem_records
                     WHERE being=$1 AND data::jsonb->>'initiative_id'=$2 GROUP BY kind ORDER BY kind",
                    &[&being, &initiative_id],
                )
                .map_err(sql)?
                .into_iter()
                .map(|row| json!({"kind":row.get::<_,String>(0),"count":row.get::<_,i64>(1)}))
                .collect::<Vec<_>>())
        })?;
        Ok(
            json!({"opportunity":self.get::<Value>("opportunity",&initiative.opportunity_id)?,
                "review":self.get::<Value>("review",&initiative.opportunity_id)?,"initiative":initiative,
                "related":related,"history":{"method":"records","params":{"initiative_id":id,"limit":super::protocol::DEFAULT_PAGE_SIZE}}}),
        )
    }
}
