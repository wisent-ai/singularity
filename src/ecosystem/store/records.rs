use super::super::control::protocol::{MAX_PAGE_SIZE, MAX_RECORD_BYTES, MAX_RESPONSE_BYTES};
use super::fleet::record;
use super::{AppError, Digest, Result, Sha256, Store, field_is, sql};
use sea_orm::sea_query::Expr;
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect};
use serde_json::{Value, json};

fn page_bounds(before: Option<i64>, limit: u32) -> Result<()> {
    if !(1..=MAX_PAGE_SIZE).contains(&limit) || before.is_some_and(|value| value <= 0) {
        return Err(AppError::Config(format!(
            "records requires limit 1..{MAX_PAGE_SIZE} and a positive before cursor"
        )));
    }
    Ok(())
}

impl Store {
    /// Newest first; the cursor is the record's sequence number in the store.
    pub fn records(
        &self,
        kind: Option<&str>,
        initiative: Option<&str>,
        before: Option<i64>,
        limit: u32,
    ) -> Result<Value> {
        page_bounds(before, limit)?;
        let being = self.being.clone();
        let kind = kind.map(str::to_owned);
        let initiative = initiative.map(str::to_owned);
        let fetch = u64::from(limit) + 1;
        type Row = (
            i64,
            String,
            String,
            String,
            String,
            i64,
            String,
            Option<String>,
        );
        let rows: Vec<Row> = self.db.run(move |db| async move {
            let mut query = record::Entity::find()
                .select_only()
                .column(record::Column::Seq)
                .column(record::Column::Kind)
                .column(record::Column::Id)
                .column(record::Column::CreatedAt)
                .column(record::Column::UpdatedAt)
                .column_as(Expr::cust("octet_length(data)::bigint"), "total_bytes")
                .column_as(
                    Expr::cust(
                        "left(coalesce(data::jsonb->>'title',data::jsonb->>'source',data::jsonb->>'purpose',
                            data::jsonb->>'operation',data::jsonb->>'detail',id),160)",
                    ),
                    "preview",
                )
                .column_as(
                    Expr::cust("left(coalesce(data::jsonb->>'state',data::jsonb->>'status'),64)"),
                    "state",
                )
                .filter(record::Column::Being.eq(being));
            if let Some(kind) = kind {
                query = query.filter(record::Column::Kind.eq(kind));
            }
            if let Some(initiative) = initiative {
                query = query.filter(field_is("initiative_id", initiative));
            }
            if let Some(before) = before {
                query = query.filter(record::Column::Seq.lt(before));
            }
            query
                .order_by_desc(record::Column::Seq)
                .limit(fetch)
                .into_tuple::<Row>()
                .all(&db)
                .await
                .map_err(sql)
        })?;
        let mut items = Vec::with_capacity(limit as usize);
        let mut last_cursor = None;
        let mut next_cursor = None;
        for (seq, kind, id, created_at, updated_at, total_bytes, preview, state) in rows {
            if items.len() == limit as usize {
                next_cursor = last_cursor;
                break;
            }
            last_cursor = Some(seq);
            items.push(json!({
                "kind":kind,
                "id":id,
                "created_at":created_at,
                "updated_at":updated_at,
                "total_bytes":total_bytes,
                "preview":preview,
                "state":state
            }));
        }
        Ok(json!({"items":items,"next_cursor":next_cursor}))
    }

    pub fn items(&self, kind: &str, before: Option<i64>, limit: u32) -> Result<Value> {
        page_bounds(before, limit)?;
        let (being, kind_owned) = (self.being.clone(), kind.to_owned());
        let fetch = u64::from(limit) + 1;
        let rows = self.db.run(move |db| async move {
            let mut query = record::Entity::find()
                .filter(record::Column::Being.eq(being))
                .filter(record::Column::Kind.eq(kind_owned));
            if let Some(before) = before {
                query = query.filter(record::Column::Seq.lt(before));
            }
            query
                .order_by_desc(record::Column::Seq)
                .limit(fetch)
                .all(&db)
                .await
                .map_err(sql)
        })?;
        let mut items = Vec::with_capacity(limit as usize);
        let mut last_cursor = None;
        let mut next_cursor = None;
        let mut remaining = MAX_RESPONSE_BYTES / 2;
        for row in rows {
            let length = row.data.len() as u64;
            if items.len() == limit as usize || length > remaining {
                if items.is_empty() {
                    let id = row.id;
                    return Err(AppError::State(format!(
                        "record {kind}/{id} exceeds a collection page; read it with ecosystem record {kind} {id}"
                    )));
                }
                next_cursor = last_cursor;
                break;
            }
            remaining -= length;
            items.push(serde_json::from_str::<Value>(&row.data)?);
            last_cursor = Some(row.seq);
        }
        Ok(json!({"items":items,"next_cursor":next_cursor}))
    }

    pub fn record(
        &self,
        kind: &str,
        id: &str,
        offset: u64,
        bytes: u32,
        revision: Option<&str>,
    ) -> Result<Value> {
        if !(4..=MAX_RECORD_BYTES).contains(&bytes) {
            return Err(AppError::Config(format!(
                "record requires bytes 4..{MAX_RECORD_BYTES}"
            )));
        }
        if offset > 0 && revision.is_none() {
            return Err(AppError::Config(
                "continuing a record requires its content_sha256 in revision".into(),
            ));
        }
        if revision.is_some_and(|value| {
            value.len() != 64
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }) {
            return Err(AppError::Config(
                "record revision must be a lowercase SHA-256 digest".into(),
            ));
        }
        let key = (self.being.clone(), kind.to_owned(), id.to_owned());
        let (kind_owned, id_owned) = (kind.to_owned(), id.to_owned());
        let (digest, total, data): (String, u64, Vec<u8>) = self.db.run(move |db| async move {
            let row = record::Entity::find_by_id(key)
                .one(&db)
                .await
                .map_err(sql)?
                .ok_or_else(|| {
                    AppError::State(format!("unknown record {kind_owned}/{id_owned}"))
                })?;
            let total = row.data.len() as u64;
            Ok((row.content_sha256, total, row.data.into_bytes()))
        })?;
        if let Some(expected) = revision {
            if expected != digest {
                return Err(AppError::State(format!(
                    "record {kind}/{id} revision mismatch: requested {expected}, observed {digest}; restart at offset 0 without revision to read its current contents"
                )));
            }
        }
        if offset > total {
            return Err(AppError::Config(format!(
                "record {kind}/{id} offset {offset} exceeds its {total} bytes"
            )));
        }
        let start = offset as usize;
        let end = (start + bytes as usize).min(data.len());
        let text = match String::from_utf8(data[start..end].to_vec()) {
            Ok(text) => text,
            Err(error) if error.utf8_error().error_len().is_none() => {
                let valid = error.utf8_error().valid_up_to();
                let mut data = error.into_bytes();
                data.truncate(valid);
                String::from_utf8(data).map_err(|error| AppError::State(error.to_string()))?
            }
            Err(_) => {
                return Err(AppError::Config(format!(
                    "record {kind}/{id} offset {offset} is not a UTF-8 boundary, or its stored text is invalid"
                )));
            }
        };
        let end = offset + text.len() as u64;
        if end == offset && offset < total {
            return Err(AppError::State(format!(
                "record {kind}/{id} returned no complete character at offset {offset}"
            )));
        }
        debug_assert_eq!(digest, format!("{:x}", Sha256::digest(&data)));
        Ok(
            json!({"kind":kind,"id":id,"offset":offset,"total_bytes":total,
            "content_sha256":digest,"text":text,"next_offset":(end < total).then_some(end)}),
        )
    }
}
