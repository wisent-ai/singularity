use super::super::control::protocol::{MAX_PAGE_SIZE, MAX_RECORD_BYTES, MAX_RESPONSE_BYTES};
use super::{AppError, Digest, Result, Sha256, Store, sql};
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
        let fetch = i64::from(limit) + 1;
        let rows = self.db.run(move |client| {
            client
                .query(
                    "SELECT seq,kind,id,created_at,updated_at,octet_length(data)::bigint,
                        left(coalesce(data::jsonb->>'title',data::jsonb->>'source',data::jsonb->>'purpose',
                            data::jsonb->>'operation',data::jsonb->>'detail',id),160),
                        left(coalesce(data::jsonb->>'state',data::jsonb->>'status'),64)
                     FROM ecosystem_records WHERE being=$1
                        AND ($2::text IS NULL OR kind=$2)
                        AND ($3::text IS NULL OR data::jsonb->>'initiative_id'=$3)
                        AND ($4::bigint IS NULL OR seq<$4)
                     ORDER BY seq DESC LIMIT $5",
                    &[&being, &kind, &initiative, &before, &fetch],
                )
                .map_err(sql)
        })?;
        let mut items = Vec::with_capacity(limit as usize);
        let mut last_cursor = None;
        let mut next_cursor = None;
        for row in rows {
            if items.len() == limit as usize {
                next_cursor = last_cursor;
                break;
            }
            last_cursor = Some(row.get::<_, i64>(0));
            items.push(json!({
                "kind":row.get::<_,String>(1),
                "id":row.get::<_,String>(2),
                "created_at":row.get::<_,String>(3),
                "updated_at":row.get::<_,String>(4),
                "total_bytes":row.get::<_,i64>(5),
                "preview":row.get::<_,String>(6),
                "state":row.get::<_,Option<String>>(7)
            }));
        }
        Ok(json!({"items":items,"next_cursor":next_cursor}))
    }

    pub fn items(&self, kind: &str, before: Option<i64>, limit: u32) -> Result<Value> {
        page_bounds(before, limit)?;
        let (being, kind_owned) = (self.being.clone(), kind.to_owned());
        let fetch = i64::from(limit) + 1;
        let rows = self.db.run(move |client| {
            client
                .query(
                    "SELECT seq,id,octet_length(data)::bigint,data FROM ecosystem_records
                     WHERE being=$1 AND kind=$2 AND ($3::bigint IS NULL OR seq<$3)
                     ORDER BY seq DESC LIMIT $4",
                    &[&being, &kind_owned, &before, &fetch],
                )
                .map_err(sql)
        })?;
        let mut items = Vec::with_capacity(limit as usize);
        let mut last_cursor = None;
        let mut next_cursor = None;
        let mut remaining = MAX_RESPONSE_BYTES / 2;
        for row in rows {
            let length = row.get::<_, i64>(2) as u64;
            if items.len() == limit as usize || length > remaining {
                if items.is_empty() {
                    let id: String = row.get(1);
                    return Err(AppError::State(format!(
                        "record {kind}/{id} exceeds a collection page; read it with ecosystem record {kind} {id}"
                    )));
                }
                next_cursor = last_cursor;
                break;
            }
            remaining -= length;
            items.push(serde_json::from_str::<Value>(row.get::<_, &str>(3))?);
            last_cursor = Some(row.get::<_, i64>(0));
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
        let (being, kind_owned, id_owned) = (self.being.clone(), kind.to_owned(), id.to_owned());
        let (digest, total, data): (String, u64, Vec<u8>) =
            self.db.run(move |client| {
                let row = client
                .query_opt(
                    "SELECT content_sha256,octet_length(data)::bigint,data FROM ecosystem_records
                     WHERE being=$1 AND kind=$2 AND id=$3",
                    &[&being, &kind_owned, &id_owned],
                )
                .map_err(sql)?
                .ok_or_else(|| AppError::State(format!("unknown record {kind_owned}/{id_owned}")))?;
                let data: String = row.get(2);
                Ok((row.get(0), row.get::<_, i64>(1) as u64, data.into_bytes()))
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
