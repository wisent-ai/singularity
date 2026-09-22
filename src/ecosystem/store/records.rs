use super::super::control::protocol::{MAX_PAGE_SIZE, MAX_RECORD_BYTES, MAX_RESPONSE_BYTES};
use super::{AppError, Digest, Result, Sha256, Store, sql};
use rusqlite::{OptionalExtension, params};
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
    pub fn records(
        &self,
        kind: Option<&str>,
        initiative: Option<&str>,
        before: Option<i64>,
        limit: u32,
    ) -> Result<Value> {
        page_bounds(before, limit)?;
        let filter = match (kind, initiative) {
            (Some(_), Some(_)) => "kind=?1 AND json_extract(data,'$.initiative_id')=?2",
            (Some(_), None) => "kind=?1",
            (None, Some(_)) => "json_extract(data,'$.initiative_id')=?2",
            (None, None) => "1",
        };
        let mut statement = self.connection.prepare(&format!(
            "SELECT rowid,kind,id,created_at,updated_at,length(CAST(data AS BLOB)),
                substr(CAST(coalesce(json_extract(data,'$.title'),json_extract(data,'$.source'),
                    json_extract(data,'$.purpose'),json_extract(data,'$.operation'),json_extract(data,'$.detail'),id) AS TEXT),1,160),
                substr(CAST(coalesce(json_extract(data,'$.state'),json_extract(data,'$.status')) AS TEXT),1,64)
             FROM records WHERE {filter} AND (?3 IS NULL OR rowid<?3)
             ORDER BY rowid DESC LIMIT ?4"
        )).map_err(sql)?;
        let mut rows = statement
            .query(params![kind, initiative, before, limit + 1])
            .map_err(sql)?;
        let mut items = Vec::with_capacity(limit as usize);
        let mut last_cursor = None;
        let mut next_cursor = None;
        while let Some(row) = rows.next().map_err(sql)? {
            if items.len() == limit as usize {
                next_cursor = last_cursor;
                break;
            }
            last_cursor = Some(row.get::<_, i64>(0).map_err(sql)?);
            items.push(json!({
                "kind":row.get::<_,String>(1).map_err(sql)?,
                "id":row.get::<_,String>(2).map_err(sql)?,
                "created_at":row.get::<_,String>(3).map_err(sql)?,
                "updated_at":row.get::<_,String>(4).map_err(sql)?,
                "total_bytes":row.get::<_,u64>(5).map_err(sql)?,
                "preview":row.get::<_,String>(6).map_err(sql)?,
                "state":row.get::<_,Option<String>>(7).map_err(sql)?
            }));
        }
        Ok(json!({"items":items,"next_cursor":next_cursor}))
    }

    pub fn items(&self, kind: &str, before: Option<i64>, limit: u32) -> Result<Value> {
        page_bounds(before, limit)?;
        let mut statement = self
            .connection
            .prepare(
                "SELECT rowid,id,length(CAST(data AS BLOB)),data FROM records
             WHERE kind=?1 AND (?2 IS NULL OR rowid<?2) ORDER BY rowid DESC LIMIT ?3",
            )
            .map_err(sql)?;
        let mut rows = statement
            .query(params![kind, before, limit + 1])
            .map_err(sql)?;
        let mut items = Vec::with_capacity(limit as usize);
        let mut last_cursor = None;
        let mut next_cursor = None;
        let mut remaining = MAX_RESPONSE_BYTES / 2;
        while let Some(row) = rows.next().map_err(sql)? {
            let length: u64 = row.get(2).map_err(sql)?;
            if items.len() == limit as usize || length > remaining {
                if items.is_empty() {
                    let id: String = row.get(1).map_err(sql)?;
                    return Err(AppError::State(format!(
                        "record {kind}/{id} exceeds a collection page; read it with ecosystem record {kind} {id}"
                    )));
                }
                next_cursor = last_cursor;
                break;
            }
            remaining -= length;
            let data: String = row.get(3).map_err(sql)?;
            items.push(serde_json::from_str::<Value>(&data)?);
            last_cursor = Some(row.get::<_, i64>(0).map_err(sql)?);
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
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let (mut digest, total): (String, u64) = tx.query_row(
            "SELECT content_sha256,length(CAST(data AS BLOB)) FROM records WHERE kind=?1 AND id=?2",
            params![kind,id], |row| Ok((row.get(0)?,row.get(1)?)),
        ).optional().map_err(sql)?.ok_or_else(|| AppError::State(format!("unknown record {kind}/{id}")))?;
        // Schema-one histories acquire their digest when first read, not by loading the whole history at startup.
        if digest.is_empty() {
            let data: String = tx
                .query_row(
                    "SELECT data FROM records WHERE kind=?1 AND id=?2",
                    params![kind, id],
                    |row| row.get(0),
                )
                .map_err(sql)?;
            digest = format!("{:x}", Sha256::digest(data.as_bytes()));
            tx.execute(
                "UPDATE records SET content_sha256=?3 WHERE kind=?1 AND id=?2",
                params![kind, id, digest],
            )
            .map_err(sql)?;
        }
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
        let data: Vec<u8> = tx
            .query_row(
                "SELECT substr(CAST(data AS BLOB),?3,?4) FROM records WHERE kind=?1 AND id=?2",
                params![kind, id, offset + 1, bytes],
                |row| row.get(0),
            )
            .map_err(sql)?;
        let text = match String::from_utf8(data) {
            Ok(text) => text,
            Err(error) if error.utf8_error().error_len().is_none() => {
                let end = error.utf8_error().valid_up_to();
                let mut data = error.into_bytes();
                data.truncate(end);
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
        tx.commit().map_err(sql)?;
        Ok(
            json!({"kind":kind,"id":id,"offset":offset,"total_bytes":total,
            "content_sha256":digest,"text":text,"next_offset":(end < total).then_some(end)}),
        )
    }
}
