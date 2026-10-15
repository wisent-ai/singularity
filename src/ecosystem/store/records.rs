use std::num::{NonZeroU32, NonZeroU64};

use super::fleet::record;
use super::{AppError, Digest, Result, Sha256, Store, field_is, sql};
use sea_orm::sea_query::Expr;
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect, Select};
use serde_json::{Value, json};

/// A page holds `limit` records when the caller names one, and every record otherwise.
fn page_bounds(before: Option<i64>, limit: Option<u32>) -> Result<()> {
    let empty_page = limit.is_some_and(|limit| NonZeroU32::new(limit).is_none());
    let unusable_cursor = before.is_some_and(|value| !value.is_positive());
    if empty_page || unusable_cursor {
        return Err(AppError::Config(
            "records takes a limit of at least 1 when given and a positive before cursor".into(),
        ));
    }
    Ok(())
}

/// `being`'s records, narrowed to one kind and one initiative when named.
fn listing(
    being: String,
    kind: Option<String>,
    initiative: Option<String>,
) -> Select<record::Entity> {
    let mut query = record::Entity::find().filter(record::Column::Being.eq(being));
    if let Some(kind) = kind {
        query = query.filter(record::Column::Kind.eq(kind));
    }
    if let Some(initiative) = initiative {
        query = query.filter(field_is("initiative_id", initiative));
    }
    query
}

/// Whether a full page has an older record behind it: the next page exists
/// exactly when the listing holds a record older than this page's last.
fn page_is_full(limit: Option<u32>, read: usize) -> bool {
    limit.is_some_and(|limit| read == limit as usize)
}

impl Store {
    /// Newest first; the cursor is the record's sequence number in the store.
    pub fn records(
        &self,
        kind: Option<&str>,
        initiative: Option<&str>,
        before: Option<i64>,
        limit: Option<u32>,
    ) -> Result<Value> {
        page_bounds(before, limit)?;
        let being = self.being.clone();
        let kind = kind.map(str::to_owned);
        let initiative = initiative.map(str::to_owned);
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
        let (rows, older): (Vec<Row>, bool) = self.db.run(move |db| async move {
            let base = listing(being, kind, initiative);
            let mut page = base
                .clone()
                .select_only()
                .column(record::Column::Seq)
                .column(record::Column::Kind)
                .column(record::Column::Id)
                .column(record::Column::CreatedAt)
                .column(record::Column::UpdatedAt)
                .column_as(Expr::cust("octet_length(data)::bigint"), "total_bytes")
                .column_as(
                    Expr::cust(
                        "coalesce(data::jsonb->>'title',data::jsonb->>'source',data::jsonb->>'purpose',
                            data::jsonb->>'operation',data::jsonb->>'detail',id)",
                    ),
                    "preview",
                )
                .column_as(
                    Expr::cust("coalesce(data::jsonb->>'state',data::jsonb->>'status')"),
                    "state",
                );
            if let Some(before) = before {
                page = page.filter(record::Column::Seq.lt(before));
            }
            if let Some(limit) = limit {
                page = page.limit(u64::from(limit));
            }
            let rows = page
                .order_by_desc(record::Column::Seq)
                .into_tuple::<Row>()
                .all(&db)
                .await
                .map_err(sql)?;
            let older = match rows.last() {
                Some(last) if page_is_full(limit, rows.len()) => base
                    .filter(record::Column::Seq.lt(last.0))
                    .select_only()
                    .column(record::Column::Seq)
                    .into_tuple::<i64>()
                    .one(&db)
                    .await
                    .map_err(sql)?
                    .is_some(),
                _ => false,
            };
            Ok((rows, older))
        })?;
        let next_cursor = match rows.last() {
            Some(last) if older => Some(last.0),
            _ => None,
        };
        let items = rows
            .into_iter()
            .map(
                |(_, kind, id, created_at, updated_at, total_bytes, preview, state)| {
                    json!({
                        "kind":kind,
                        "id":id,
                        "created_at":created_at,
                        "updated_at":updated_at,
                        "total_bytes":total_bytes,
                        "preview":preview,
                        "state":state
                    })
                },
            )
            .collect::<Vec<_>>();
        Ok(json!({"items":items,"next_cursor":next_cursor}))
    }

    pub fn items(&self, kind: &str, before: Option<i64>, limit: Option<u32>) -> Result<Value> {
        page_bounds(before, limit)?;
        let (being, kind_owned) = (self.being.clone(), kind.to_owned());
        let (rows, older) = self.db.run(move |db| async move {
            let base = listing(being, Some(kind_owned), None);
            let mut page = base.clone();
            if let Some(before) = before {
                page = page.filter(record::Column::Seq.lt(before));
            }
            if let Some(limit) = limit {
                page = page.limit(u64::from(limit));
            }
            let rows = page
                .order_by_desc(record::Column::Seq)
                .all(&db)
                .await
                .map_err(sql)?;
            let older = match rows.last() {
                Some(last) if page_is_full(limit, rows.len()) => base
                    .filter(record::Column::Seq.lt(last.seq))
                    .one(&db)
                    .await
                    .map_err(sql)?
                    .is_some(),
                _ => false,
            };
            Ok((rows, older))
        })?;
        let next_cursor = match rows.last() {
            Some(last) if older => Some(last.seq),
            _ => None,
        };
        let items = rows
            .iter()
            .map(|row| serde_json::from_str::<Value>(&row.data))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(json!({"items":items,"next_cursor":next_cursor}))
    }

    pub fn record(
        &self,
        kind: &str,
        id: &str,
        offset: u64,
        bytes: Option<u32>,
        revision: Option<&str>,
    ) -> Result<Value> {
        if bytes.is_some_and(|bytes| NonZeroU32::new(bytes).is_none()) {
            return Err(AppError::Config(
                "record bytes must be at least 1 when given; omit it to read the whole record"
                    .into(),
            ));
        }
        if NonZeroU64::new(offset).is_some() && revision.is_none() {
            return Err(AppError::Config(
                "continuing a record requires its content_sha256 in revision".into(),
            ));
        }
        // A revision is a lowercase hex SHA-256 digest, as long as one is.
        let digest_length = hex::encode(Sha256::digest(b"")).len();
        if revision.is_some_and(|value| {
            value.len() != digest_length
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
        // Without a fragment size the record is read whole from the offset.
        let end = match bytes {
            Some(bytes) => (start + bytes as usize).min(data.len()),
            None => data.len(),
        };
        let text = match String::from_utf8(data[start..end].to_vec()) {
            Ok(text) => text,
            Err(error) if error.utf8_error().error_len().is_none() => {
                let valid = error.utf8_error().valid_up_to();
                let mut data = error.into_bytes();
                data.truncate(valid);
                String::from_utf8(data).map_err(|error| AppError::State(error.to_string()))?
            }
            Err(error) => {
                return Err(AppError::Config(format!(
                    "record {kind}/{id} offset {offset} is not a UTF-8 boundary, or its stored text is invalid: {error}"
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
