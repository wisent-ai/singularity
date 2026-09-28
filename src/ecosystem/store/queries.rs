use super::fleet::record;
use super::{AppError, Result, Store, datas, field_is, of_kind, sql};
use crate::ecosystem::model::{Observation, Source};
use sea_orm::sea_query::Expr;
use sea_orm::{EntityTrait, PaginatorTrait, QueryFilter, QueryOrder, QuerySelect};
use serde::de::DeserializeOwned;

impl Store {
    pub fn latest_observations(&self, sources: &[Source]) -> Result<Vec<Observation>> {
        let mut keys: Vec<String> = Vec::with_capacity(sources.len());
        for source in sources {
            let key = serde_json::to_value(source)?;
            let key = key
                .as_str()
                .ok_or_else(|| AppError::State("observation source has no wire identity".into()))?
                .to_owned();
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        let being = self.being.clone();
        let records = self.db.run(move |db| async move {
            let mut records = Vec::with_capacity(keys.len());
            for key in keys {
                if let Some(row) = record::Entity::find()
                    .filter(record::Column::Being.eq(being.clone()))
                    .filter(record::Column::Kind.eq("observation"))
                    .filter(field_is("source", key))
                    .order_by_desc(record::Column::CreatedAt)
                    .order_by_desc(record::Column::Id)
                    .limit(1)
                    .one(&db)
                    .await
                    .map_err(sql)?
                {
                    records.push(row);
                }
            }
            Ok(records)
        })?;
        datas(records)
    }

    pub fn related<T: DeserializeOwned>(&self, kind: &str, initiative: &str) -> Result<Vec<T>> {
        let (being, kind, initiative) =
            (self.being.clone(), kind.to_owned(), initiative.to_owned());
        datas(self.db.run(move |db| async move {
            of_kind(being, kind)
                .filter(field_is("initiative_id", initiative))
                .all(&db)
                .await
                .map_err(sql)
        })?)
    }

    pub fn in_states<T: DeserializeOwned>(&self, kind: &str, states: &[&str]) -> Result<Vec<T>> {
        let (being, kind) = (self.being.clone(), kind.to_owned());
        let states: Vec<String> = states.iter().map(|state| (*state).to_owned()).collect();
        datas(self.db.run(move |db| async move {
            of_kind(being, kind)
                .filter(Expr::cust_with_values(
                    "data::jsonb->>'state' = ANY($1)",
                    [states],
                ))
                .all(&db)
                .await
                .map_err(sql)
        })?)
    }

    pub fn active_count(&self) -> Result<u64> {
        let being = self.being.clone();
        self.db.run(move |db| async move {
            of_kind(being, "initiative".into())
                .filter(Expr::cust(
                    "data::jsonb->>'state' NOT IN ('completed','stopped','failed')",
                ))
                .count(&db)
                .await
                .map_err(sql)
        })
    }
}
