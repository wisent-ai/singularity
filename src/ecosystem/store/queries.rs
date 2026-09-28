use super::{AppError, Result, Store, datas, sql};
use crate::ecosystem::model::{Observation, Source};
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
        let records = self.db.run(move |client| {
            let mut records = Vec::with_capacity(keys.len());
            for key in &keys {
                if let Some(row) = client
                    .query_opt(
                        "SELECT data FROM ecosystem_records WHERE being=$1 AND kind='observation'
                         AND data::jsonb->>'source'=$2 ORDER BY created_at DESC,id DESC LIMIT 1",
                        &[&being, key],
                    )
                    .map_err(sql)?
                {
                    records.push(row.get::<_, String>(0));
                }
            }
            Ok(records)
        })?;
        records
            .iter()
            .map(|record| serde_json::from_str(record).map_err(AppError::from))
            .collect()
    }

    pub fn related<T: DeserializeOwned>(&self, kind: &str, initiative: &str) -> Result<Vec<T>> {
        let (being, kind, initiative) =
            (self.being.clone(), kind.to_owned(), initiative.to_owned());
        datas(self.db.run(move |client| {
            client
                .query(
                    "SELECT data FROM ecosystem_records WHERE being=$1 AND kind=$2
                     AND data::jsonb->>'initiative_id'=$3 ORDER BY created_at,id",
                    &[&being, &kind, &initiative],
                )
                .map_err(sql)
        })?)
    }

    pub fn in_states<T: DeserializeOwned>(&self, kind: &str, states: &[&str]) -> Result<Vec<T>> {
        let (being, kind) = (self.being.clone(), kind.to_owned());
        let states: Vec<String> = states.iter().map(|state| (*state).to_owned()).collect();
        datas(self.db.run(move |client| {
            client
                .query(
                    "SELECT data FROM ecosystem_records WHERE being=$1 AND kind=$2
                     AND data::jsonb->>'state' = ANY($3) ORDER BY created_at,id",
                    &[&being, &kind, &states],
                )
                .map_err(sql)
        })?)
    }

    pub fn active_count(&self) -> Result<i64> {
        let being = self.being.clone();
        self.db.run(move |client| {
            Ok(client
                .query_one(
                    "SELECT count(*) FROM ecosystem_records WHERE being=$1 AND kind='initiative'
                     AND data::jsonb->>'state' NOT IN ('completed','stopped','failed')",
                    &[&being],
                )
                .map_err(sql)?
                .get(0))
        })
    }
}
