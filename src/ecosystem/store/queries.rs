use super::{AppError, Result, Store, sql};
use crate::ecosystem::model::{Observation, Source};
use rusqlite::{OptionalExtension, params};
use serde::de::DeserializeOwned;

impl Store {
    pub fn latest_observations(&self, sources: &[Source]) -> Result<Vec<Observation>> {
        let mut statement = self.connection.prepare_cached(
            "SELECT data FROM records WHERE kind='observation' AND json_extract(data,'$.source')=?1 ORDER BY created_at DESC,id DESC LIMIT 1"
        ).map_err(sql)?;
        let mut observations: Vec<Observation> = Vec::with_capacity(sources.len());
        for source in sources {
            if observations
                .iter()
                .any(|observation| observation.source == *source)
            {
                continue;
            }
            let key = serde_json::to_value(source)?;
            let key = key
                .as_str()
                .ok_or_else(|| AppError::State("observation source has no wire identity".into()))?;
            let record = statement
                .query_row([key], |row| row.get::<_, String>(0))
                .optional()
                .map_err(sql)?;
            if let Some(record) = record {
                observations.push(serde_json::from_str(&record)?);
            }
        }
        Ok(observations)
    }

    pub fn related<T: DeserializeOwned>(&self, kind: &str, initiative: &str) -> Result<Vec<T>> {
        let mut statement = self.connection.prepare_cached(
            "SELECT data FROM records WHERE kind=?1 AND json_extract(data,'$.initiative_id')=?2 ORDER BY created_at,id"
        ).map_err(sql)?;
        statement
            .query_map(params![kind, initiative], |row| row.get::<_, String>(0))
            .map_err(sql)?
            .map(|row| serde_json::from_str(&row.map_err(sql)?).map_err(AppError::from))
            .collect()
    }

    pub fn in_states<T: DeserializeOwned>(&self, kind: &str, states: &[&str]) -> Result<Vec<T>> {
        let mut statement = self.connection.prepare_cached(
            "SELECT data FROM records WHERE kind=?1 AND json_extract(data,'$.state') IN (SELECT value FROM json_each(?2)) ORDER BY created_at,id"
        ).map_err(sql)?;
        statement
            .query_map(params![kind, serde_json::to_string(states)?], |row| {
                row.get::<_, String>(0)
            })
            .map_err(sql)?
            .map(|row| serde_json::from_str(&row.map_err(sql)?).map_err(AppError::from))
            .collect()
    }

    pub fn active_count(&self) -> Result<i64> {
        self.connection.query_row(
            "SELECT count(*) FROM records WHERE kind='initiative' AND json_extract(data,'$.state') NOT IN ('completed','stopped','failed')",
            [], |row| row.get(0)
        ).map_err(sql)
    }
}
