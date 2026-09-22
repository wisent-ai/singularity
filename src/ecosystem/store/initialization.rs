use super::{AppError, Policy, Result, Store, sql};
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use std::path::Path;

impl Store {
    pub fn open(
        directory: &Path,
        policy: &Policy,
        identity: &crate::AgentIdentity,
    ) -> Result<Self> {
        let owner = json!({"agent_id":identity.agent_id,"role":identity.role,
            "environment":identity.environment,"workload_id":identity.workload_id});
        let connection = Connection::open(directory.join("ecosystem.sqlite3")).map_err(sql)?;
        connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL,
                data TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT,
                initiative_id TEXT, kind TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_initiative ON events(initiative_id,id);
            CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, amount TEXT NOT NULL,
                state TEXT NOT NULL, actual TEXT);").map_err(sql)?;
        let store = Self { connection };
        if let Some(version) = store.meta::<u32>("schema_version")? {
            if version != 1 {
                return Err(AppError::State(format!(
                    "unsupported ecosystem schema {version}"
                )));
            }
            let previous = store
                .meta::<Policy>("policy")?
                .ok_or_else(|| AppError::State("ecosystem policy missing".into()))?;
            if serde_json::to_value(previous)? != serde_json::to_value(policy)? {
                return Err(AppError::State("ecosystem authority differs from its persisted policy; authority cannot change on resume".into()));
            }
            if store.meta::<Value>("owner")?.as_ref() != Some(&owner) {
                return Err(AppError::State("ecosystem owner differs from its persisted principal; execution credentials cannot change owner on resume".into()));
            }
            if store
                .meta::<u64>("policy_sequence")?
                .is_none_or(|previous| previous > identity.policy_sequence)
            {
                return Err(AppError::State(
                    "ecosystem policy sequence is missing or would roll back".into(),
                ));
            }
        } else {
            let tx = store.connection.unchecked_transaction().map_err(sql)?;
            for (key, value) in [
                ("schema_version", json!(1)),
                ("policy", serde_json::to_value(policy)?),
                ("owner", owner),
                ("policy_sequence", json!(identity.policy_sequence)),
                ("paused", json!(false)),
                ("spent_usd", json!("0")),
                ("reserved_usd", json!("0")),
            ] {
                tx.execute(
                    "INSERT INTO metadata VALUES (?1,?2)",
                    params![key, value.to_string()],
                )
                .map_err(sql)?;
            }
            tx.commit().map_err(sql)?;
        }
        store.connection.execute_batch(
            "CREATE INDEX IF NOT EXISTS observations_source_time ON records(json_extract(data,'$.source'),created_at DESC,id DESC) WHERE kind='observation';
             CREATE INDEX IF NOT EXISTS records_initiative_time ON records(kind,json_extract(data,'$.initiative_id'),created_at,id);
             CREATE INDEX IF NOT EXISTS records_state_time ON records(kind,json_extract(data,'$.state'),created_at,id);"
        ).map_err(sql)?;
        store.set_meta("policy_sequence", &identity.policy_sequence)?;
        store.put(
            "owner_launch",
            &uuid::Uuid::new_v4().to_string(),
            identity,
            None,
            "Recorded configured owner identity; signed tool admission is checked separately",
        )?;
        Ok(store)
    }
}
