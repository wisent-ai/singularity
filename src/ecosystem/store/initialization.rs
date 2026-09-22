use super::{AppError, Policy, Result, Store, sql};
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use std::path::Path;

impl Store {
    pub fn open(
        directory: &Path,
        policy: &Policy,
        identity: &crate::AgentIdentity,
        start_paused: bool,
    ) -> Result<Self> {
        let owner = json!({"agent_id":identity.agent_id,"role":identity.role,
            "environment":identity.environment,"workload_id":identity.workload_id});
        let connection = Connection::open(directory.join("ecosystem.sqlite3")).map_err(sql)?;
        connection
            .execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;",
            )
            .map_err(sql)?;
        let store = Self { connection };
        let tx = store.connection.unchecked_transaction().map_err(sql)?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL,
                data TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                content_sha256 TEXT NOT NULL, PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, amount TEXT NOT NULL,
                state TEXT NOT NULL, actual TEXT);",
        )
        .map_err(sql)?;
        if let Some(version) = store.meta::<u32>("schema_version")? {
            if version != 1 && version != 2 {
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
            let occupied: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM metadata) OR EXISTS(SELECT 1 FROM records) OR EXISTS(SELECT 1 FROM reservations)",
                [], |row| row.get(0),
            ).map_err(sql)?;
            if occupied {
                return Err(AppError::State("ecosystem schema version is missing from nonempty state; refusing to adopt its records or authority".into()));
            }
            let legacy_events: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='events')",
                [], |row| row.get(0),
            ).map_err(sql)?;
            if legacy_events {
                let occupied: bool = tx
                    .query_row("SELECT EXISTS(SELECT 1 FROM events)", [], |row| row.get(0))
                    .map_err(sql)?;
                if occupied {
                    return Err(AppError::State(
                        "ecosystem schema version is missing from an existing event history".into(),
                    ));
                }
                tx.execute_batch("DROP TABLE events;").map_err(sql)?;
            }
            let has_digest: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM pragma_table_info('records') WHERE name='content_sha256')",
                [], |row| row.get(0),
            ).map_err(sql)?;
            if !has_digest {
                tx.execute_batch(
                    "ALTER TABLE records ADD COLUMN content_sha256 TEXT NOT NULL DEFAULT '';",
                )
                .map_err(sql)?;
            }
            for (key, value) in [
                ("schema_version", json!(2)),
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
        }
        if store.meta::<u32>("schema_version")? == Some(1) {
            tx.execute_batch("ALTER TABLE records ADD COLUMN content_sha256 TEXT NOT NULL DEFAULT '';
                INSERT INTO records(kind,id,data,created_at,updated_at,content_sha256)
                    SELECT 'event','event-'||id,json_object('id','event-'||id,'initiative_id',initiative_id,
                        'kind',kind,'detail',detail,'created_at',created_at),created_at,created_at,''
                    FROM events ORDER BY id;
                DROP TABLE events;
                UPDATE metadata SET value='2' WHERE key='schema_version';").map_err(sql)?;
        }
        tx.execute_batch(
            "CREATE INDEX IF NOT EXISTS observations_source_time ON records(json_extract(data,'$.source'),created_at DESC,id DESC) WHERE kind='observation';
             CREATE INDEX IF NOT EXISTS records_initiative_time ON records(kind,json_extract(data,'$.initiative_id'),created_at,id);
             CREATE INDEX IF NOT EXISTS records_kind_sequence ON records(kind);
             CREATE INDEX IF NOT EXISTS records_initiative_sequence ON records(json_extract(data,'$.initiative_id'));
             CREATE INDEX IF NOT EXISTS records_state_time ON records(kind,json_extract(data,'$.state'),created_at,id);"
        ).map_err(sql)?;
        store.set_meta("policy_sequence", &identity.policy_sequence)?;
        if start_paused {
            store.set_meta("paused", &true)?;
        }
        tx.commit().map_err(sql)?;
        store.put(
            "owner_launch",
            &uuid::Uuid::new_v4().to_string(),
            &json!({"identity":identity,"admissions_paused":store.paused()?}),
            None,
            "Recorded configured owner identity; signed tool admission is checked separately",
        )?;
        Ok(store)
    }
}
