use super::{AppError, Policy, Result, Store, fleet, meta_text, parsed, set_meta_text, sql};
use serde_json::{Value, json};
use std::path::Path;

/// Tables and indexes of the ecosystem store; `data` stays text so a record's
/// digest and byte offsets are those of the JSON the being wrote.
const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS ecosystem_metadata (being TEXT NOT NULL, key TEXT NOT NULL,
        value TEXT NOT NULL, PRIMARY KEY(being,key));
    CREATE TABLE IF NOT EXISTS ecosystem_records (seq BIGSERIAL UNIQUE, being TEXT NOT NULL,
        kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, PRIMARY KEY(being,kind,id));
    CREATE TABLE IF NOT EXISTS ecosystem_reservations (being TEXT NOT NULL, id TEXT NOT NULL,
        amount TEXT NOT NULL, state TEXT NOT NULL, actual TEXT, PRIMARY KEY(being,id));
    CREATE INDEX IF NOT EXISTS ecosystem_observations_source_time ON ecosystem_records
        (being,(data::jsonb->>'source'),created_at DESC,id DESC) WHERE kind='observation';
    CREATE INDEX IF NOT EXISTS ecosystem_records_initiative_time ON ecosystem_records
        (being,kind,(data::jsonb->>'initiative_id'),created_at,id);
    CREATE INDEX IF NOT EXISTS ecosystem_records_state_time ON ecosystem_records
        (being,kind,(data::jsonb->>'state'),created_at,id);
    CREATE INDEX IF NOT EXISTS ecosystem_records_sequence ON ecosystem_records(being,seq);";

impl Store {
    pub fn open(
        directory: &Path,
        policy: &Policy,
        identity: &crate::AgentIdentity,
        start_paused: bool,
    ) -> Result<Self> {
        let legacy = directory.join("ecosystem.sqlite3");
        if legacy.exists() {
            return Err(AppError::State(format!(
                "{} is a private ecosystem store from before the fleet database; this being's state now lives in the fleet database singularity, and a private file beside it would split its history",
                legacy.display()
            )));
        }
        let owner = json!({"agent_id":identity.agent_id,"role":identity.role,
            "environment":identity.environment,"workload_id":identity.workload_id});
        let policy_value = serde_json::to_value(policy)?;
        let sequence = identity.policy_sequence;
        let being = identity.agent_id.clone();
        let db = fleet::Db::start()?;
        let scope = being.clone();
        db.run(move |client| {
            client.batch_execute(SCHEMA).map_err(sql)?;
            let mut tx = client.transaction().map_err(sql)?;
            let being = scope.as_str();
            if let Some(version) = parsed::<u32>(meta_text(&mut tx, being, "schema_version")?)? {
                if version != 2 {
                    return Err(AppError::State(format!("unsupported ecosystem schema {version}")));
                }
                let previous = parsed::<Value>(meta_text(&mut tx, being, "policy")?)?
                    .ok_or_else(|| AppError::State("ecosystem policy missing".into()))?;
                if previous != policy_value {
                    return Err(AppError::State("ecosystem authority differs from its persisted policy; authority cannot change on resume".into()));
                }
                if parsed::<Value>(meta_text(&mut tx, being, "owner")?)?.as_ref() != Some(&owner) {
                    return Err(AppError::State("ecosystem owner differs from its persisted principal; execution credentials cannot change owner on resume".into()));
                }
                if parsed::<u64>(meta_text(&mut tx, being, "policy_sequence")?)?
                    .is_none_or(|previous| previous > sequence)
                {
                    return Err(AppError::State(
                        "ecosystem policy sequence is missing or would roll back".into(),
                    ));
                }
            } else {
                let occupied: bool = tx
                    .query_one(
                        "SELECT EXISTS(SELECT 1 FROM ecosystem_metadata WHERE being=$1)
                            OR EXISTS(SELECT 1 FROM ecosystem_records WHERE being=$1)
                            OR EXISTS(SELECT 1 FROM ecosystem_reservations WHERE being=$1)",
                        &[&being],
                    )
                    .map_err(sql)?
                    .get(0);
                if occupied {
                    return Err(AppError::State("ecosystem schema version is missing from nonempty state; refusing to adopt its records or authority".into()));
                }
                for (key, value) in [
                    ("schema_version", json!(2)),
                    ("policy", policy_value),
                    ("owner", owner),
                    ("policy_sequence", json!(sequence)),
                    ("paused", json!(false)),
                    ("spent_usd", json!("0")),
                    ("reserved_usd", json!("0")),
                ] {
                    tx.execute(
                        "INSERT INTO ecosystem_metadata(being,key,value) VALUES ($1,$2,$3)",
                        &[&being, &key, &value.to_string()],
                    )
                    .map_err(sql)?;
                }
            }
            set_meta_text(&mut tx, being, "policy_sequence", &json!(sequence).to_string())?;
            if start_paused {
                set_meta_text(&mut tx, being, "paused", "true")?;
            }
            tx.commit().map_err(sql)
        })?;
        let store = Self { db, being };
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
