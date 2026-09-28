use super::fleet::{metadata, record, reservation};
use super::{AppError, Policy, Result, Store, fleet, meta_text, parsed, set_meta_text, sql};
use sea_orm::{ActiveValue::Set, ColumnTrait, EntityTrait, QueryFilter, TransactionTrait};
use serde_json::{Value, json};
use std::path::Path;

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
        db.run(move |db| async move {
            let tx = db.begin().await.map_err(sql)?;
            let being = scope.as_str();
            if let Some(version) = parsed::<u32>(meta_text(&tx, being, "schema_version").await?)? {
                if version != 2 {
                    return Err(AppError::State(format!("unsupported ecosystem schema {version}")));
                }
                let previous = parsed::<Value>(meta_text(&tx, being, "policy").await?)?
                    .ok_or_else(|| AppError::State("ecosystem policy missing".into()))?;
                if previous != policy_value {
                    return Err(AppError::State("ecosystem authority differs from its persisted policy; authority cannot change on resume".into()));
                }
                if parsed::<Value>(meta_text(&tx, being, "owner").await?)?.as_ref() != Some(&owner) {
                    return Err(AppError::State("ecosystem owner differs from its persisted principal; execution credentials cannot change owner on resume".into()));
                }
                if parsed::<u64>(meta_text(&tx, being, "policy_sequence").await?)?
                    .is_none_or(|previous| previous > sequence)
                {
                    return Err(AppError::State(
                        "ecosystem policy sequence is missing or would roll back".into(),
                    ));
                }
            } else {
                let occupied = metadata::Entity::find()
                    .filter(metadata::Column::Being.eq(being))
                    .one(&tx)
                    .await
                    .map_err(sql)?
                    .is_some()
                    || record::Entity::find()
                        .filter(record::Column::Being.eq(being))
                        .one(&tx)
                        .await
                        .map_err(sql)?
                        .is_some()
                    || reservation::Entity::find()
                        .filter(reservation::Column::Being.eq(being))
                        .one(&tx)
                        .await
                        .map_err(sql)?
                        .is_some();
                if occupied {
                    return Err(AppError::State("ecosystem schema version is missing from nonempty state; refusing to adopt its records or authority".into()));
                }
                let rows = [
                    ("schema_version", json!(2)),
                    ("policy", policy_value),
                    ("owner", owner),
                    ("policy_sequence", json!(sequence)),
                    ("paused", json!(false)),
                    ("spent_usd", json!("0")),
                    ("reserved_usd", json!("0")),
                ]
                .into_iter()
                .map(|(key, value)| metadata::ActiveModel {
                    being: Set(being.to_owned()),
                    key: Set(key.to_owned()),
                    value: Set(value.to_string()),
                });
                metadata::Entity::insert_many(rows)
                    .exec_without_returning(&tx)
                    .await
                    .map_err(sql)?;
            }
            set_meta_text(&tx, being, "policy_sequence", &json!(sequence).to_string()).await?;
            if start_paused {
                set_meta_text(&tx, being, "paused", "true").await?;
            }
            tx.commit().await.map_err(sql)
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
