use super::super::Shared;
use crate::{AppError, LasSupervisor, RuntimeConfig};
use serde_json::json;
use std::sync::Arc;

/// Verify the signed catalogue independently of observation readers and execution reconciliation.
pub(in crate::ecosystem) async fn monitor(shared: Shared, config: Arc<RuntimeConfig>) -> Result<(), AppError> {
    let mut supervisor: Option<LasSupervisor> = None;
    let mut cadence = tokio::time::interval(config.cycle_interval);
    cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    shared.lock()?.store.set_meta("las_catalog_ready", &false)?;
    loop {
        cadence.tick().await;
        let result = match supervisor.as_mut() {
            Some(client) => client.refresh(&config.required_surfaces).await,
            None => match LasSupervisor::spawn(
                &config.las_command, &config.las_entrypoint, &config.las_only,
                config.las_skip.as_deref(), Some(&config.identity.agent_id),
                &config.las_release_manifest, &config.las_release_manifest_signature,
                &config.las_release_trust_store, &config.las_release_watermark,
                &config.required_surfaces,
            ).await {
                Ok(client) => { supervisor = Some(client); Ok(()) }
                Err(error) => Err(error),
            },
        };
        match result {
            Ok(()) => {
                let state = shared.lock()?;
                state.store.set_meta("las_catalog_ready", &true)?;
                state.store.clear_issue("las.catalog")?;
                state.store.put("dependency", "las", &json!({"catalog_ready":true,
                    "observed_at":chrono::Utc::now(), "tool_count":supervisor.as_ref().map(|client| client.tools().len())}),
                    None, "Read verified signed Las catalogue; this is not proof of a product operation")?;
            }
            Err(error) => {
                shared.lock()?.store.set_meta("las_catalog_ready", &false)?;
                shared.failure("las.catalog", &error)?;
                if let Some(mut client) = supervisor.take() {
                    if let Err(error) = client.shutdown(config.shutdown_grace).await {
                        shared.failure("las.shutdown", &error)?;
                    }
                }
            }
        }
    }
}
