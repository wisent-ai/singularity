mod constants;
mod process;
use super::{
    Shared,
    model::{Issue, Observation, Source},
};
use crate::AppError;
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use secrecy::ExposeSecret;
use tokio::process::Command;

pub async fn text(program: &str, args: &[&str]) -> Result<String, AppError> {
    let mut command = Command::new(program);
    command.args(args);
    process::text(command, &format!("{program} {}", args.join(" "))).await
}

pub async fn command(program: &str, args: &[&str]) -> Result<Value, AppError> {
    serde_json::from_str(&text(program, args).await?).map_err(|e| {
        AppError::Runtime(format!(
            "{program} {}: successful exit but invalid JSON: {e}",
            args.join(" ")
        ))
    })
}

pub async fn products(shared: &Shared, args: &[&str]) -> Result<Value, AppError> {
    let root = shared.lock()?.policy.workspace_root.clone();
    let catalog = root.join("wisent-products/catalog/products.yml");
    let mut child = Command::new("wisent-products");
    child.arg("--catalog").arg(&catalog).args(args).env("WISENT_WORKSPACE", &root);
    let operation = format!("wisent-products --catalog {} {}", catalog.display(), args.join(" "));
    let output = process::text(child, &operation).await?;
    serde_json::from_str(&output).map_err(|error| AppError::Runtime(format!("{operation}: invalid JSON: {error}")))
}

pub async fn jeden(shared: &Shared, args: &[&str]) -> Result<Value, AppError> {
    let (config, directory) = {
        let state = shared.lock()?;
        (state.config.clone(), state.directory.join("jeden"))
    };
    let mut child = Command::new("jeden");
    child.args(args)
        .env("BRAMA_URL", config.brama_url.as_str())
        .env("BRAMA_TOKEN", config.brama_bearer.expose_secret())
        .env("WISENT_APP_AGENT_AUTH_SECRET", config.brama_secret.expose_secret())
        .env("WISENT_APP_AGENT_ID", &config.identity.agent_id)
        .env("JEDEN_MODEL", &config.brama_model)
        .env("JEDEN_PURSUIT_STATE_ROOT", directory);
    let operation = format!("jeden {}", args.join(" "));
    let output = process::text(child, &operation).await?;
    serde_json::from_str(&output).map_err(|error| AppError::Runtime(format!("{operation}: invalid JSON: {error}")))
}

pub async fn monitor(shared: Shared, source: Source) -> Result<(), AppError> {
    let interval = shared.lock()?.policy.observation_interval_seconds;
    let mut cadence = tokio::time::interval(std::time::Duration::from_secs(interval));
    cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        cadence.tick().await;
        let operation = format!("observe.{source:?}");
        let previous = shared.lock()?.store.meta::<DateTime<Utc>>(&operation)?;
        if previous.is_some_and(|at| (Utc::now()-at).num_seconds()<interval as i64) {
            continue;
        }
        let (program, args): (&str, &[&str]) = match source {
            Source::ProductCatalog => ("wisent-products", &["catalog", "--json"]),
            Source::ProductAnalytics => ("echo-cli", &["analytics", "7"]),
            Source::MarketResearch => ("echo-cli", &["market"]),
            Source::OperatorDecisions => ("oko-cli", &["transcripts", "tasks", "--open", "--read-only", "--json"]),
            Source::FleetServices => ("stado", &["service", "list", "--json"]),
        };
        let result = match source {
            Source::ProductCatalog => products(&shared, args).await,
            _ => command(program, args).await,
        };
        let state = shared.lock()?;
        match result {
            Ok(content) => {
                let observation = Observation {
                    id: format!("observation-{}", uuid::Uuid::new_v4()),
                    source: source.clone(),
                    observed_at: Utc::now(),
                    source_revision: None,
                    content_sha256: hex::encode(Sha256::digest(serde_json::to_vec(&content)?)),
                    content,
                };
                state.store.put("observation", &observation.id, &observation, None,
                    &format!("Observed {program} {}", args.join(" ")))?;
                state.store.clear_issue(&operation)?;
            }
            Err(error) => state.store.issue(&Issue {
                code: "observation_failed".into(), operation: operation.clone(),
                message: error.to_string(), retryable: true,
            })?,
        }
        state.store.set_meta(&operation, &Utc::now())?;
    }
}

pub fn facts(shared: &Shared) -> Result<Value, AppError> {
    let state = shared.lock()?;
    let latest = state.store.latest_observations(&state.policy.sources)?;
    Ok(
        json!({"observations":latest,"opportunities":state.store.list::<Value>("opportunity")?,
        "initiatives":state.store.list::<Value>("initiative")?,"outcomes":state.store.list::<Value>("outcome")?,
        "status":state.store.status(&state.policy)?}),
    )
}
