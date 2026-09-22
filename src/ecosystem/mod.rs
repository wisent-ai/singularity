//! Autonomous portfolio direction. Product identity, execution and deployment remain with their owners.
mod control;
mod direction;
mod execution;
mod model;
mod observe;
mod outcomes;
mod store;
mod runtime;
use runtime::run;
use control::protocol;

use crate::config::CommonArgs;
use crate::{AppError, RuntimeConfig};
use clap::{Args, Subcommand};
use model::{Issue, Policy};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};
use store::Store;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Args)]
pub struct EcosystemArgs {
    #[command(subcommand)]
    pub command: EcosystemCommand,
}
#[derive(Debug, Subcommand)]
pub enum EcosystemCommand {
    /// Run the durable autonomous portfolio within a fixed delegated policy.
    Run(RunArgs),
    Status(ClientArgs),
    Opportunities(ClientArgs),
    Initiatives(ClientArgs),
    Explain(ExplainArgs),
    /// Stop admitting new work; retain and observe already dispatched operations.
    Pause(ClientArgs),
    Resume(ClientArgs),
}
#[derive(Debug, Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    /// Owner-approved policy whose SHA-256 equals --policy-digest.
    #[arg(long)]
    pub policy: PathBuf,
}
#[derive(Debug, Args)]
pub struct ClientArgs {
    #[arg(long, env = "SINGULARITY_STATE_DIR", default_value = ".singularity")]
    pub state_dir: PathBuf,
    #[arg(long)]
    pub json: bool,
}
#[derive(Debug, Args)]
pub struct ExplainArgs {
    pub id: String,
    #[command(flatten)]
    pub client: ClientArgs,
}

struct State {
    store: Store,
    policy: Policy,
    directory: PathBuf,
    config: Arc<RuntimeConfig>,
}
#[derive(Clone)]
struct Shared(Arc<Mutex<State>>);
impl Shared {
    fn lock(&self) -> Result<MutexGuard<'_, State>, AppError> {
        self.0.lock().map_err(|_| {
            AppError::State("ecosystem state owner failed while holding its lock".into())
        })
    }
    fn admission_open(&self) -> Result<bool, AppError> {
        let state = self.lock()?;
        Ok(!state.store.paused()? && state.store.meta::<bool>("las_catalog_ready")?.unwrap_or(false))
    }
    fn failure(&self, operation: &str, error: &AppError) -> Result<(), AppError> {
        let retryable = !matches!(
            error,
            AppError::Config(_)
                | AppError::Secret(_)
                | AppError::Brama {
                    class: crate::ErrorClass::Permanent,
                    ..
                }
        );
        self.lock()?.store.issue(&Issue {
            code: "operation_failed".into(),
            operation: operation.into(),
            message: error.to_string(),
            retryable,
        })
    }
}

pub async fn execute(args: EcosystemArgs, cancellation: CancellationToken) -> Result<(), AppError> {
    let (method, client, id) = match args.command {
        EcosystemCommand::Run(args) => return run(args, cancellation).await,
        EcosystemCommand::Status(args) => ("status", args, None),
        EcosystemCommand::Opportunities(args) => ("opportunities", args, None),
        EcosystemCommand::Initiatives(args) => ("initiatives", args, None),
        EcosystemCommand::Explain(args) => ("explain", args.client, Some(args.id)),
        EcosystemCommand::Pause(args) => ("pause", args, None),
        EcosystemCommand::Resume(args) => ("resume", args, None),
    };
    let response = match control::request(&client.state_dir, method, id.as_deref()).await {
        Ok(response) => response,
        Err(error) => {
            if client.json {
                println!(
                    "{}",
                    json!({"schema_version":protocol::SCHEMA_VERSION,"ok":false,"error":{"code":"owner_unavailable","operation":method,"message":error.to_string(),"retryable":true}})
                );
            }
            return Err(error);
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(if client.json {
            &response
        } else {
            &response["result"]
        })?
    );
    if response["ok"] != true {
        return Err(AppError::State(
            response["error"]["message"]
                .as_str()
                .unwrap_or("ecosystem refused the operation")
                .into(),
        ));
    }
    Ok(())
}

