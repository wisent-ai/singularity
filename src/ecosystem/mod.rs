//! Autonomous portfolio direction. Product identity, execution and deployment remain with their owners.
mod control;
mod direction;
mod execution;
mod model;
mod observe;
mod outcomes;
mod runtime;
mod store;
use control::protocol;
use runtime::run;

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
    Opportunities(PageArgs),
    Initiatives(PageArgs),
    /// List bounded record summaries, newest inserts first.
    Records(RecordsArgs),
    /// Read a UTF-8 fragment; use next_offset and content_sha256 to continue.
    Record(RecordArgs),
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
    /// Pause admissions before opening control; recorded work still reconciles.
    #[arg(long)]
    pub start_paused: bool,
    /// Emit a flushed JSON event once local control is listening; dependencies may still be unavailable.
    #[arg(long)]
    pub ready_json: bool,
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
#[derive(Debug, Args)]
pub struct PageArgs {
    #[command(flatten)]
    pub client: ClientArgs,
    /// Continuation returned in next_cursor; omit to start a fresh listing.
    #[arg(long)]
    pub before: Option<i64>,
    #[arg(long, default_value_t = protocol::DEFAULT_PAGE_SIZE)]
    pub limit: u32,
}
#[derive(Debug, Args)]
pub struct RecordsArgs {
    pub kind: Option<String>,
    #[arg(long)]
    pub initiative_id: Option<String>,
    #[command(flatten)]
    pub page: PageArgs,
}
#[derive(Debug, Args)]
pub struct RecordArgs {
    pub kind: String,
    pub id: String,
    #[arg(long, default_value_t = 0)]
    pub offset: u64,
    #[arg(long, default_value_t = protocol::DEFAULT_RECORD_BYTES)]
    pub bytes: u32,
    /// The first fragment's content_sha256; required after offset zero.
    #[arg(long)]
    pub revision: Option<String>,
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
        Ok(!state.store.paused()?
            && state
                .store
                .meta::<bool>("las_catalog_ready")?
                .unwrap_or(false))
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
    let (method, client, params) = match args.command {
        EcosystemCommand::Run(args) => return run(args, cancellation).await,
        EcosystemCommand::Status(args) => ("status", args, json!({})),
        EcosystemCommand::Opportunities(args) => (
            "opportunities",
            args.client,
            json!({"before":args.before,"limit":args.limit}),
        ),
        EcosystemCommand::Initiatives(args) => (
            "initiatives",
            args.client,
            json!({"before":args.before,"limit":args.limit}),
        ),
        EcosystemCommand::Records(args) => (
            "records",
            args.page.client,
            json!({"kind":args.kind,"initiative_id":args.initiative_id,"before":args.page.before,"limit":args.page.limit}),
        ),
        EcosystemCommand::Record(args) => (
            "record",
            args.client,
            json!({"kind":args.kind,"id":args.id,"offset":args.offset,"bytes":args.bytes,"revision":args.revision}),
        ),
        EcosystemCommand::Explain(args) => ("explain", args.client, json!({"id":args.id})),
        EcosystemCommand::Pause(args) => ("pause", args, json!({})),
        EcosystemCommand::Resume(args) => ("resume", args, json!({})),
    };
    let response = match control::request(&client.state_dir, method, &params).await {
        Ok(response) => response,
        Err(error) => {
            if client.json {
                println!(
                    "{}",
                    json!({"schema_version":protocol::SCHEMA_VERSION,"ok":false,"error":{"code":"control_request_failed","operation":format!("ecosystem.{method}"),"message":error.to_string(),"retryable":false}})
                );
            }
            return Err(error);
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(if client.json || response["ok"] != true {
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
