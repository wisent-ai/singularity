use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use serde_json::Value;

use crate::domain::{AgentIdentity, Mission};
use crate::error::AppError;

const MAX_GOAL_BYTES: usize = 4096;

#[derive(Debug, Parser)]
#[command(
    name = "singularity",
    version,
    about = "Autonomous Wisent mission supervisor powered by Jeden"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run(CommonArgs),
    Once(CommonArgs),
    Doctor(JedenArgs),
    Tools(JedenArgs),
}

#[derive(Debug, Clone, Args)]
pub struct JedenArgs {
    #[arg(long, env = "JEDEN_COMMAND", default_value = "jeden")]
    pub jeden_command: PathBuf,
    #[arg(long, env = "SINGULARITY_WORKSPACE", default_value = ".")]
    pub workspace: PathBuf,
    #[arg(long, env = "SINGULARITY_LAS_SERVER", default_value = "las")]
    pub las_server: String,
    #[arg(
        long,
        env = "SINGULARITY_JEDEN_RPC_TIMEOUT_SECS",
        default_value = "300"
    )]
    pub rpc_timeout_secs: u64,
}

#[derive(Debug, Clone, Args)]
pub struct CommonArgs {
    #[command(flatten)]
    pub jeden: JedenArgs,
    #[arg(long, env = "SINGULARITY_GOAL")]
    pub goal: String,
    #[arg(long, env = "SINGULARITY_AGENT_NAME", default_value = "MyAgent")]
    pub agent_name: String,
    #[arg(long, env = "SINGULARITY_AGENT_TICKER", default_value = "AGENT")]
    pub agent_ticker: String,
    #[arg(long, env = "SINGULARITY_SPECIALTY", default_value = "general")]
    pub specialty: String,
    #[arg(long, env = "SINGULARITY_STATE_DIR", default_value = ".singularity")]
    pub state_dir: PathBuf,
    #[arg(long, env = "SINGULARITY_RESUME", default_value = "false")]
    pub resume: bool,
    #[arg(long, env = "SINGULARITY_MAX_CYCLES", default_value = "100")]
    pub max_cycles: u64,
    #[arg(long, env = "SINGULARITY_CYCLE_INTERVAL_SECS", default_value = "5")]
    pub cycle_interval_secs: u64,
    #[arg(long, env = "JEDEN_MODEL")]
    pub model: Option<String>,
    #[arg(long, env = "SINGULARITY_MAX_STEPS", default_value = "64")]
    pub max_steps: u32,
    #[arg(long, env = "SINGULARITY_ALLOW_WRITE", default_value = "false")]
    pub allow_write: bool,
    #[arg(long, env = "SINGULARITY_ALLOW_COMMAND", default_value = "false")]
    pub allow_command: bool,
    #[arg(long, env = "SINGULARITY_AUTO_APPROVE", default_value = "false")]
    pub auto_approve: bool,
}

pub struct RuntimeConfig {
    pub identity: AgentIdentity,
    pub mission: Mission,
    pub state_dir: PathBuf,
    pub resume: bool,
    pub resume_session: Option<PathBuf>,
    pub max_cycles: u64,
    pub cycle_interval: Duration,
    pub jeden_command: PathBuf,
    pub workspace: PathBuf,
    pub model: Option<String>,
    pub max_steps: u32,
    pub allow_write: bool,
    pub allow_command: bool,
    pub auto_approve: bool,
    pub las_server: String,
    pub rpc_timeout: Duration,
}

impl RuntimeConfig {
    pub fn from_args(args: &CommonArgs, resume_session: Option<PathBuf>) -> Result<Self, AppError> {
        let goal = normalized_goal(&args.goal)?;
        if args.max_cycles == 0 {
            return Err(AppError::Config("max cycles must be positive".into()));
        }
        if args.max_steps == 0 {
            return Err(AppError::Config("Jeden max steps must be positive".into()));
        }
        let workspace = canonical_workspace(&args.jeden.workspace)?;
        validate_las_config(&workspace, &args.jeden.las_server)?;
        let model = args
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let identity = AgentIdentity {
            name: nonempty(&args.agent_name, "agent name")?,
            ticker: nonempty(&args.agent_ticker, "agent ticker")?,
            specialty: nonempty(&args.specialty, "specialty")?,
        };
        let mission = Mission {
            goal,
            workspace: workspace.clone(),
            model: model.clone(),
            allow_write: args.allow_write,
            allow_command: args.allow_command,
            auto_approve: args.auto_approve,
            max_steps: args.max_steps,
        };
        Ok(Self {
            identity,
            mission,
            state_dir: args.state_dir.clone(),
            resume: args.resume,
            resume_session,
            max_cycles: args.max_cycles,
            cycle_interval: Duration::from_secs(args.cycle_interval_secs),
            jeden_command: args.jeden.jeden_command.clone(),
            workspace,
            model,
            max_steps: args.max_steps,
            allow_write: args.allow_write,
            allow_command: args.allow_command,
            auto_approve: args.auto_approve,
            las_server: args.jeden.las_server.clone(),
            rpc_timeout: Duration::from_secs(args.jeden.rpc_timeout_secs),
        })
    }
}

pub fn validate_jeden_args(args: &JedenArgs) -> Result<PathBuf, AppError> {
    if args.rpc_timeout_secs == 0 {
        return Err(AppError::Config(
            "Jeden RPC timeout must be positive".into(),
        ));
    }
    let workspace = canonical_workspace(&args.workspace)?;
    validate_las_config(&workspace, &args.las_server)?;
    Ok(workspace)
}

pub fn validate_las_config(workspace: &Path, server_name: &str) -> Result<(), AppError> {
    let server_name = server_name.trim();
    if server_name.is_empty() {
        return Err(AppError::Config(
            "Las MCP server name must not be empty".into(),
        ));
    }
    let mut server = None;
    let mut disabled = false;
    let mut paths = Vec::new();
    if let Some(home) = env::var_os("HOME") {
        paths.push(PathBuf::from(home).join(".jeden/mcp.json"));
    }
    paths.push(workspace.join(".jeden/mcp.json"));
    for path in paths {
        if !path.is_file() {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?).map_err(|error| {
            AppError::Config(format!(
                "invalid Jeden MCP configuration {}: {error}",
                path.display()
            ))
        })?;
        if let Some(candidate) = value
            .get("mcpServers")
            .and_then(Value::as_object)
            .and_then(|servers| servers.get(server_name))
        {
            server = Some(candidate.clone());
        }
        if value
            .get("disabledServers")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some(server_name))
            })
        {
            disabled = true;
        }
    }
    let server = server.ok_or_else(|| {
        AppError::Config(format!(
            "Jeden MCP server '{server_name}' is not configured in user or workspace scope"
        ))
    })?;
    if disabled || server.get("enabled").and_then(Value::as_bool) == Some(false) {
        return Err(AppError::Config(format!(
            "Jeden MCP server '{server_name}' is disabled"
        )));
    }
    if server.get("command").and_then(Value::as_str).is_none() {
        return Err(AppError::Config(format!(
            "Jeden MCP server '{server_name}' must be a local stdio command"
        )));
    }
    Ok(())
}

fn canonical_workspace(path: &Path) -> Result<PathBuf, AppError> {
    let canonical = fs::canonicalize(path).map_err(|error| {
        AppError::Config(format!(
            "workspace {} is unavailable: {error}",
            path.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(AppError::Config(format!(
            "workspace is not a directory: {}",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn normalized_goal(value: &str) -> Result<String, AppError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(AppError::Config("goal must not be empty".into()));
    }
    if value.len() > MAX_GOAL_BYTES {
        return Err(AppError::Config(format!(
            "goal must not exceed {MAX_GOAL_BYTES} bytes"
        )));
    }
    if value.contains('\0') {
        return Err(AppError::Config("goal must not contain NUL".into()));
    }
    Ok(value.to_owned())
}

fn nonempty(value: &str, label: &str) -> Result<String, AppError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(AppError::Config(format!("{label} must not be empty")));
    }
    Ok(value.to_owned())
}
