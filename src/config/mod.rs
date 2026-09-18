use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use rust_decimal::Decimal;

/// A stimulus is at most 64 KiB; an identity component at most 128 bytes; a digest is
/// 64 hex characters; a secret file may grant the group and others no mode bits.
const MAX_STIMULUS_BYTES: usize = 64 * 1024;
const MAX_IDENTITY_COMPONENT_BYTES: usize = 128;
const DIGEST_HEX_CHARS: usize = 64;
pub const GROUP_OR_OTHER_ACCESS: u32 = 0o077;

#[derive(Debug, Parser)]
#[command(
    name = "singularity",
    version,
    about = "Autonomous Wisent agent runtime in Rust"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run(CycleArgs),
    Once(CycleArgs),
    Doctor(CommonArgs),
    Tools(ToolsArgs),
    /// Import existing memory, knowledge, and profile records into a being
    Import(ImportArgs),
    /// Show the first-use walkthrough
    Onboarding(OnboardingArgs),
}

#[derive(Debug, Clone, Args)]
pub struct OnboardingArgs {
    /// Discard recorded progress and evidence, then show the walkthrough from its first screen
    #[arg(long, default_value_t = false)]
    pub reset: bool,
    /// Import this file through the canonical state owner before showing the walkthrough
    #[arg(long, value_name = "JSON")]
    pub import_file: Option<PathBuf>,
    /// State directory of the being receiving imported records
    #[arg(long, env = "SINGULARITY_STATE_DIR", default_value = ".singularity")]
    pub state_dir: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ImportArgs {
    /// JSON document using schema singularity-mind-import-v1
    #[arg(long, value_name = "JSON")]
    pub file: PathBuf,
    /// State directory of the being receiving imported records
    #[arg(long, env = "SINGULARITY_STATE_DIR", default_value = ".singularity")]
    pub state_dir: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct ToolsArgs {
    #[arg(long, env = "SINGULARITY_AGENT_ID")]
    pub agent_id: Option<String>,
    #[arg(long, env = "LAS_COMMAND", default_value = "node")]
    pub las_command: String,
    #[arg(long, env = "LAS_MCP_ENTRYPOINT", default_value = "../las/src/mcp.mjs")]
    pub las_entrypoint: PathBuf,
    #[arg(
        long,
        env = "LAS_ONLY",
        default_value = "weles,skarbiec,tama,stado,lem,echo,most,probierz,byk,brama,warsztat,finance"
    )]
    pub las_only: String,
    #[arg(long, env = "LAS_SKIP")]
    pub las_skip: Option<String>,
    #[arg(long, env = "LAS_RELEASE_MANIFEST_FILE")]
    pub las_release_manifest: PathBuf,
    #[arg(long, env = "LAS_RELEASE_MANIFEST_SIGNATURE_FILE")]
    pub las_release_manifest_signature: PathBuf,
    #[arg(long, env = "LAS_RELEASE_TRUST_STORE_FILE")]
    pub las_release_trust_store: PathBuf,
    #[arg(long, env = "LAS_RELEASE_WATERMARK_FILE")]
    pub las_release_watermark: PathBuf,
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Json,
    Table,
}

#[derive(Debug, Clone, Args)]
pub struct CycleArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    /// Import existing mind records into a new state before its first cycle
    #[arg(long, value_name = "JSON")]
    pub import_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct CommonArgs {
    #[arg(long, env = "SINGULARITY_STIMULUS")]
    pub stimulus: Option<String>,
    #[arg(long, env = "SINGULARITY_AGENT_ID")]
    pub agent_id: String,
    #[arg(long, env = "SINGULARITY_AGENT_NAME", default_value = "MyAgent")]
    pub agent_name: String,
    #[arg(long, env = "SINGULARITY_AGENT_TICKER", default_value = "AGENT")]
    pub agent_ticker: String,
    #[arg(long, env = "SINGULARITY_AGENT_TYPE", default_value = "general")]
    pub agent_type: String,
    #[arg(long, env = "SINGULARITY_SPECIALTY", default_value = "general")]
    pub specialty: String,
    #[arg(long, env = "SINGULARITY_ROLE")]
    pub role: String,
    #[arg(long, env = "SINGULARITY_ENVIRONMENT")]
    pub environment: String,
    #[arg(long, env = "SINGULARITY_HOST")]
    pub host: String,
    #[arg(long, env = "SINGULARITY_WORKLOAD_ID")]
    pub workload_id: String,
    #[arg(long, env = "SINGULARITY_WORKLOAD_PUBLIC_KEY")]
    pub workload_public_key: String,
    #[arg(long, env = "SINGULARITY_EXECUTABLE_SHA256")]
    pub executable_digest: String,
    #[arg(long, env = "SINGULARITY_CODE_SHA256")]
    pub code_digest: String,
    #[arg(long, env = "SINGULARITY_POLICY_SHA256")]
    pub policy_digest: String,
    #[arg(long, env = "SINGULARITY_POLICY_SEQUENCE")]
    pub policy_sequence: u64,
    #[arg(long, env = "SINGULARITY_STARTING_BALANCE_USD", default_value = "10")]
    pub starting_balance: Decimal,
    #[arg(long, env = "SINGULARITY_INSTANCE_USD_PER_HOUR", default_value = "0")]
    pub instance_price: Decimal,
    #[arg(long, env = "BRAMA_INPUT_PRICE_USD_PER_MILLION", default_value = "0")]
    pub input_price: Decimal,
    #[arg(long, env = "BRAMA_OUTPUT_PRICE_USD_PER_MILLION", default_value = "0")]
    pub output_price: Decimal,
    #[arg(long, env = "SINGULARITY_CYCLE_INTERVAL_SECS", default_value = "5")]
    pub cycle_interval_secs: u64,
    #[arg(long, env = "SINGULARITY_MAX_TOOL_ROUNDS", default_value = "8")]
    pub max_tool_rounds: usize,
    #[arg(long, env = "SINGULARITY_STATE_DIR", default_value = ".singularity")]
    pub state_dir: PathBuf,
    #[arg(long, env = "SINGULARITY_WORKSPACE", default_value = ".")]
    pub workspace: PathBuf,
    #[arg(long, env = "SINGULARITY_RESUME", default_value = "false")]
    pub resume: bool,
    #[arg(long, env = "BRAMA_BASE_URL", default_value = "http://127.0.0.1:8081")]
    pub brama_url: String,
    #[arg(long, env = "BRAMA_MODEL", default_value = "any")]
    pub brama_model: String,
    #[arg(long, env = "BRAMA_HMAC_SECRET_FILE")]
    pub brama_secret_file: Option<PathBuf>,
    #[arg(long, env = "BRAMA_MAX_TOKENS", default_value = "2048")]
    pub max_tokens: u32,
    #[arg(long, env = "BRAMA_TEMPERATURE", default_value = "0.2")]
    pub temperature: f64,
    #[arg(long, env = "LAS_COMMAND", default_value = "node")]
    pub las_command: String,
    #[arg(long, env = "LAS_MCP_ENTRYPOINT", default_value = "../las/src/mcp.mjs")]
    pub las_entrypoint: PathBuf,
    #[arg(
        long,
        env = "LAS_ONLY",
        default_value = "weles,skarbiec,tama,stado,lem,echo,most,probierz,byk,brama,warsztat,finance"
    )]
    pub las_only: String,
    #[arg(long, env = "LAS_SKIP")]
    pub las_skip: Option<String>,
    #[arg(long, env = "LAS_RELEASE_MANIFEST_FILE")]
    pub las_release_manifest: PathBuf,
    #[arg(long, env = "LAS_RELEASE_MANIFEST_SIGNATURE_FILE")]
    pub las_release_manifest_signature: PathBuf,
    #[arg(long, env = "LAS_RELEASE_TRUST_STORE_FILE")]
    pub las_release_trust_store: PathBuf,
    #[arg(long, env = "LAS_RELEASE_WATERMARK_FILE")]
    pub las_release_watermark: PathBuf,
    #[arg(
        long,
        env = "SINGULARITY_REQUIRED_SURFACES",
        default_value = "skarbiec,finance"
    )]
    pub required_surfaces: String,
    #[arg(long, env = "MOST_BASE_URL", default_value = "http://127.0.0.1:8080")]
    pub most_url: String,
    #[arg(long, env = "MOST_SERVICE_TOKEN_FILE")]
    pub most_token_file: Option<PathBuf>,
    #[arg(long, env = "SINGULARITY_HTTP_TIMEOUT_SECS", default_value = "120")]
    pub http_timeout_secs: u64,
    #[arg(long, env = "SINGULARITY_MCP_TIMEOUT_SECS", default_value = "120")]
    pub mcp_timeout_secs: u64,
    #[arg(long, env = "SINGULARITY_SHUTDOWN_GRACE_SECS", default_value = "10")]
    pub shutdown_grace_secs: u64,
}

mod runtime;
mod validation;

pub use runtime::*;
pub use validation::*;
