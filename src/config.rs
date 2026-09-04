use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};
use rust_decimal::Decimal;
use secrecy::SecretString;
use url::Url;

use crate::domain::{AgentIdentity, Pricing};
use crate::error::AppError;

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
    Run(CommonArgs),
    Once(CommonArgs),
    Doctor(CommonArgs),
    Tools(ToolsArgs),
    /// Show the first-use walkthrough
    Onboarding(OnboardingArgs),
}

#[derive(Debug, Clone, Args)]
pub struct OnboardingArgs {
    /// Discard recorded progress and evidence, then show the walkthrough from its first screen
    #[arg(long, default_value_t = false)]
    pub reset: bool,
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

pub struct RuntimeConfig {
    pub identity: AgentIdentity,
    pub stimulus: Option<String>,
    pub starting_balance: Decimal,
    pub pricing: Pricing,
    pub cycle_interval: Duration,
    pub max_tool_rounds: usize,
    pub workspace: PathBuf,
    pub state_dir: PathBuf,
    pub resume: bool,
    pub brama_url: Url,
    pub brama_model: String,
    pub brama_secret: SecretString,
    pub max_tokens: u32,
    pub temperature: f64,
    pub las_command: String,
    pub las_entrypoint: PathBuf,
    pub las_only: String,
    pub las_skip: Option<String>,
    pub las_release_manifest: PathBuf,
    pub las_release_manifest_signature: PathBuf,
    pub las_release_trust_store: PathBuf,
    pub las_release_watermark: PathBuf,
    pub most_token: Option<SecretString>,
    pub required_surfaces: Vec<String>,
    pub most_url: Url,
    pub http_timeout: Duration,
    pub mcp_timeout: Duration,
    pub shutdown_grace: Duration,
}

impl RuntimeConfig {
    pub fn from_args(args: &CommonArgs) -> Result<Self, AppError> {
        let stimulus = args
            .stimulus
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        if stimulus
            .as_deref()
            .is_some_and(|value| value.len() > 64 * 1024 || value.contains('\0'))
        {
            return Err(AppError::Config(
                "stimulus must be at most 65536 bytes and contain no NUL".into(),
            ));
        }
        let workspace = std::fs::canonicalize(&args.workspace)
            .map_err(|error| AppError::Config(format!("workspace: {error}")))?;
        if !workspace.is_dir() {
            return Err(AppError::Config("workspace must be a directory".into()));
        }
        if args.max_tool_rounds == usize::default() {
            return Err(AppError::Config("max tool rounds must be positive".into()));
        }
        if args.starting_balance.is_sign_negative()
            || args.input_price.is_sign_negative()
            || args.output_price.is_sign_negative()
            || args.instance_price.is_sign_negative()
        {
            return Err(AppError::Config(
                "prices and balance cannot be negative".into(),
            ));
        }
        let max_temperature: f64 = "2".parse().expect("static temperature is valid");
        if args.temperature.is_sign_negative()
            || args.temperature > max_temperature
            || !args.temperature.is_finite()
        {
            return Err(AppError::Config(
                "temperature must be finite and between zero and two".into(),
            ));
        }
        if !args.las_entrypoint.is_file() {
            return Err(AppError::Config(format!(
                "LAS entrypoint not found: {}",
                args.las_entrypoint.display()
            )));
        }
        let required_surfaces = parse_csv(&args.required_surfaces);
        let las_only = args.las_only.clone();
        for (path, label) in [
            (&args.las_release_manifest, "LAS release manifest"),
            (
                &args.las_release_manifest_signature,
                "LAS release manifest signature",
            ),
            (&args.las_release_trust_store, "LAS release trust store"),
        ] {
            if !path.is_absolute() || !path.is_file() {
                return Err(AppError::Config(format!(
                    "{label} must be an absolute regular file"
                )));
            }
        }
        if !args.las_release_watermark.is_absolute() {
            return Err(AppError::Config(
                "LAS release watermark must be an absolute path".into(),
            ));
        }
        validate_identity_component(&args.agent_id, "agent id")?;
        validate_identity_component(&args.role, "role")?;
        validate_identity_component(&args.environment, "environment")?;
        validate_identity_component(&args.host, "host")?;
        validate_identity_component(&args.workload_id, "workload id")?;
        validate_digest(&args.workload_public_key, "workload public key")?;
        validate_digest(&args.executable_digest, "executable digest")?;
        validate_digest(&args.code_digest, "code digest")?;
        validate_digest(&args.policy_digest, "policy digest")?;
        Ok(Self {
            identity: AgentIdentity {
                agent_id: args.agent_id.clone(),
                name: args.agent_name.clone(),
                ticker: args.agent_ticker.clone(),
                agent_type: args.agent_type.clone(),
                specialty: args.specialty.clone(),
                role: args.role.clone(),
                environment: args.environment.clone(),
                host: args.host.clone(),
                workload_id: args.workload_id.clone(),
                workload_public_key: args.workload_public_key.clone(),
                executable_digest: args.executable_digest.clone(),
                code_digest: args.code_digest.clone(),
                policy_digest: args.policy_digest.clone(),
                policy_sequence: args.policy_sequence,
            },
            stimulus,
            starting_balance: args.starting_balance,
            pricing: Pricing {
                input_per_million: args.input_price,
                output_per_million: args.output_price,
                instance_per_hour: args.instance_price,
            },
            cycle_interval: Duration::from_secs(args.cycle_interval_secs),
            max_tool_rounds: args.max_tool_rounds,
            workspace,
            state_dir: args.state_dir.clone(),
            resume: args.resume,
            brama_url: parse_http_url(&args.brama_url, "BRAMA_BASE_URL")?,
            brama_model: args.brama_model.clone(),
            brama_secret: match args.brama_secret_file.as_ref() {
                Some(path) => read_secret(path)?,
                None => SecretString::from(std::env::var("WISENT_APP_AGENT_AUTH_SECRET").map_err(
                    |_| {
                        AppError::Secret(
                            "BRAMA_HMAC_SECRET_FILE or WISENT_APP_AGENT_AUTH_SECRET is required"
                                .into(),
                        )
                    },
                )?),
            },
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            las_command: args.las_command.clone(),
            las_entrypoint: args.las_entrypoint.clone(),
            las_only,
            las_skip: args.las_skip.clone(),
            las_release_manifest: args.las_release_manifest.clone(),
            las_release_manifest_signature: args.las_release_manifest_signature.clone(),
            las_release_trust_store: args.las_release_trust_store.clone(),
            las_release_watermark: args.las_release_watermark.clone(),
            required_surfaces,
            most_url: parse_http_url(&args.most_url, "MOST_BASE_URL")?,
            most_token: args.most_token_file.as_ref().map(read_secret).transpose()?,
            http_timeout: Duration::from_secs(args.http_timeout_secs),
            mcp_timeout: Duration::from_secs(args.mcp_timeout_secs),
            shutdown_grace: Duration::from_secs(args.shutdown_grace_secs),
        })
    }
}

fn validate_identity_component(value: &str, label: &str) -> Result<(), AppError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(AppError::Config(format!(
            "{label} is not a valid immutable identifier"
        )));
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), AppError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(AppError::Config(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}
pub fn parse_http_url(value: &str, name: &str) -> Result<Url, AppError> {
    let url = Url::parse(value).map_err(|error| AppError::Config(format!("{name}: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(AppError::Config(format!("{name} must use http or https")));
    }
    Ok(url)
}

pub fn read_secret(path: &PathBuf) -> Result<SecretString, AppError> {
    if !path.is_file() {
        return Err(AppError::Secret(format!(
            "not a regular file: {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let forbidden: u32 = "63".parse().expect("static permission mask is valid");
        if fs::metadata(path)?.permissions().mode() & forbidden != u32::default() {
            return Err(AppError::Secret(format!(
                "{} must not be group/world accessible",
                path.display()
            )));
        }
    }
    let value = fs::read_to_string(path)?
        .trim_end_matches(['\r', '\n'])
        .to_owned();
    if value.is_empty() {
        return Err(AppError::Secret(format!("{} is empty", path.display())));
    }
    Ok(SecretString::from(value))
}
