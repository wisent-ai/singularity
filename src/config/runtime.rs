//! The runtime configuration the command line arguments are turned into.
use std::path::PathBuf;
use std::time::Duration;

use rust_decimal::Decimal;
use secrecy::SecretString;
use url::Url;

use super::*;
use crate::domain::{AgentIdentity, Pricing};
use crate::error::AppError;

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
            .is_some_and(|value| value.len() > MAX_STIMULUS_BYTES || value.contains('\0'))
        {
            return Err(AppError::Config(format!(
                "stimulus must be at most {MAX_STIMULUS_BYTES} bytes and contain no NUL"
            )));
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
