//! The runtime configuration the command line arguments are turned into.
use std::path::PathBuf;
use std::time::Duration;

use rust_decimal::Decimal;
use secrecy::SecretString;
use url::Url;

use super::*;
use crate::domain::AgentIdentity;
use crate::error::AppError;

pub struct RuntimeConfig {
    pub identity: AgentIdentity,
    pub stimulus: Option<String>,
    pub starting_balance: Decimal,
    /// The host's stated hourly price; model calls are priced by Brama's catalog.
    pub instance_price: Decimal,
    pub cycle_interval: Duration,
    pub workspace: PathBuf,
    pub state_dir: PathBuf,
    pub resume: bool,
    pub brama_url: Url,
    pub brama_model: String,
    pub brama_secret: SecretString,
    pub brama_bearer: SecretString,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub las_command: String,
    pub las_entrypoint: PathBuf,
    pub las_only: String,
    pub las_skip: Option<String>,
    pub las_release_manifest: PathBuf,
    pub las_release_manifest_signature: PathBuf,
    pub las_release_trust_store: PathBuf,
    pub las_release_watermark: PathBuf,
    /// Most, when a Most credential is configured; its URL is then required.
    pub most: Option<MostEndpoint>,
    pub required_surfaces: Vec<String>,
    pub shutdown_grace: Duration,
}

/// Where Most is reached and the service credential that reaches it.
pub struct MostEndpoint {
    pub url: Url,
    pub token: SecretString,
}

impl RuntimeConfig {
    pub fn from_args(args: &CommonArgs) -> Result<Self, AppError> {
        let inherited = crate::bootstrap::inherited_credentials()?;
        let stimulus = args
            .stimulus
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        if stimulus
            .as_deref()
            .is_some_and(|value| value.contains('\0'))
        {
            return Err(AppError::Config("stimulus must contain no NUL".into()));
        }
        let workspace = std::fs::canonicalize(&args.workspace)
            .map_err(|error| AppError::Config(format!("workspace: {error}")))?;
        if !workspace.is_dir() {
            return Err(AppError::Config("workspace must be a directory".into()));
        }
        if args.starting_balance.is_sign_negative() || args.instance_price.is_sign_negative() {
            return Err(AppError::Config(
                "prices and balance cannot be negative".into(),
            ));
        }
        if args
            .temperature
            .is_some_and(|temperature| temperature.is_sign_negative() || !temperature.is_finite())
        {
            return Err(AppError::Config(
                "temperature must be finite and not negative".into(),
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
            instance_price: args.instance_price,
            cycle_interval: Duration::from_secs(args.cycle_interval_secs),
            workspace,
            state_dir: args.state_dir.clone(),
            resume: args.resume,
            brama_url: parse_http_url(&args.brama_url, "BRAMA_BASE_URL")?,
            brama_model: args.brama_model.clone(),
            brama_secret: match (args.brama_secret_file.as_ref(), inherited) {
                (Some(path), _) => read_secret(path)?,
                (None, Some(credentials)) => credentials.brama.clone(),
                (None, None) => SecretString::from(std::env::var("WISENT_APP_AGENT_AUTH_SECRET").map_err(
                    |_| {
                        AppError::Secret(
                            "BRAMA_HMAC_SECRET_FILE or WISENT_APP_AGENT_AUTH_SECRET is required"
                                .into(),
                        )
                    },
                )?),
            },
            brama_bearer: match (args.brama_bearer_file.as_ref(), inherited) {
                (Some(path), _) => read_secret(path)?,
                (None, Some(credentials)) => credentials.bearer.clone(),
                (None, None) => return Err(AppError::Secret(
                    "BRAMA_BEARER_TOKEN_FILE or a bootstrap handoff is required: an agent signature does not replace Brama caller authorization".into()
                )),
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
            most: match (
                args.most_url.as_deref(),
                match (args.most_token_file.as_ref(), inherited) {
                    (Some(path), _) => Some(read_secret(path)?),
                    (None, Some(credentials)) => Some(credentials.most.clone()),
                    (None, None) => None,
                },
            ) {
                (Some(url), Some(token)) => Some(MostEndpoint {
                    url: parse_http_url(url, "MOST_BASE_URL")?,
                    token,
                }),
                (None, Some(_)) => {
                    return Err(AppError::Config(
                        "a Most credential is configured but MOST_BASE_URL (--most-url) is not: \
                         no Most address is assumed"
                            .into(),
                    ));
                }
                (_, None) => None,
            },
            shutdown_grace: Duration::from_secs(args.shutdown_grace_secs),
        })
    }
}
