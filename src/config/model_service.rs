//! Where cognition's model calls go. A launch that names Brama's address
//! (`BRAMA_BASE_URL`) uses it; one that names none never dials Brama itself:
//! Stado's capability resolution (`stado capabilities resolve inference
//! --consumer singularity`) hands out Brama's route, or the OpenAI-compatible
//! endpoint the operator declared in its place, with its model and the role
//! of the item holding its key. When no alternative answers, the refusal
//! names each one with its route and cause.
use std::process::Command;

use secrecy::SecretString;
use serde_json::Value;
use url::Url;

use super::{parse_http_url, secret_value};
use crate::error::AppError;

/// The consumer a being asks Stado's capability resolution as.
const CONSUMER: &str = "singularity";

/// The model service and the credentials only it may receive.
#[derive(Clone)]
pub enum ModelService {
    /// Brama: every request carries the bearer and the agent's signature.
    Brama {
        url: Url,
        secret: SecretString,
        bearer: SecretString,
    },
    /// The endpoint Stado hands out in Brama's place: no agent signature,
    /// and its key only when its role names one.
    InPlace {
        provider: String,
        url: Url,
        key: Option<SecretString>,
    },
}

impl ModelService {
    pub fn url(&self) -> &Url {
        match self {
            Self::Brama { url, .. } | Self::InPlace { url, .. } => url,
        }
    }

    /// `brama`, or the provider Stado names for the endpoint in its place.
    pub fn provider(&self) -> &str {
        match self {
            Self::Brama { .. } => "brama",
            Self::InPlace { provider, .. } => provider,
        }
    }
}

/// What Stado's capability resolution answered.
pub(super) enum Resolved {
    /// Brama answers this being at the route.
    Brama(Url),
    /// Brama does not; this endpoint answers in its place, serving `model`.
    InPlace {
        provider: String,
        url: Url,
        model: String,
        key: Option<SecretString>,
    },
}

/// The executable answering Stado's commands: `STADO_BIN`, else `stado`.
fn stado() -> String {
    std::env::var("STADO_BIN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "stado".into())
}

/// Run one Stado command and hand back its stdout, or the refusal naming it.
fn run(words: &[&str]) -> Result<std::process::Output, AppError> {
    let binary = stado();
    stado_wait::output(Command::new(&binary).args(words)).map_err(|error| {
        AppError::Config(format!(
            "no BRAMA_BASE_URL (--brama-url) is named and `{binary} {}` could not run: {error}; name Brama's address or install Stado",
            words.join(" ")
        ))
    })
}

/// Ask Stado where this being's model calls go.
pub(super) fn resolve() -> Result<Resolved, AppError> {
    let words = [
        "capabilities",
        "resolve",
        "inference",
        "--consumer",
        CONSUMER,
        "--json",
    ];
    let output = run(&words)?;
    if !output.status.success() {
        let named = [&output.stdout, &output.stderr]
            .into_iter()
            .filter_map(|bytes| serde_json::from_slice::<Value>(bytes).ok())
            .find_map(|envelope| envelope["details"]["unreachable"].as_array().cloned())
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        format!(
                            "{} at {}: {}",
                            item["product"].as_str().unwrap_or("an alternative"),
                            item["route"].as_str().unwrap_or("its route"),
                            item["cause"].as_str().unwrap_or("no cause given")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .filter(|named| !named.is_empty());
        return Err(AppError::Config(match named {
            Some(named) => format!(
                "no BRAMA_BASE_URL (--brama-url) is named and no inference alternative answers {CONSUMER}: {named}; name Brama's address, or declare an endpoint in Brama's place (stado config set capabilities.inference.endpoint.url <url> and capabilities.inference.endpoint.model <model>)"
            ),
            None => format!(
                "no BRAMA_BASE_URL (--brama-url) is named and `{} {}` exited {}: {}",
                stado(),
                words.join(" "),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        }));
    }
    let answer: Value = serde_json::from_slice(&output.stdout).map_err(|error| {
        AppError::Config(format!(
            "`stado {}` printed an answer that is not JSON: {error}",
            words.join(" ")
        ))
    })?;
    let text = |field: &str| {
        answer[field]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    let (Some(provider), Some(route)) = (text("provider"), text("route")) else {
        return Err(AppError::Config(format!(
            "`stado {}` answered no provider and route: {answer}",
            words.join(" ")
        )));
    };
    let url = parse_http_url(&route, &format!("the {provider} route Stado resolved"))?;
    if provider == "brama" {
        return Ok(Resolved::Brama(url));
    }
    let model = text("model").ok_or_else(|| {
        AppError::Config(format!(
            "{provider} at {route} answers in Brama's place but declares no model: set capabilities.inference.endpoint.model"
        ))
    })?;
    let key = match text("key_role") {
        Some(role) => Some(role_key(&provider, &route, &role)?),
        None => None,
    };
    Ok(Resolved::InPlace {
        provider,
        url,
        model,
        key,
    })
}

/// The `token` field of the item playing `role`, read with
/// `stado credentials get --role`.
fn role_key(provider: &str, route: &str, role: &str) -> Result<SecretString, AppError> {
    let words = ["credentials", "get", "--role", role, "--field", "token"];
    let output = run(&words)?;
    if !output.status.success() {
        return Err(AppError::Secret(format!(
            "{provider} at {route} answers in Brama's place, but `stado {}` exited {}: {}",
            words.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let text = zeroize::Zeroizing::new(String::from_utf8(output.stdout).map_err(|_| {
        AppError::Secret(format!(
            "the key of role {role} for {provider} at {route} is not UTF-8"
        ))
    })?);
    secret_value(&text, &format!("the key of role {role}"))
}
