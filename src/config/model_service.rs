//! Where cognition's model calls go. A launch that names Brama's address
//! (`BRAMA_BASE_URL`) posts to it with Brama's credentials; one that names
//! none hands each request to Stado (`stado capabilities call inference
//! --consumer singularity`), which asks the first alternative in its selected
//! order that answers it — Brama, or a statement recorded for that exact
//! request — and prints that answer, or refuses naming every alternative with
//! its cause. The being names no alternative and reads no credential then, so
//! one Stado gains later is used with no change here.
use std::process::{Command, Stdio};

use secrecy::SecretString;
use serde_json::Value;
use url::Url;

use crate::error::AppError;

/// The consumer a being asks Stado's capabilities as.
pub const CONSUMER: &str = "singularity";

/// The model service and the credentials only it may receive.
#[derive(Clone)]
pub enum ModelService {
    /// Brama at the address the launch names: every request carries the
    /// bearer and the agent's signature.
    Brama {
        url: Url,
        secret: SecretString,
        bearer: SecretString,
    },
    /// Stado's inference alternatives: each request is handed to `stado
    /// capabilities call`, which presents what the alternative it asks needs.
    Stado,
}

impl ModelService {
    /// `brama`, or `stado` when requests go to Stado's alternatives.
    pub fn provider(&self) -> &str {
        match self {
            Self::Brama { .. } => "brama",
            Self::Stado => "stado",
        }
    }

    /// Where requests go, as a report and a refusal name it.
    pub fn describe(&self) -> String {
        match self {
            Self::Brama { url, .. } => url.to_string(),
            Self::Stado => call_operation(),
        }
    }
}

/// The executable answering Stado's commands: `STADO_BIN`, else `stado`.
fn stado() -> String {
    std::env::var("STADO_BIN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "stado".into())
}

const CALL: [&str; 6] = [
    "capabilities",
    "call",
    "inference",
    "--consumer",
    CONSUMER,
    "--json",
];

/// The command each request is handed to.
pub fn call_operation() -> String {
    format!("{} {}", stado(), CALL.join(" "))
}

/// The `Error:` line of a failed Stado command, else its last line.
fn cause(said: &str) -> String {
    let lines: Vec<&str> = said
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    lines
        .iter()
        .rev()
        .find_map(|line| line.strip_prefix("Error: "))
        .or_else(|| lines.last().copied())
        .unwrap_or("it said nothing")
        .to_owned()
}

/// Run `stado <words>` with `input` on its standard input to its end.
async fn run(words: &[&str], input: &[u8]) -> Result<std::process::Output, String> {
    let binary = stado();
    let operation = format!("{binary} {}", words.join(" "));
    let mut child = tokio::process::Command::new(&binary)
        .args(words)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| {
            format!("no BRAMA_BASE_URL (--brama-url) is named and `{operation}` could not run: {error}; name Brama's address or install Stado")
        })?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("`{operation}` has no input pipe"))?;
    tokio::io::AsyncWriteExt::write_all(&mut stdin, input)
        .await
        .map_err(|error| format!("cannot hand the request to `{operation}`: {error}"))?;
    drop(stdin);
    stado_wait::until(
        stado_wait::Kind::Process,
        &operation,
        "this host's Stado",
        child.wait_with_output(),
    )
    .await
    .map_err(|error| format!("`{operation}` did not end: {error}"))
}

/// One chat-completions request body, answered by the alternative Stado
/// chose, as the bytes an endpoint would have answered with. Refused with
/// Stado's sentence when no alternative answers.
pub async fn call(body: &[u8]) -> Result<Vec<u8>, String> {
    let output = run(&CALL, body).await?;
    if !output.status.success() {
        return Err(format!(
            "no inference alternative answered `{}` ({}): {}",
            call_operation(),
            output.status,
            cause(&String::from_utf8_lossy(&output.stderr))
        ));
    }
    let printed: Value = serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "`{}` printed an answer that is not JSON ({error}): {}",
            call_operation(),
            String::from_utf8_lossy(&output.stdout).trim()
        )
    })?;
    match printed.get("answer") {
        Some(answer) if !answer.is_null() => {
            serde_json::to_vec(answer).map_err(|error| error.to_string())
        }
        _ => Err(format!(
            "`{}` printed no answer: {printed}",
            call_operation()
        )),
    }
}

/// Stado's inference alternatives for this being, in their selected order,
/// read without asking any of them: what `doctor` checks when no Brama
/// address is named.
pub fn alternatives() -> Result<Vec<String>, AppError> {
    let binary = stado();
    let operation = format!("{binary} capabilities inference --json");
    let output = stado_wait::output(
        Command::new(&binary)
            .args(["capabilities", "inference", "--json"])
            .stdin(Stdio::null()),
    )
    .map_err(|error| {
        AppError::Config(format!(
            "no BRAMA_BASE_URL (--brama-url) is named and `{operation}` could not run: {error}; name Brama's address or install Stado"
        ))
    })?;
    if !output.status.success() {
        return Err(AppError::Config(format!(
            "no BRAMA_BASE_URL (--brama-url) is named and `{operation}` failed ({}): {}",
            output.status,
            cause(&String::from_utf8_lossy(&output.stderr))
        )));
    }
    let listed: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| AppError::Config(format!("`{operation}` printed no JSON: {error}")))?;
    let order: Vec<String> = listed["alternatives"]
        .as_array()
        .and_then(|families| {
            families
                .iter()
                .find(|family| family["family"].as_str() == Some("inference"))
        })
        .and_then(|family| family["order"].as_array())
        .map(|order| {
            order
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if order.is_empty() {
        return Err(AppError::Config(format!(
            "`{operation}` selects no inference alternative: {listed}"
        )));
    }
    Ok(order)
}
