use std::path::PathBuf;
use std::process::ExitCode;

use ed25519_dalek::SIGNATURE_LENGTH;
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use url::Url;
use zeroize::Zeroize;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExecutionResponse {
    executor_id: String,
    executor_reference_hash: String,
    executor_signature_hex: String,
    worm_receipt_file: PathBuf,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("singularity-finance-executor-http: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Forward one execution request to the custody service and pass its signed
/// answer back. The request waits for the custody service's answer, and
/// stderr says what it waits on while it does: an indeterminate submission
/// is the finance service's to reconcile, never this adapter's to cut short.
async fn run() -> Result<(), String> {
    let endpoint = required_https_url("SINGULARITY_FINANCE_CUSTODY_URL")?;
    let token_file = required_owner_file("SINGULARITY_FINANCE_CUSTODY_TOKEN_FILE")?;
    let mut token = std::fs::read_to_string(&token_file).map_err(|error| {
        format!(
            "cannot read custody token {}: {error}",
            token_file.display()
        )
    })?;
    while token.ends_with(['\r', '\n']) {
        token.pop();
    }
    if token.is_empty() {
        return Err(format!("custody token {} is empty", token_file.display()));
    }

    let mut bytes = Vec::new();
    let read = tokio::io::stdin().read_to_end(&mut bytes).await;
    let request = read
        .map_err(|error| format!("cannot read execution request: {error}"))
        .and_then(|_| {
            serde_json::from_slice::<Value>(&bytes)
                .map_err(|error| format!("execution request is invalid JSON: {error}"))
        })
        .and_then(|request| {
            if request.is_object() {
                Ok(request)
            } else {
                Err(format!(
                    "execution request must be a JSON object, not {}; send the finance service's execution request object on stdin",
                    request_kind(&request)
                ))
            }
        });
    let request = match request {
        Ok(request) => request,
        Err(error) => {
            token.zeroize();
            return Err(error);
        }
    };

    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(Policy::none())
        .build()
        .map_err(|error| format!("cannot build custody client: {error}"));
    let response = match client {
        Ok(client) => stado_wait::request(client.post(endpoint).bearer_auth(&token).json(&request))
            .await
            .map_err(|error| format!("custody request failed: {error}")),
        Err(error) => Err(error),
    };
    token.zeroize();
    let response = response?;
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("cannot read custody response (HTTP {status}): {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "custody service refused with HTTP {status}: {}",
            String::from_utf8_lossy(&bytes)
        ));
    }
    let response: ExecutionResponse = serde_json::from_slice(&bytes)
        .map_err(|error| format!("custody response is invalid: {error}"))?;
    validate(&response)?;
    let output = serde_json::to_vec(&response)
        .map_err(|error| format!("cannot encode executor response: {error}"))?;
    tokio::io::stdout()
        .write_all(&output)
        .await
        .map_err(|error| format!("cannot write executor response: {error}"))?;
    Ok(())
}

/// The custody answer's fields, each refused by name when it is not what
/// the finance service records.
fn validate(response: &ExecutionResponse) -> Result<(), String> {
    if response.executor_id.is_empty() {
        return Err("custody response names no executor_id".into());
    }
    if !hex_of_length(&response.executor_reference_hash, Sha256::output_size()) {
        return Err(format!(
            "custody response executor_reference_hash is not a hex SHA-256 digest: {:?}",
            response.executor_reference_hash
        ));
    }
    if !hex_of_length(&response.executor_signature_hex, SIGNATURE_LENGTH) {
        return Err(format!(
            "custody response executor_signature_hex is not a hex Ed25519 signature: {:?}",
            response.executor_signature_hex
        ));
    }
    if !response.worm_receipt_file.is_absolute() {
        return Err(format!(
            "custody response worm_receipt_file is not an absolute path: {}",
            response.worm_receipt_file.display()
        ));
    }
    Ok(())
}

/// Whether `value` is hex that decodes to exactly `bytes` bytes.
fn hex_of_length(value: &str, bytes: usize) -> bool {
    hex::decode(value).is_ok_and(|decoded| decoded.len() == bytes)
}

/// The JSON kind a request that is not an object turned out to be.
fn request_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn required_https_url(name: &str) -> Result<Url, String> {
    let raw = std::env::var(name).map_err(|error| format!("{name} is required: {error}"))?;
    let url = Url::parse(&raw).map_err(|error| format!("{name}: {error}"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(format!("{name} must be a credential-free HTTPS URL"));
    }
    Ok(url)
}

fn required_owner_file(name: &str) -> Result<PathBuf, String> {
    let value = std::env::var_os(name).ok_or_else(|| format!("{name} is required"))?;
    let path = PathBuf::from(value);
    if !path.is_absolute()
        || !path.is_file()
        || path
            .symlink_metadata()
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err(format!(
            "{name} must name an absolute regular non-symlink file"
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let metadata = std::fs::metadata(&path).map_err(|error| format!("{name}: {error}"))?;
        // The group's and others' read, write and search bits.
        let shared = u32::from(libc::S_IRWXG | libc::S_IRWXO);
        let mode = metadata.permissions().mode();
        let owner_only = mode & shared == 0;
        let owner = unsafe { libc::geteuid() };
        if metadata.uid() != owner || !owner_only {
            return Err(format!(
                "{name} must be owner-only and owned by the current user; observed {} owned by uid {} with mode {:o} (effective uid {owner}); chmod go-rwx and chown it to this user",
                path.display(),
                metadata.uid(),
                mode
            ));
        }
    }
    Ok(path)
}
