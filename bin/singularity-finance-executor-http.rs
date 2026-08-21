use std::path::PathBuf;
use std::time::Duration;

use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use url::Url;
use zeroize::Zeroize;

const MAX_DOCUMENT_BYTES: u64 = 64 * 1024;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExecutionResponse {
    executor_id: String,
    executor_reference_hash: String,
    executor_signature_hex: String,
    worm_receipt_file: PathBuf,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("singularity-finance-executor-http: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let endpoint = required_https_url("SINGULARITY_FINANCE_CUSTODY_URL")?;
    let token_file = required_owner_file("SINGULARITY_FINANCE_CUSTODY_TOKEN_FILE")?;
    let mut token = std::fs::read_to_string(&token_file)
        .map_err(|error| format!("cannot read custody token: {error}"))?;
    while token.ends_with(['\r', '\n']) {
        token.pop();
    }
    if token.is_empty() {
        return Err("custody token is empty".into());
    }

    let mut bytes = Vec::new();
    tokio::io::stdin()
        .take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|error| format!("cannot read execution request: {error}"))?;
    if bytes.len() as u64 > MAX_DOCUMENT_BYTES {
        token.zeroize();
        return Err("execution request exceeds size limit".into());
    }
    let request: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("execution request is invalid JSON: {error}"))?;
    if !request.is_object() {
        token.zeroize();
        return Err("execution request must be a JSON object".into());
    }

    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(Policy::none())
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|error| format!("cannot build custody client: {error}"))?;
    let response = client
        .post(endpoint)
        .bearer_auth(&token)
        .json(&request)
        .send()
        .await
        .map_err(|error| format!("custody request failed: {error}"));
    token.zeroize();
    let response = response?;
    if !response.status().is_success() {
        return Err(format!(
            "custody service refused with HTTP {}",
            response.status().as_u16()
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_DOCUMENT_BYTES)
    {
        return Err("custody response exceeds size limit".into());
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("cannot read custody response: {error}"))?;
    if bytes.len() as u64 > MAX_DOCUMENT_BYTES {
        return Err("custody response exceeds size limit".into());
    }
    let response: ExecutionResponse = serde_json::from_slice(&bytes)
        .map_err(|error| format!("custody response is invalid: {error}"))?;
    if response.executor_id.is_empty()
        || !is_hash(&response.executor_reference_hash)
        || !is_signature(&response.executor_signature_hex)
        || !response.worm_receipt_file.is_absolute()
    {
        return Err("custody response failed validation".into());
    }
    let output = serde_json::to_vec(&response)
        .map_err(|error| format!("cannot encode executor response: {error}"))?;
    tokio::io::stdout()
        .write_all(&output)
        .await
        .map_err(|error| format!("cannot write executor response: {error}"))?;
    Ok(())
}

fn required_https_url(name: &str) -> Result<Url, String> {
    let raw = std::env::var(name).map_err(|_| format!("{name} is required"))?;
    let url = Url::parse(&raw).map_err(|error| format!("{name}: {error}"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || url.username() != ""
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
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!(
                "{name} must be owner-only and owned by the current user"
            ));
        }
    }
    Ok(path)
}

fn is_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_signature(value: &str) -> bool {
    value.len() == 128 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
