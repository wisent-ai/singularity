use std::fs::{self, DirBuilder};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use chrono::{DateTime, Utc};
use ed25519_dalek::SigningKey;
use serde::Deserialize;
use uuid::Uuid;
use zeroize::Zeroize;

use crate::error::AppError;

const MANIFEST_DOMAIN: &[u8] = b"SINGULARITY-BOOTSTRAP-MANIFEST\0v1\0";
const PROOF_DOMAIN: &[u8] = b"SKARBIEC-WORKLOAD-PROOF\0v1\0";
const WIRE_VERSION: &str = "skarbiec.redeem.v1";
const MAX_CONTROL_LINE: usize = 4096;
const MAX_SECRET_BYTES: usize = 64 * 1024;
const MAX_MANIFEST_LIFETIME: i64 = 300;
/// A manifest issued up to thirty seconds in the future is clock skew, not forgery.
const MAX_ISSUED_AT_SKEW_SECONDS: i64 = 30;
const BROKER_IO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
/// An Ed25519 key is 32 bytes, spelled as 64 hex characters.
const KEY_BYTES: usize = 32;
const KEY_HEX_CHARS: usize = 64;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapManifest {
    pub version: String,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub agent_id: String,
    pub role: String,
    pub environment: String,
    pub host: String,
    pub workload_id: String,
    pub workload_public_key: String,
    pub executable_digest: String,
    pub code_digest: String,
    pub policy_digest: String,
    pub policy_sequence: u64,
    pub broker_socket: PathBuf,
    pub workload_private_key_file: PathBuf,
    pub singularity_executable: PathBuf,
    pub singularity_args: Vec<String>,
    pub capabilities: BootstrapCapabilities,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapCapabilities {
    pub brama: BootstrapCapability,
    pub most: BootstrapCapability,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapCapability {
    pub id: String,
    pub target: String,
    pub purpose: String,
    pub resource: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct RedeemRequest<'a> {
    version: &'static str,
    capability_id: &'a str,
    nonce: &'a str,
    workload_id: &'a str,
    proof: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RedeemControl {
    version: String,
    status: String,
    #[serde(default)]
    secret_len: Option<usize>,
}

pub fn run_bootstrap(
    manifest_path: &Path,
    signature_path: &Path,
    trust_root_path: &Path,
    runtime_root: &Path,
) -> Result<ExitStatus, AppError> {
    for path in [manifest_path, signature_path, trust_root_path] {
        require_owner_file(path)?;
    }
    if !runtime_root.is_absolute() {
        return Err(AppError::Config(
            "bootstrap runtime root must be absolute".into(),
        ));
    }

    let manifest_bytes = fs::read(manifest_path)?;
    verify_manifest(&manifest_bytes, signature_path, trust_root_path)?;
    let manifest: BootstrapManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&manifest)?;
    verify_executable(
        &manifest.singularity_executable,
        &manifest.executable_digest,
    )?;

    let mut private_key = read_hex_32(&manifest.workload_private_key_file, "workload key")?;
    let signing_key = SigningKey::from_bytes(&private_key);
    private_key.zeroize();
    let derived_public = hex::encode(signing_key.verifying_key().as_bytes());
    if derived_public != manifest.workload_public_key {
        return Err(AppError::Config(
            "bootstrap workload key does not match manifest".into(),
        ));
    }

    prepare_runtime_root(runtime_root)?;
    let runtime_dir = runtime_root.join(format!("singularity-{}", Uuid::new_v4()));
    DirBuilder::new().mode(0o700).create(&runtime_dir)?;
    let cleanup = RuntimeCleanup::new(runtime_dir);
    let brama_path = cleanup.path().join("brama.hmac");
    let most_path = cleanup.path().join("most.token");
    let result = (|| {
        materialize(
            &manifest.broker_socket,
            &manifest.capabilities.brama.id,
            &manifest.workload_id,
            &signing_key,
            &brama_path,
        )?;
        materialize(
            &manifest.broker_socket,
            &manifest.capabilities.most.id,
            &manifest.workload_id,
            &signing_key,
            &most_path,
        )?;
        validate_manifest(&manifest)?;
        launch(&manifest, &brama_path, &most_path)
    })();

    drop(cleanup);
    result
}

mod files;
mod manifest;
mod redeem;

use files::*;
use manifest::*;
use redeem::*;

#[cfg(test)]
#[path = "../../tests/bootstrap/cases.rs"]
mod tests;
