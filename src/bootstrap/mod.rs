use std::fs::{self, DirBuilder};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zeroize::Zeroize;

use crate::error::AppError;

const MANIFEST_DOMAIN: &[u8] = b"SINGULARITY-BOOTSTRAP-MANIFEST\0v2\0";
const PROOF_DOMAIN: &[u8] = b"SKARBIEC-WORKLOAD-PROOF\0v1\0";
const WIRE_VERSION: &str = "skarbiec.redeem.v1";
/// The one operation a bootstrap performs on a capability.
const REDEEM_OPERATION: &str = "redeem";
/// A bootstrap capability is issued without an authorization id; the wire
/// carries it, and the proof covers it, as the empty string.
const NO_AUTHORIZATION: &str = "";
// How long a manifest lives is the issuer's signed expires_at; a broker answer is waited for,
// and a secret or control line is as long as the broker sends.
/// An Ed25519 key is 32 bytes, spelled as 64 hex characters.
pub(crate) const KEY_BYTES: usize = 32;
const KEY_HEX_CHARS: usize = 64;
/// The manifest edition this bootstrap accepts, and the target and purposes
/// its three capabilities must be bound to; the issuer writes the same words.
pub(crate) const MANIFEST_VERSION: &str = "singularity.bootstrap.v2";
pub(crate) const CAPABILITY_TARGET: &str = "singularity-bootstrap";
pub(crate) const BRAMA_PURPOSE: &str = "singularity.brama.bootstrap";
pub(crate) const BRAMA_BEARER_PURPOSE: &str = "singularity.brama.authorization";
pub(crate) const MOST_PURPOSE: &str = "singularity.most.bootstrap";

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapCapabilities {
    pub brama: BootstrapCapability,
    pub brama_bearer: BootstrapCapability,
    pub most: BootstrapCapability,
}

#[derive(Debug, Deserialize, Serialize)]
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
    operation: &'static str,
    capability_id: &'a str,
    nonce: &'a str,
    workload_id: &'a str,
    authorization_id: &'static str,
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

/// The bytes a launch ticket's signature covers: the v2 domain, then the
/// manifest exactly as written.
pub(crate) fn signed_manifest_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut signed = Vec::with_capacity(MANIFEST_DOMAIN.len() + bytes.len());
    signed.extend_from_slice(MANIFEST_DOMAIN);
    signed.extend_from_slice(bytes);
    signed
}

/// Every check `singularity-bootstrap` runs on a launch ticket before it
/// redeems anything: owner-only files, the signature against the trust root,
/// the manifest's own rules, and the executable's digest. An issuer runs it on
/// what it has just written, so a ticket the consumer would refuse is refused
/// where it was made, with the same sentence.
pub fn check_ticket(
    manifest_path: &Path,
    signature_path: &Path,
    trust_root_path: &Path,
) -> Result<BootstrapManifest, AppError> {
    for path in [manifest_path, signature_path, trust_root_path] {
        require_owner_file(path)?;
    }
    let manifest_bytes = fs::read(manifest_path)?;
    verify_manifest(&manifest_bytes, signature_path, trust_root_path)?;
    let manifest: BootstrapManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&manifest)?;
    verify_executable(
        &manifest.singularity_executable,
        &manifest.executable_digest,
    )?;
    Ok(manifest)
}

pub fn run_bootstrap(
    manifest_path: &Path,
    signature_path: &Path,
    trust_root_path: &Path,
    runtime_root: &Path,
) -> Result<std::convert::Infallible, AppError> {
    if !runtime_root.is_absolute() {
        return Err(AppError::Config(
            "bootstrap runtime root must be absolute".into(),
        ));
    }
    let manifest = check_ticket(manifest_path, signature_path, trust_root_path)?;

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
    let bearer_path = cleanup.path().join("brama.token");
    let most_path = cleanup.path().join("most.token");
    let brama = materialize(
        &manifest.broker_socket,
        &manifest.capabilities.brama.id,
        &manifest.workload_id,
        &signing_key,
        &brama_path,
    )?;
    let bearer = materialize(
        &manifest.broker_socket,
        &manifest.capabilities.brama_bearer.id,
        &manifest.workload_id,
        &signing_key,
        &bearer_path,
    )?;
    let most = materialize(
        &manifest.broker_socket,
        &manifest.capabilities.most.id,
        &manifest.workload_id,
        &signing_key,
        &most_path,
    )?;
    validate_manifest(&manifest)?;
    // Each credential is already unlinked but held by its open descriptor.
    // Remove the empty directory before exec; no guardian process is needed.
    drop(cleanup);
    launch(&manifest, &[brama, bearer, most])
}

mod credentials;
mod files;
mod manifest;
mod redeem;

pub use credentials::adopt_credentials;
pub(crate) use credentials::{inherit_for_child, inherited_credentials};

use files::*;
pub(crate) use files::read_hex_32;
use manifest::*;
pub(crate) use manifest::validate_manifest;
use redeem::*;
