//! `singularity ticket key|sign`: the issuer of the signed launch ticket
//! `singularity-bootstrap` accepts.
//!
//! `singularity-bootstrap` starts a managed being only from a
//! `singularity.bootstrap.v2` manifest signed by the supervisor whose public
//! key is the trust root, naming a workload key, the executable's and the
//! policy's digests and three Skarbiec capabilities. Nothing wrote one, so no
//! host could start a managed Singularity. `key` writes a fresh Ed25519 key
//! (a supervisor's, whose public half is the trust root, or a workload's,
//! whose public half `skarbiec grant issue --workload-public-key-file`
//! registers). `sign` writes the manifest from explicit inputs, signs the
//! bytes the consumer verifies, and runs the consumer's own `check_ticket`
//! on what it wrote: a ticket `singularity-bootstrap` would refuse is removed
//! and refused here with the same sentence.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::{Signer, SigningKey};
use serde::Serialize;
use sha2::{Digest, Sha256};
use zeroize::Zeroize;

use crate::bootstrap::{
    check_ticket, read_hex_32, signed_manifest_bytes, validate_manifest, BootstrapCapabilities, BootstrapCapability,
    BootstrapManifest, BRAMA_BEARER_PURPOSE, BRAMA_PURPOSE, CAPABILITY_TARGET, KEY_BYTES, MANIFEST_VERSION,
    MOST_PURPOSE,
};
use crate::error::AppError;

mod args;

pub use args::{KeyArgs, SignArgs, TicketArgs, TicketVerb};

/// Where a fresh key's seed is read from.
const RANDOM_SOURCE: &str = "/dev/urandom";

/// What `ticket key` wrote.
#[derive(Debug, Serialize)]
pub struct KeyReport {
    pub private_key: PathBuf,
    pub public_key: PathBuf,
    pub public_key_hex: String,
}

/// What `ticket sign` wrote and checked.
#[derive(Debug, Serialize)]
pub struct TicketReport {
    pub manifest: PathBuf,
    pub signature: PathBuf,
    pub workload_public_key: String,
    pub executable_digest: String,
    pub policy_digest: String,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub checked_by: &'static str,
}

/// Write a fresh Ed25519 key pair.
pub fn key(args: &KeyArgs) -> Result<KeyReport, AppError> {
    absolute(&args.out, "--out")?;
    absolute(&args.public_out, "--public-out")?;
    let mut seed = <[u8; KEY_BYTES]>::default();
    File::open(RANDOM_SOURCE)
        .and_then(|mut source| source.read_exact(&mut seed))
        .map_err(|error| AppError::Config(format!("{RANDOM_SOURCE} gave no key seed: {error}")))?;
    let signing = SigningKey::from_bytes(&seed);
    let mut private_hex = hex::encode(seed);
    seed.zeroize();
    let written = write_owner_only(&args.out, private_hex.as_bytes());
    private_hex.zeroize();
    written?;
    let public_key_hex = hex::encode(signing.verifying_key().as_bytes());
    write_owner_only(&args.public_out, public_key_hex.as_bytes())?;
    Ok(KeyReport { private_key: args.out.clone(), public_key: args.public_out.clone(), public_key_hex })
}

/// Write, sign and check one launch ticket.
pub fn sign(args: &SignArgs) -> Result<TicketReport, AppError> {
    absolute(&args.manifest_out, "--manifest-out")?;
    absolute(&args.signature_out, "--signature-out")?;
    let mut workload_seed = read_hex_32(&args.workload_key, "workload key")?;
    let workload_public_key = hex::encode(SigningKey::from_bytes(&workload_seed).verifying_key().as_bytes());
    workload_seed.zeroize();
    let executable_digest = file_digest(&args.executable, "--executable")?;
    let policy_digest = file_digest(&args.policy_file, "--policy-file")?;
    let issued_at = Utc::now();
    let expires_at = issued_at + Duration::seconds(i64::from(args.expires_in_seconds));
    let manifest = BootstrapManifest {
        version: MANIFEST_VERSION.to_owned(),
        issued_at,
        expires_at,
        agent_id: args.agent_id.clone(),
        role: args.role.clone(),
        environment: args.environment.clone(),
        host: args.host.clone(),
        workload_id: args.workload_id.clone(),
        workload_public_key: workload_public_key.clone(),
        executable_digest: executable_digest.clone(),
        code_digest: args.code_digest.clone(),
        policy_digest: policy_digest.clone(),
        policy_sequence: args.policy_sequence,
        broker_socket: args.broker_socket.clone(),
        workload_private_key_file: args.workload_key.clone(),
        singularity_executable: args.executable.clone(),
        singularity_args: args.singularity_args.clone(),
        capabilities: BootstrapCapabilities {
            brama: capability(&args.brama_capability, BRAMA_PURPOSE, &args.brama_resource),
            brama_bearer: capability(&args.brama_bearer_capability, BRAMA_BEARER_PURPOSE, &args.brama_bearer_resource),
            most: capability(&args.most_capability, MOST_PURPOSE, &args.most_resource),
        },
    };
    // The consumer's rules, before anything is written.
    validate_manifest(&manifest)?;
    let bytes = serde_json::to_vec(&manifest)?;
    let mut supervisor_seed = read_hex_32(&args.supervisor_key, "supervisor key")?;
    let signature = SigningKey::from_bytes(&supervisor_seed).sign(&signed_manifest_bytes(&bytes));
    supervisor_seed.zeroize();
    write_owner_only(&args.manifest_out, &bytes)?;
    if let Err(error) = write_owner_only(&args.signature_out, hex::encode(signature.to_bytes()).as_bytes()) {
        fs::remove_file(&args.manifest_out)?;
        return Err(error);
    }
    if let Err(refusal) = check_ticket(&args.manifest_out, &args.signature_out, &args.trust_root) {
        fs::remove_file(&args.manifest_out)?;
        fs::remove_file(&args.signature_out)?;
        return Err(AppError::Config(format!(
            "singularity-bootstrap would refuse the ticket written, so it was removed: {refusal}"
        )));
    }
    Ok(TicketReport {
        manifest: args.manifest_out.clone(),
        signature: args.signature_out.clone(),
        workload_public_key,
        executable_digest,
        policy_digest,
        issued_at,
        expires_at,
        checked_by: "singularity-bootstrap check_ticket",
    })
}

fn capability(id: &str, purpose: &str, resource: &str) -> BootstrapCapability {
    BootstrapCapability {
        id: id.to_owned(),
        target: CAPABILITY_TARGET.to_owned(),
        purpose: purpose.to_owned(),
        resource: resource.to_owned(),
    }
}

fn absolute(path: &Path, flag: &str) -> Result<(), AppError> {
    if path.is_absolute() {
        return Ok(());
    }
    Err(AppError::Config(format!(
        "{flag} {} is not absolute; singularity-bootstrap reads only absolute owner-only files",
        path.display()
    )))
}

/// Lowercase SHA-256 hex of a file's bytes.
fn file_digest(path: &Path, flag: &str) -> Result<String, AppError> {
    let mut file = File::open(path)
        .map_err(|error| AppError::Config(format!("{flag} {} cannot be read: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

/// Create `path` as a new file only its owner can read and write.
fn write_owner_only(path: &Path, bytes: &[u8]) -> Result<(), AppError> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(u32::from(libc::S_IRUSR | libc::S_IWUSR))
        .open(path)
        .map_err(|error| {
            AppError::Config(format!("{} could not be created as a new owner-only file: {error}", path.display()))
        })?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
