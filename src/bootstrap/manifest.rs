//! The manifest a bootstrap is given, and the checks it must pass before anything runs.
use std::path::Path;

use chrono::{Duration, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use super::*;
use crate::error::AppError;
pub(super) fn verify_manifest(
    bytes: &[u8],
    signature_path: &Path,
    trust_root_path: &Path,
) -> Result<(), AppError> {
    let trust = read_hex_32(trust_root_path, "bootstrap trust root")?;
    let key = VerifyingKey::from_bytes(&trust)
        .map_err(|_| AppError::Config("invalid bootstrap trust root".into()))?;
    let signature_bytes = read_hex_64(signature_path)?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|_| AppError::Config("invalid bootstrap manifest signature".into()))?;
    let mut signed = Vec::with_capacity(MANIFEST_DOMAIN.len() + bytes.len());
    signed.extend_from_slice(MANIFEST_DOMAIN);
    signed.extend_from_slice(bytes);
    key.verify(&signed, &signature)
        .map_err(|_| AppError::Config("bootstrap manifest signature verification failed".into()))
}

pub(super) fn validate_manifest(manifest: &BootstrapManifest) -> Result<(), AppError> {
    let now = Utc::now();
    if manifest.version != "singularity.bootstrap.v1"
        || manifest.policy_sequence == 0
        || manifest.issued_at > now + Duration::seconds(MAX_ISSUED_AT_SKEW_SECONDS)
        || manifest.expires_at <= now
        || manifest.expires_at <= manifest.issued_at
        || manifest.expires_at - manifest.issued_at > Duration::seconds(MAX_MANIFEST_LIFETIME)
    {
        return Err(AppError::Config(
            "bootstrap manifest is invalid or expired".into(),
        ));
    }
    for digest in [
        &manifest.workload_public_key,
        &manifest.executable_digest,
        &manifest.code_digest,
        &manifest.policy_digest,
        &manifest.capabilities.brama.id,
        &manifest.capabilities.most.id,
    ] {
        if !is_lower_hex_64(digest) {
            return Err(AppError::Config(
                "bootstrap manifest contains an invalid digest or capability".into(),
            ));
        }
    }
    if manifest.capabilities.brama.id == manifest.capabilities.most.id
        || !valid_capability_binding(
            &manifest.capabilities.brama,
            "singularity.brama.bootstrap",
            "brama:",
        )
        || !valid_capability_binding(
            &manifest.capabilities.most,
            "singularity.most.bootstrap",
            "most:",
        )
        || !manifest.broker_socket.is_absolute()
        || !manifest.workload_private_key_file.is_absolute()
        || !manifest.singularity_executable.is_absolute()
        || !valid_atom(&manifest.workload_id, 128)
        || !valid_atom(&manifest.agent_id, 128)
        || !valid_atom(&manifest.role, 128)
        || !valid_atom(&manifest.environment, 128)
        || !valid_atom(&manifest.host, 255)
    {
        return Err(AppError::Config(
            "bootstrap manifest binding is invalid".into(),
        ));
    }
    require_owner_file(&manifest.workload_private_key_file)
}
