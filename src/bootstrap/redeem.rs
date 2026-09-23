//! Redeeming a capability with Skarbiec and materializing the secret it answers with.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileTypeExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::{Signer, SigningKey};
use uuid::Uuid;
use zeroize::Zeroize;

use super::*;
use crate::config::GROUP_OR_OTHER_ACCESS;
use crate::error::AppError;
pub(super) fn redeem(
    socket: &Path,
    capability_id: &str,
    workload_id: &str,
    signing_key: &SigningKey,
) -> Result<Vec<u8>, AppError> {
    let socket_metadata = fs::symlink_metadata(socket)
        .map_err(|_| AppError::Secret("capability redemption denied".into()))?;
    if socket_metadata.file_type().is_symlink() || !socket_metadata.file_type().is_socket() {
        return Err(AppError::Secret("capability redemption denied".into()));
    }
    let nonce = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let mut proof_input = Vec::with_capacity(
        PROOF_DOMAIN.len() + capability_id.len() + nonce.len() + workload_id.len() + 2,
    );
    proof_input.extend_from_slice(PROOF_DOMAIN);
    proof_input.extend_from_slice(capability_id.as_bytes());
    proof_input.push(0);
    proof_input.extend_from_slice(nonce.as_bytes());
    proof_input.push(0);
    proof_input.extend_from_slice(workload_id.as_bytes());
    let proof = URL_SAFE_NO_PAD.encode(signing_key.sign(&proof_input).to_bytes());
    proof_input.zeroize();

    let request = RedeemRequest {
        version: WIRE_VERSION,
        capability_id,
        nonce: &nonce,
        workload_id,
        proof,
    };
    let mut encoded = serde_json::to_vec(&request)?;
    encoded.push(b'\n');
    let mut stream = UnixStream::connect(socket)
        .map_err(|_| AppError::Secret("capability redemption denied".into()))?;
    stream
        .set_read_timeout(Some(BROKER_IO_TIMEOUT))
        .and_then(|_| stream.set_write_timeout(Some(BROKER_IO_TIMEOUT)))
        .map_err(|_| AppError::Secret("capability redemption denied".into()))?;
    let write_result = stream.write_all(&encoded);
    encoded.zeroize();
    write_result.map_err(|_| AppError::Secret("capability redemption denied".into()))?;

    let line = read_control_line(&mut stream)?;
    let control: RedeemControl = serde_json::from_slice(&line)
        .map_err(|_| AppError::Secret("capability redemption denied".into()))?;
    if control.version != WIRE_VERSION || control.status != "ok" {
        return Err(AppError::Secret("capability redemption denied".into()));
    }
    let length = control
        .secret_len
        .filter(|length| *length > 0 && *length <= MAX_SECRET_BYTES)
        .ok_or_else(|| AppError::Secret("capability redemption denied".into()))?;
    let mut secret = vec![0_u8; length];
    if stream.read_exact(&mut secret).is_err() {
        secret.zeroize();
        return Err(AppError::Secret("capability redemption denied".into()));
    }
    let mut extra = [0_u8; 1];
    match stream.read(&mut extra) {
        Ok(0) => {}
        Ok(_) | Err(_) => {
            secret.zeroize();
            return Err(AppError::Secret("capability redemption denied".into()));
        }
    }
    Ok(secret)
}

pub(super) fn materialize(
    socket: &Path,
    capability_id: &str,
    workload_id: &str,
    signing_key: &SigningKey,
    destination: &Path,
) -> Result<File, AppError> {
    let mut secret = redeem(socket, capability_id, workload_id, signing_key)?;
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(destination)?;
        file.write_all(&secret)?;
        file.sync_all()?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.permissions().mode() & GROUP_OR_OTHER_ACCESS != 0 {
            return Err(AppError::Config(
                "bootstrap credential file is not owner-only".into(),
            ));
        }
        fs::remove_file(destination)?;
        Ok(file)
    })();
    secret.zeroize();
    result
}

pub(super) fn launch(
    manifest: &BootstrapManifest,
    credentials: &[File; 3],
) -> Result<std::convert::Infallible, AppError> {
    let (home, path) = crate::config::environment::runtime_paths()?;
    let mut command = Command::new(&manifest.singularity_executable);
    command
        .args(&manifest.singularity_args)
        .env_clear()
        .env("HOME", home)
        .env("PATH", path)
        .env("LANG", "C.UTF-8")
        .env("LC_ALL", "C.UTF-8")
        .env("SINGULARITY_AGENT_ID", &manifest.agent_id)
        .env("SINGULARITY_ROLE", &manifest.role)
        .env("SINGULARITY_ENVIRONMENT", &manifest.environment)
        .env("SINGULARITY_HOST", &manifest.host)
        .env("SINGULARITY_WORKLOAD_ID", &manifest.workload_id)
        .env(
            "SINGULARITY_WORKLOAD_PUBLIC_KEY",
            &manifest.workload_public_key,
        )
        .env("SINGULARITY_EXECUTABLE_SHA256", &manifest.executable_digest)
        .env("SINGULARITY_CODE_SHA256", &manifest.code_digest)
        .env("SINGULARITY_POLICY_SHA256", &manifest.policy_digest)
        .env(
            "SINGULARITY_POLICY_SEQUENCE",
            manifest.policy_sequence.to_string(),
        )
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    super::credentials::pass_files(&mut command, credentials);
    Err(AppError::Io(command.exec()))
}
