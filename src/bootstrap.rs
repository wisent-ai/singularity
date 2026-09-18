use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zeroize::Zeroize;

use crate::config::GROUP_OR_OTHER_ACCESS;
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
    #[serde(default)]
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

fn verify_manifest(
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

fn validate_manifest(manifest: &BootstrapManifest) -> Result<(), AppError> {
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

fn redeem(
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

fn materialize(
    socket: &Path,
    capability_id: &str,
    workload_id: &str,
    signing_key: &SigningKey,
    destination: &Path,
) -> Result<(), AppError> {
    let mut secret = redeem(socket, capability_id, workload_id, signing_key)?;
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
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
        Ok(())
    })();
    secret.zeroize();
    result
}

fn launch(
    manifest: &BootstrapManifest,
    brama_path: &Path,
    most_path: &Path,
) -> Result<ExitStatus, AppError> {
    Command::new(&manifest.singularity_executable)
        .args(&manifest.singularity_args)
        .env_clear()
        .env(
            "PATH",
            "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
        )
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
        .env("BRAMA_HMAC_SECRET_FILE", brama_path)
        .env("MOST_SERVICE_TOKEN_FILE", most_path)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(AppError::Io)
}

fn read_control_line(stream: &mut UnixStream) -> Result<Vec<u8>, AppError> {
    let mut line = Vec::new();
    while line.len() < MAX_CONTROL_LINE {
        let mut byte = [0_u8; 1];
        stream
            .read_exact(&mut byte)
            .map_err(|_| AppError::Secret("capability redemption denied".into()))?;
        if byte[0] == b'\n' {
            return Ok(line);
        }
        line.push(byte[0]);
    }
    Err(AppError::Secret("capability redemption denied".into()))
}

fn require_owner_file(path: &Path) -> Result<(), AppError> {
    if !path.is_absolute() {
        return Err(AppError::Config(
            "bootstrap requires absolute regular owner-only files".into(),
        ));
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & GROUP_OR_OTHER_ACCESS != 0
    {
        return Err(AppError::Config(
            "bootstrap requires absolute regular owner-only files".into(),
        ));
    }
    Ok(())
}

fn prepare_runtime_root(path: &Path) -> Result<(), AppError> {
    if path.exists() {
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.is_dir()
            || metadata.file_type().is_symlink()
            || metadata.permissions().mode() & GROUP_OR_OTHER_ACCESS != 0
        {
            return Err(AppError::Config(
                "bootstrap runtime root must be an owner-only directory".into(),
            ));
        }
        return Ok(());
    }
    DirBuilder::new().recursive(true).mode(0o700).create(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & GROUP_OR_OTHER_ACCESS != 0
    {
        return Err(AppError::Config(
            "bootstrap runtime root must be an owner-only directory".into(),
        ));
    }
    Ok(())
}

fn valid_atom(value: &str, max_len: usize) -> bool {
    !value.is_empty() && value.len() <= max_len && value == value.trim() && !value.contains('\0')
}

fn valid_capability_binding(
    capability: &BootstrapCapability,
    purpose: &str,
    resource_prefix: &str,
) -> bool {
    capability.target == "singularity-bootstrap"
        && capability.purpose == purpose
        && capability.resource.starts_with(resource_prefix)
        && valid_atom(&capability.resource[resource_prefix.len()..], 512)
        && !capability.resource.contains('*')
}

fn read_hex_32(path: &Path, label: &str) -> Result<[u8; KEY_BYTES], AppError> {
    require_owner_file(path)?;
    let mut text = fs::read_to_string(path)?;
    let mut decoded =
        hex::decode(text.trim()).map_err(|_| AppError::Config(format!("invalid {label}")))?;
    text.zeroize();
    if decoded.len() != KEY_BYTES {
        decoded.zeroize();
        return Err(AppError::Config(format!("invalid {label}")));
    }
    let mut result = [0_u8; KEY_BYTES];
    result.copy_from_slice(&decoded);
    decoded.zeroize();
    Ok(result)
}

fn read_hex_64(path: &Path) -> Result<[u8; 64], AppError> {
    require_owner_file(path)?;
    let text = fs::read_to_string(path)?;
    let decoded = hex::decode(text.trim())
        .map_err(|_| AppError::Config("invalid bootstrap manifest signature".into()))?;
    decoded
        .try_into()
        .map_err(|_| AppError::Config("invalid bootstrap manifest signature".into()))
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == KEY_HEX_CHARS
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn verify_executable(path: &Path, expected: &str) -> Result<(), AppError> {
    if !path.is_absolute() {
        return Err(AppError::Config(
            "singularity executable must be an absolute regular file".into(),
        ));
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(AppError::Config(
            "singularity executable must be an absolute regular file".into(),
        ));
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if hex::encode(hasher.finalize()) != expected {
        return Err(AppError::Config(
            "singularity executable digest mismatch".into(),
        ));
    }
    Ok(())
}

struct RuntimeCleanup {
    directory: PathBuf,
}

impl RuntimeCleanup {
    fn new(directory: PathBuf) -> Self {
        Self { directory }
    }

    fn path(&self) -> &Path {
        &self.directory
    }
}

impl Drop for RuntimeCleanup {
    fn drop(&mut self) {
        secure_remove(&self.directory.join("brama.hmac"));
        secure_remove(&self.directory.join("most.token"));
        let _ = fs::remove_dir(&self.directory);
    }
}

fn secure_remove(path: &Path) {
    if let Ok(metadata) = fs::metadata(path) {
        if let Ok(mut file) = OpenOptions::new().write(true).open(path) {
            let zeros = vec![0_u8; metadata.len().min(MAX_SECRET_BYTES as u64) as usize];
            let _ = file.write_all(&zeros);
            let _ = file.sync_all();
        }
    }
    let _ = fs::remove_file(path);
}


#[cfg(test)]
#[path = "../tests/bootstrap/cases.rs"]
mod tests;
