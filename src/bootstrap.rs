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

use crate::error::AppError;

const MANIFEST_DOMAIN: &[u8] = b"SINGULARITY-BOOTSTRAP-MANIFEST\0v1\0";
const PROOF_DOMAIN: &[u8] = b"SKARBIEC-WORKLOAD-PROOF\0v1\0";
const WIRE_VERSION: &str = "skarbiec.redeem.v1";
const MAX_CONTROL_LINE: usize = 4096;
const MAX_SECRET_BYTES: usize = 64 * 1024;
const MAX_MANIFEST_LIFETIME: i64 = 300;
const BROKER_IO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

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
        || manifest.issued_at > now + Duration::seconds(30)
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
        if !metadata.is_file() || metadata.permissions().mode() & 0o077 != 0 {
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
        || metadata.permissions().mode() & 0o077 != 0
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
            || metadata.permissions().mode() & 0o077 != 0
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
        || metadata.permissions().mode() & 0o077 != 0
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

fn read_hex_32(path: &Path, label: &str) -> Result<[u8; 32], AppError> {
    require_owner_file(path)?;
    let mut text = fs::read_to_string(path)?;
    let mut decoded =
        hex::decode(text.trim()).map_err(|_| AppError::Config(format!("invalid {label}")))?;
    text.zeroize();
    if decoded.len() != 32 {
        decoded.zeroize();
        return Err(AppError::Config(format!("invalid {label}")));
    }
    let mut result = [0_u8; 32];
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
    value.len() == 64
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
mod tests {
    use super::*;

    struct TempDirectory {
        path: PathBuf,
    }

    impl TempDirectory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("singularity-bootstrap-test-{}", Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn capability(target: &str, purpose: &str, resource: &str) -> BootstrapCapability {
        BootstrapCapability {
            id: "a".repeat(64),
            target: target.to_owned(),
            purpose: purpose.to_owned(),
            resource: resource.to_owned(),
        }
    }

    #[test]
    fn valid_capability_binding_accepts_canonical_brama_and_most_bindings() {
        let cases = [
            (
                capability(
                    "singularity-bootstrap",
                    "singularity.brama.bootstrap",
                    "brama:agent-42",
                ),
                "singularity.brama.bootstrap",
                "brama:",
            ),
            (
                capability(
                    "singularity-bootstrap",
                    "singularity.most.bootstrap",
                    "most:publisher-42",
                ),
                "singularity.most.bootstrap",
                "most:",
            ),
        ];

        for (binding, expected_purpose, expected_resource_prefix) in cases {
            assert!(
                valid_capability_binding(&binding, expected_purpose, expected_resource_prefix),
                "canonical binding was rejected: {binding:?}"
            );
        }
    }

    #[test]
    fn valid_capability_binding_rejects_cross_service_and_broad_bindings() {
        let cases = [
            (
                "swapped purpose",
                capability(
                    "singularity-bootstrap",
                    "singularity.most.bootstrap",
                    "brama:agent-42",
                ),
            ),
            (
                "wrong purpose",
                capability(
                    "singularity-bootstrap",
                    "singularity.brama.admin",
                    "brama:agent-42",
                ),
            ),
            (
                "wrong target",
                capability(
                    "singularity",
                    "singularity.brama.bootstrap",
                    "brama:agent-42",
                ),
            ),
            (
                "wildcard resource",
                capability(
                    "singularity-bootstrap",
                    "singularity.brama.bootstrap",
                    "brama:agent-*",
                ),
            ),
            (
                "namespace-only resource",
                capability(
                    "singularity-bootstrap",
                    "singularity.brama.bootstrap",
                    "brama:",
                ),
            ),
            (
                "swapped resource namespace",
                capability(
                    "singularity-bootstrap",
                    "singularity.brama.bootstrap",
                    "most:publisher-42",
                ),
            ),
        ];

        for (name, binding) in cases {
            assert!(
                !valid_capability_binding(&binding, "singularity.brama.bootstrap", "brama:"),
                "{name} must not authorize bootstrap redemption: {binding:?}"
            );
        }
    }

    #[test]
    fn runtime_cleanup_removes_both_credentials_and_the_runtime_directory() {
        let temp = TempDirectory::new();
        let runtime_path = temp.path().join("runtime");
        fs::create_dir(&runtime_path).unwrap();
        let brama_path = runtime_path.join("brama.hmac");
        let most_path = runtime_path.join("most.token");
        fs::write(&brama_path, b"brama-secret").unwrap();
        fs::write(&most_path, b"most-secret").unwrap();

        let cleanup = RuntimeCleanup::new(runtime_path.clone());
        drop(cleanup);

        assert!(!brama_path.exists(), "Brama credential survived cleanup");
        assert!(!most_path.exists(), "Most credential survived cleanup");
        assert!(!runtime_path.exists(), "runtime directory survived cleanup");
    }

    #[test]
    fn materialize_hands_off_exact_secret_in_an_owner_only_file() {
        let temp = TempDirectory::new();
        let socket_path = PathBuf::from("/tmp").join(format!("sb-{}.sock", Uuid::new_v4()));
        let listener = std::os::unix::net::UnixListener::bind(&socket_path).unwrap();
        let broker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            loop {
                let mut byte = [0_u8; 1];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
                if byte[0] == b'\n' {
                    break;
                }
            }
            let request: serde_json::Value = serde_json::from_slice(&request).unwrap();
            assert_eq!("skarbiec.redeem.v1", request["version"]);
            assert_eq!("a".repeat(64), request["capability_id"]);
            assert_eq!("singularity-bootstrap", request["workload_id"]);
            stream
                .write_all(b"{\"version\":\"skarbiec.redeem.v1\",\"status\":\"ok\",\"secret_len\":12}\nexact-secret")
                .unwrap();
        });
        let destination = temp.path().join("brama.hmac");

        materialize(
            &socket_path,
            &"a".repeat(64),
            "singularity-bootstrap",
            &SigningKey::from_bytes(&[9_u8; 32]),
            &destination,
        )
        .unwrap();
        broker.join().unwrap();
        fs::remove_file(&socket_path).unwrap();

        assert_eq!(b"exact-secret", fs::read(&destination).unwrap().as_slice());
        assert_eq!(
            0o600,
            fs::metadata(&destination).unwrap().permissions().mode() & 0o777,
            "materialized credential must be readable and writable only by its workload UID"
        );
    }

    #[test]
    fn require_owner_file_rejects_symlinks() {
        let temp = TempDirectory::new();
        let owner_only_path = temp.path().join("owner-only.key");
        fs::write(&owner_only_path, b"secret").unwrap();
        fs::set_permissions(&owner_only_path, fs::Permissions::from_mode(0o600)).unwrap();
        let symlink_path = temp.path().join("linked.key");
        std::os::unix::fs::symlink(&owner_only_path, &symlink_path).unwrap();

        assert!(
            require_owner_file(&symlink_path).is_err(),
            "a symlink must not be accepted as an owner-only credential file"
        );
    }

    #[test]
    fn require_owner_file_rejects_group_or_world_access() {
        let temp = TempDirectory::new();
        let cases = [("group-readable", 0o640), ("world-readable", 0o604)];

        for (name, mode) in cases {
            let path = temp.path().join(format!("{name}.key"));
            fs::write(&path, b"secret").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();

            assert!(
                require_owner_file(&path).is_err(),
                "{name} credentials must be rejected"
            );
        }
    }
    fn write_owner_only(path: &Path, contents: &[u8]) {
        fs::write(path, contents).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    }

    fn manifest_with_expiry(expires_at: DateTime<Utc>) -> BootstrapManifest {
        BootstrapManifest {
            version: "singularity.bootstrap.v1".to_owned(),
            issued_at: expires_at - Duration::seconds(60),
            expires_at,
            agent_id: "agent-42".to_owned(),
            role: "publisher".to_owned(),
            environment: "test".to_owned(),
            host: "test-host".to_owned(),
            workload_id: "workload-42".to_owned(),
            workload_public_key: "1".repeat(64),
            executable_digest: "2".repeat(64),
            code_digest: "3".repeat(64),
            policy_digest: "4".repeat(64),
            policy_sequence: 1,
            broker_socket: PathBuf::from("/tmp/bootstrap-test-broker.sock"),
            workload_private_key_file: PathBuf::from("/tmp/bootstrap-test-workload.key"),
            singularity_executable: PathBuf::from("/tmp/bootstrap-test-executable"),
            singularity_args: Vec::new(),
            capabilities: BootstrapCapabilities {
                brama: BootstrapCapability {
                    id: "5".repeat(64),
                    target: "singularity-bootstrap".to_owned(),
                    purpose: "singularity.brama.bootstrap".to_owned(),
                    resource: "brama:agent-42".to_owned(),
                },
                most: BootstrapCapability {
                    id: "6".repeat(64),
                    target: "singularity-bootstrap".to_owned(),
                    purpose: "singularity.most.bootstrap".to_owned(),
                    resource: "most:publisher-42".to_owned(),
                },
            },
        }
    }

    #[test]
    fn verify_manifest_requires_an_exact_domain_separated_signature() {
        let temp = TempDirectory::new();
        let trust_root_path = temp.path().join("manifest-trust-root.hex");
        let signature_path = temp.path().join("manifest-signature.hex");
        let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
        write_owner_only(
            &trust_root_path,
            hex::encode(signing_key.verifying_key().as_bytes()).as_bytes(),
        );

        let manifest_bytes = br#"{"version":"singularity.bootstrap.v1","agent_id":"agent-42"}"#;
        let mut domain_separated = Vec::with_capacity(MANIFEST_DOMAIN.len() + manifest_bytes.len());
        domain_separated.extend_from_slice(MANIFEST_DOMAIN);
        domain_separated.extend_from_slice(manifest_bytes);
        let valid_signature = signing_key.sign(&domain_separated);
        write_owner_only(
            &signature_path,
            hex::encode(valid_signature.to_bytes()).as_bytes(),
        );

        assert!(
            verify_manifest(manifest_bytes, &signature_path, &trust_root_path).is_ok(),
            "the exact domain-separated manifest bytes must verify"
        );

        let tampered_bytes = br#"{"version":"singularity.bootstrap.v1","agent_id":"agent-43"}"#;
        assert!(
            verify_manifest(tampered_bytes, &signature_path, &trust_root_path).is_err(),
            "changing the signed manifest bytes must invalidate the signature"
        );

        let unscoped_signature = signing_key.sign(manifest_bytes);
        write_owner_only(
            &signature_path,
            hex::encode(unscoped_signature.to_bytes()).as_bytes(),
        );
        assert!(
            verify_manifest(manifest_bytes, &signature_path, &trust_root_path).is_err(),
            "a signature over the payload without MANIFEST_DOMAIN must be rejected"
        );
    }

    #[test]
    fn validate_manifest_rejects_expiry_and_excessive_lifetime() {
        let now = Utc::now();
        let expired = manifest_with_expiry(now - Duration::seconds(1));
        let mut excessive_lifetime = manifest_with_expiry(now + Duration::seconds(250));
        excessive_lifetime.issued_at =
            excessive_lifetime.expires_at - Duration::seconds(MAX_MANIFEST_LIFETIME + 1);
        let cases = [
            ("expired", expired),
            ("lifetime over 300 seconds", excessive_lifetime),
        ];

        for (name, manifest) in cases {
            assert!(
                validate_manifest(&manifest).is_err(),
                "bootstrap must reject a manifest that is {name}"
            );
        }
    }

    #[test]
    fn verify_executable_requires_the_exact_sha256_of_a_regular_file() {
        let temp = TempDirectory::new();
        let executable_path = temp.path().join("singularity-test-executable");
        let executable_bytes = b"deterministic executable fixture\n";
        write_owner_only(&executable_path, executable_bytes);
        let expected = hex::encode(Sha256::digest(executable_bytes));

        assert!(
            verify_executable(&executable_path, &expected).is_ok(),
            "a regular file with the exact expected SHA-256 must verify"
        );

        let wrong_digest = hex::encode(Sha256::digest(b"different executable bytes"));
        assert!(
            verify_executable(&executable_path, &wrong_digest).is_err(),
            "an executable whose bytes do not match the manifest digest must be rejected"
        );
    }
}
