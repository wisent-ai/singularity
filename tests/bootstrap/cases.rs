// Bootstrap stories, attached to src/bootstrap.rs.

use super::*;
use chrono::Duration;
use ed25519_dalek::Signer;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;

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
