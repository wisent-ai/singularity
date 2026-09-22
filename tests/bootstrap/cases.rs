// Bootstrap stories, attached to src/bootstrap.rs.

use super::*;
use chrono::Duration;
use ed25519_dalek::Signer;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;

#[path = "manifest.rs"]
mod manifest;

struct TempDirectory {
    path: PathBuf,
}

impl TempDirectory {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/bootstrap-tests");
        fs::create_dir_all(&root).unwrap();
        let path = root.join(format!("singularity-bootstrap-test-{}", Uuid::new_v4()));
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
