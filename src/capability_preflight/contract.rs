//! The deployment contract a capability-isolated process is checked against:
//! which files it reads, the modes and owners they must have, and the release
//! binaries whose digests it must match.

use std::fs;
use std::io::Read;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

/// The capability-contract workloads an agent unit may run as.
pub(super) const CONTRACT_TARGETS: [&str; 4] =
    ["brama", "most-service", "singularity-bootstrap", "weles"];
pub(super) const CLIENT_GROUP: &str = "skarbiec-capability-clients";
/// The exact, ordered names the signed Skarbiec MCP policy passes through.
pub(super) const SKARBIEC_MCP_ENV_NAMES: [&str; 12] = [
    "SKARBIEC_VAULT_FILE",
    "SKARBIEC_CAP_POLICY",
    "SKARBIEC_CAP_POLICY_SIG",
    "SKARBIEC_CAP_TRUST_ROOT",
    "SKARBIEC_WORKLOAD_REGISTRY",
    "SKARBIEC_WORKLOAD_REGISTRY_SIG",
    "SKARBIEC_CAP_STATE",
    "SKARBIEC_CAP_SOCKET",
    "SKARBIEC_WORM_RECEIPT_DIR",
    "SKARBIEC_WORM_CHECKPOINT",
    "SKARBIEC_WORM_RECEIPT_COMMAND",
    "SKARBIEC_MCP_AGENT_ID",
];
// The modes a capability-isolated deployment is held to: secret files grant the
// group and others nothing, private directories are 0700, shared ones 0750,
// executables are not group/world writable and are owner-executable, and the
// broker socket is 0660.
const GROUP_OR_OTHER_ACCESS: u32 = 0o077;
const PRIVATE_DIR_MODE: u32 = 0o700;
const SHARED_DIR_MODE: u32 = 0o750;
const GROUP_OR_OTHER_WRITE: u32 = 0o022;
const OWNER_EXECUTE: u32 = 0o100;
pub(super) const BROKER_SOCKET_MODE: u32 = 0o660;
const MODE_BITS: u32 = 0o7777;

#[derive(Clone, Copy)]
pub(super) enum Kind {
    File,
    Dir,
    SharedDir,
    Executable,
}

pub(super) type Check<T> = Result<T, String>;

pub(super) fn is_digest(value: &str) -> bool {
    value.len() == Sha256::output_size() * 2
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

pub(super) fn require_absolute(path: &Path, label: &str) -> Check<()> {
    if !path.is_absolute() || path.components().any(|part| part == Component::ParentDir) {
        return Err(format!("{label} must be an absolute normalized path"));
    }
    Ok(())
}

/// Refuses a symlink anywhere along the path; stops at the first component
/// that does not exist yet.
pub(super) fn reject_symlinks(path: &Path) -> Check<()> {
    let mut current = PathBuf::new();
    for part in path.components() {
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(format!(
                    "symlink path component is forbidden: {}",
                    current.display()
                ))
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    Ok(())
}

pub(super) fn require_secure(
    path: &Path,
    kind: Kind,
    owner: Option<u32>,
    group: Option<u32>,
) -> Check<()> {
    require_absolute(path, &path.display().to_string())?;
    reject_symlinks(path)?;
    let info =
        fs::symlink_metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if info.file_type().is_symlink() {
        return Err(format!("symlink is forbidden: {}", path.display()));
    }
    if owner.is_some_and(|uid| info.uid() != uid) {
        return Err(format!(
            "wrong owner for {}: expected uid {}",
            path.display(),
            owner.unwrap_or_default()
        ));
    }
    if group.is_some_and(|gid| info.gid() != gid) {
        return Err(format!(
            "wrong group for {}: expected gid {}",
            path.display(),
            group.unwrap_or_default()
        ));
    }
    let mode = info.permissions().mode() & MODE_BITS;
    let refusal = match kind {
        Kind::File if !info.is_file() || mode & GROUP_OR_OTHER_ACCESS != 0 => {
            "owner-only regular file required"
        }
        Kind::Dir if !info.is_dir() || mode != PRIVATE_DIR_MODE => "0700 directory required",
        Kind::SharedDir if !info.is_dir() || mode != SHARED_DIR_MODE => {
            "0750 shared directory required"
        }
        Kind::Executable
            if !info.is_file() || mode & GROUP_OR_OTHER_WRITE != 0 || mode & OWNER_EXECUTE == 0 =>
        {
            "non-writable owner-executable regular file required"
        }
        _ => return Ok(()),
    };
    Err(format!("{refusal}: {}", path.display()))
}

pub(super) fn is_socket(path: &Path) -> Check<Option<(u32, bool)>> {
    match fs::symlink_metadata(path) {
        Ok(info) => Ok(Some((
            info.permissions().mode() & MODE_BITS,
            info.file_type().is_socket(),
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}

fn digest(path: &Path) -> Check<String> {
    let mut file = fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut block = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut block)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&block[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// A root-owned release binary whose SHA-256 is the one the deployment names.
pub(super) fn verify_binary(path: &str, expected: &str, label: &str) -> Check<PathBuf> {
    let path = PathBuf::from(path);
    require_secure(&path, Kind::Executable, Some(0), None)?;
    if !is_digest(expected) {
        return Err(format!("{label} digest must be lowercase SHA-256"));
    }
    if digest(&path)? != expected {
        return Err(format!("{label} release binary digest mismatch"));
    }
    Ok(path)
}
