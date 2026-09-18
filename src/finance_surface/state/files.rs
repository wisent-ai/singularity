//! Owner-only files and directories, and the bytes written into them.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::finance_surface::policy::GROUP_OR_OTHER_ACCESS;
use crate::finance_surface::{SurfaceError, SurfaceResult};

pub(crate) fn canonical_bytes<T: Serialize>(value: &T) -> SurfaceResult<Vec<u8>> {
    let value = serde_json::to_value(value)
        .map_err(|e| SurfaceError::internal(format!("cannot canonicalize journal: {e}")))?;
    crate::finance_surface::policy::canonical_json(&value)
}

pub(super) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, kind: &str) -> SurfaceResult<T> {
    crate::finance_surface::policy::require_owner_only_file(path)?;
    let bytes =
        fs::read(path).map_err(|e| SurfaceError::state(format!("cannot read {kind}: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| SurfaceError::state(format!("invalid {kind}: {e}")))
}

#[cfg(unix)]
pub(super) fn ensure_owner_dir(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    if !path.exists() {
        let mut b = fs::DirBuilder::new();
        b.recursive(true).mode(0o700);
        b.create(path)
            .map_err(|e| SurfaceError::state(format!("cannot create state directory: {e}")))?;
    }
    let m = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::state(format!("cannot stat state directory: {e}")))?;
    if m.file_type().is_symlink()
        || !m.is_dir()
        || m.uid() != unsafe { geteuid() }
        || m.mode() & GROUP_OR_OTHER_ACCESS != 0
    {
        return Err(SurfaceError::policy(
            "state and WORM directories must be owner-only, current-user-owned, and not symlinks",
        ));
    }
    Ok(())
}
#[cfg(not(unix))]
pub(super) fn ensure_owner_dir(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}
#[cfg(unix)]
pub(super) fn require_owner_dir(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::MetadataExt;
    let m = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::state(format!("external WORM sink is not provisioned: {e}")))?;
    if m.file_type().is_symlink()
        || !m.is_dir()
        || m.uid() != unsafe { geteuid() }
        || m.mode() & GROUP_OR_OTHER_ACCESS != 0
    {
        return Err(SurfaceError::policy(
            "external WORM sink must be owner-only, current-user-owned, and not a symlink",
        ));
    }
    Ok(())
}
#[cfg(not(unix))]
pub(super) fn require_owner_dir(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}

#[cfg(unix)]
pub(super) fn create_new_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    sync_parent(path)
}

#[cfg(unix)]
pub(super) fn sync_parent(path: &Path) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("path has no parent"))?;
    fs::File::open(parent)?.sync_all()
}

pub(super) fn audit_hash(
    sequence: u64,
    timestamp: DateTime<Utc>,
    event: &Value,
    previous_hash: &str,
) -> SurfaceResult<String> {
    let value = serde_json::json!({"event":event,"previous_hash":previous_hash,"sequence":sequence,"timestamp":timestamp});
    Ok(hex::encode(Sha256::digest(
        crate::finance_surface::policy::canonical_json(&value)?,
    )))
}

#[cfg(unix)]
pub(super) fn atomic_json<T: Serialize>(
    path: &Path,
    value: &T,
    new_only: bool,
) -> SurfaceResult<()> {
    let bytes = serde_json::to_vec(value)
        .map_err(|e| SurfaceError::internal(format!("cannot serialize state: {e}")))?;
    if new_only {
        return create_new_bytes(path, &bytes).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                SurfaceError::conflict("record already exists")
            } else {
                SurfaceError::state(format!("cannot persist state: {e}"))
            }
        });
    }
    let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    create_new_bytes(&tmp, &bytes)
        .map_err(|e| SurfaceError::state(format!("cannot persist temporary state: {e}")))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        SurfaceError::state(format!("cannot install state: {e}"))
    })?;
    sync_parent(path).map_err(|e| SurfaceError::state(format!("cannot sync state directory: {e}")))
}
#[cfg(not(unix))]
pub(super) fn atomic_json<T: Serialize>(
    _path: &Path,
    _value: &T,
    _new_only: bool,
) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}

#[cfg(unix)]
pub(super) const LOCK_EX: i32 = 2;
#[cfg(unix)]
pub(super) const LOCK_UN: i32 = 8;
#[cfg(unix)]
unsafe extern "C" {
    pub(super) fn flock(fd: i32, operation: i32) -> i32;
    pub(super) fn geteuid() -> u32;
}
