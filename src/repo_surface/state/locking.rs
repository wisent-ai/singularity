//! Owner-only directories and the advisory locks a workspace or a request is held under.
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::repo_surface::policy::GROUP_OR_OTHER_ACCESS;
use crate::repo_surface::{SurfaceError, SurfaceResult};

#[cfg(unix)]
pub(super) fn create_owner_dir(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true).mode(0o700);
    builder
        .create(path)
        .map_err(|e| SurfaceError::state(format!("cannot create state directory: {e}")))?;
    require_owner_dir(path)
}

#[cfg(not(unix))]
pub(super) fn create_owner_dir(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}

#[cfg(unix)]
pub(super) fn require_owner_dir(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::state(format!("cannot stat state directory: {e}")))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || metadata.uid() != unsafe { current_euid() }
        || metadata.mode() & GROUP_OR_OTHER_ACCESS != 0
    {
        return Err(SurfaceError::policy(
            "state directory must be owner-only, current-user-owned, and not a symlink",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
pub(super) fn require_owner_dir(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}

#[cfg(unix)]
pub(super) const LOCK_EX: i32 = 2;
#[cfg(unix)]
pub(super) const LOCK_UN: i32 = 8;
#[cfg(unix)]
unsafe extern "C" {
    pub(super) fn flock(fd: i32, operation: i32) -> i32;
}

#[cfg(unix)]
unsafe extern "C" {
    fn geteuid() -> u32;
}
#[cfg(unix)]
unsafe fn current_euid() -> u32 {
    unsafe { geteuid() }
}

pub(super) fn read_owner_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
    kind: &str,
) -> SurfaceResult<T> {
    crate::repo_surface::policy::require_owner_only_file(path)?;
    let bytes =
        fs::read(path).map_err(|e| SurfaceError::state(format!("cannot read {kind}: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| SurfaceError::state(format!("invalid {kind}: {e}")))
}

#[cfg(unix)]
pub(super) fn atomic_owner_json<T: Serialize>(path: &Path, value: &T) -> SurfaceResult<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let bytes = serde_json::to_vec(value)
        .map_err(|e| SurfaceError::internal(format!("cannot serialize state: {e}")))?;
    let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| -> SurfaceResult<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|e| SurfaceError::state(format!("cannot create state file: {e}")))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| SurfaceError::state(format!("cannot persist state file: {e}")))?;
        fs::rename(&tmp, path)
            .map_err(|e| SurfaceError::state(format!("cannot install state file: {e}")))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(unix)]
pub(super) fn atomic_owner_json_new<T: Serialize>(path: &Path, value: &T) -> SurfaceResult<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let bytes = serde_json::to_vec(value)
        .map_err(|error| SurfaceError::internal(format!("cannot serialize state: {error}")))?;
    let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| -> SurfaceResult<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|error| {
                SurfaceError::state(format!("cannot create request record: {error}"))
            })?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| {
                SurfaceError::state(format!("cannot persist request record: {error}"))
            })?;
        fs::hard_link(&tmp, path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                SurfaceError::conflict("request_id already exists")
            } else {
                SurfaceError::state(format!("cannot install request record: {error}"))
            }
        })?;
        fs::remove_file(&tmp).map_err(|error| {
            SurfaceError::state(format!("cannot remove request temp file: {error}"))
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(not(unix))]
pub(super) fn atomic_owner_json_new<T: Serialize>(_path: &Path, _value: &T) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}

#[cfg(not(unix))]
pub(super) fn atomic_owner_json<T: Serialize>(_path: &Path, _value: &T) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}
