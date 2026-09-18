//! What an identifier, an asset code and a protected file must look like.
use std::fs;
use std::path::Path;

use super::*;
use super::{SurfaceError, SurfaceResult};
pub fn validate_id(kind: &str, value: &str) -> SurfaceResult<()> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        return Err(SurfaceError::invalid(format!("invalid {kind}")));
    }
    Ok(())
}

pub fn validate_asset(value: &str) -> SurfaceResult<()> {
    if value.is_empty()
        || value.len() > MAX_ASSET_BYTES
        || !value
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
    {
        return Err(SurfaceError::invalid(
            "asset must be 1..=16 uppercase ASCII letters/digits",
        ));
    }
    Ok(())
}

#[cfg(unix)]
pub fn require_owner_only_file(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::policy(format!("cannot stat protected file: {e}")))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.uid() != unsafe { geteuid() }
        || metadata.mode() & GROUP_OR_OTHER_ACCESS != 0
    {
        return Err(SurfaceError::policy(
            "protected file must be owner-only, current-user-owned, regular, and not a symlink",
        ));
    }
    Ok(())
}

#[cfg(unix)]
unsafe extern "C" {
    fn geteuid() -> u32;
}

#[cfg(not(unix))]
pub fn require_owner_only_file(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy(
        "finance policy enforcement requires Unix",
    ))
}
