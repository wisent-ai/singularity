use crate::AppError;
use std::{env, ffi::OsString, os::unix::fs::MetadataExt, path::PathBuf};

/// Preserve the current Unix principal while exposing the product manager's installed binaries.
pub(crate) fn runtime_paths() -> Result<(OsString, OsString), AppError> {
    let home = env::var_os("HOME").ok_or_else(|| AppError::Config(
        "HOME is required to resolve product-owned CLI installations".into()))?;
    let directory = PathBuf::from(&home);
    if !directory.is_absolute() {
        return Err(AppError::Config("HOME must name an absolute owned directory".into()));
    }
    let metadata = directory.metadata().map_err(|error| AppError::Config(
        format!("cannot inspect HOME {}: {error}", directory.display())))?;
    let owner = unsafe { libc::geteuid() };
    if !metadata.is_dir() || metadata.uid() != owner
        || metadata.mode() & super::GROUP_OR_OTHER_WRITE_ACCESS != 0 {
        return Err(AppError::Config(format!(
            "HOME {} is unsafe: directory={}, owner={}, mode={:o}; expected a directory owned by UID {owner} without group or other write access",
            directory.display(), metadata.is_dir(), metadata.uid(), metadata.mode())));
    }
    let paths = [directory.join(".local/bin"), directory.join(".stado/bin"),
        PathBuf::from("/usr/bin"), PathBuf::from("/bin"), PathBuf::from("/usr/sbin"),
        PathBuf::from("/sbin"), PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin")];
    let path = env::join_paths(paths).map_err(|error| AppError::Config(format!("invalid product executable path: {error}")))?;
    Ok((home, path))
}
