//! Owner-only files, the runtime root, and the hex keys and digests read from disk.
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use zeroize::Zeroize;

use super::*;
use crate::config::GROUP_OR_OTHER_ACCESS;
use crate::error::AppError;
pub(super) fn read_control_line(stream: &mut UnixStream) -> Result<Vec<u8>, AppError> {
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

pub(super) fn require_owner_file(path: &Path) -> Result<(), AppError> {
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

pub(super) fn prepare_runtime_root(path: &Path) -> Result<(), AppError> {
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

pub(super) fn valid_atom(value: &str, max_len: usize) -> bool {
    !value.is_empty() && value.len() <= max_len && value == value.trim() && !value.contains('\0')
}

pub(super) fn valid_capability_binding(
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

pub(super) fn read_hex_32(path: &Path, label: &str) -> Result<[u8; KEY_BYTES], AppError> {
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

pub(super) fn read_hex_64(path: &Path) -> Result<[u8; 64], AppError> {
    require_owner_file(path)?;
    let text = fs::read_to_string(path)?;
    let decoded = hex::decode(text.trim())
        .map_err(|_| AppError::Config("invalid bootstrap manifest signature".into()))?;
    decoded
        .try_into()
        .map_err(|_| AppError::Config("invalid bootstrap manifest signature".into()))
}

pub(super) fn is_lower_hex_64(value: &str) -> bool {
    value.len() == KEY_HEX_CHARS
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

pub(super) fn verify_executable(path: &Path, expected: &str) -> Result<(), AppError> {
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

pub(super) struct RuntimeCleanup {
    directory: PathBuf,
}

impl RuntimeCleanup {
    pub(super) fn new(directory: PathBuf) -> Self {
        Self { directory }
    }

    pub(super) fn path(&self) -> &Path {
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

pub(super) fn secure_remove(path: &Path) {
    if let Ok(metadata) = fs::metadata(path) {
        if let Ok(mut file) = OpenOptions::new().write(true).open(path) {
            let zeros = vec![0_u8; metadata.len().min(MAX_SECRET_BYTES as u64) as usize];
            let _ = file.write_all(&zeros);
            let _ = file.sync_all();
        }
    }
    let _ = fs::remove_file(path);
}
