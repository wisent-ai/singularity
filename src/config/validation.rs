//! What an identity component, a digest, a list and a URL must look like, and how a secret is read.
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use secrecy::SecretString;
use url::Url;

use super::*;
use crate::error::AppError;
pub(super) fn validate_identity_component(value: &str, label: &str) -> Result<(), AppError> {
    if value.is_empty()
        || value.len() > MAX_IDENTITY_COMPONENT_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(AppError::Config(format!(
            "{label} is not a valid immutable identifier"
        )));
    }
    Ok(())
}

pub(super) fn validate_digest(value: &str, label: &str) -> Result<(), AppError> {
    if value.len() != DIGEST_HEX_CHARS
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(AppError::Config(format!(
            "{label} must be {DIGEST_HEX_CHARS} lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

pub(super) fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}
pub fn parse_http_url(value: &str, name: &str) -> Result<Url, AppError> {
    let url = Url::parse(value).map_err(|error| AppError::Config(format!("{name}: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(AppError::Config(format!("{name} must use http or https")));
    }
    Ok(url)
}

pub fn read_secret(path: &PathBuf) -> Result<SecretString, AppError> {
    if !path.is_file() {
        return Err(AppError::Secret(format!(
            "not a regular file: {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(path)?.permissions().mode() & GROUP_OR_OTHER_ACCESS != u32::default() {
            return Err(AppError::Secret(format!(
                "{} must not be group/world accessible",
                path.display()
            )));
        }
    }
    let value = fs::read_to_string(path)?
        .trim_end_matches(['\r', '\n'])
        .to_owned();
    if value.is_empty() {
        return Err(AppError::Secret(format!("{} is empty", path.display())));
    }
    Ok(SecretString::from(value))
}
