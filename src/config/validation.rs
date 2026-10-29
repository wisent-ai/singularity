//! What an identity component, a digest, a list and a URL must look like, and how a secret is read.
use std::fs;
use std::path::PathBuf;

use secrecy::SecretString;
use url::Url;

use super::*;
use crate::error::AppError;
pub(super) fn validate_identity_component(value: &str, label: &str) -> Result<(), AppError> {
    if value.is_empty()
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
/// A URL whose scheme is one of HTTP's own (`http::uri::Scheme`), the only
/// ones the HTTP client sends to.
pub fn parse_http_url(value: &str, name: &str) -> Result<Url, AppError> {
    let url = Url::parse(value).map_err(|error| AppError::Config(format!("{name}: {error}")))?;
    let schemes = [http::uri::Scheme::HTTP, http::uri::Scheme::HTTPS];
    if !schemes.iter().any(|scheme| scheme.as_str() == url.scheme()) {
        return Err(AppError::Config(format!(
            "{name} must use {} or {}, not {}",
            http::uri::Scheme::HTTP,
            http::uri::Scheme::HTTPS,
            url.scheme()
        )));
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
        let mode = fs::metadata(path)?.permissions().mode();
        if mode & GROUP_OR_OTHER_ACCESS != 0 {
            return Err(AppError::Secret(format!(
                "{} must not be group/world accessible; observed mode {mode:o}; chmod go-rwx it",
                path.display()
            )));
        }
    }
    let value = zeroize::Zeroizing::new(fs::read_to_string(path)?);
    secret_value(&value, &path.display().to_string())
}

pub(crate) fn secret_value(value: &str, source: &str) -> Result<SecretString, AppError> {
    let value = value.trim_end_matches(['\r', '\n']);
    if value.is_empty() {
        return Err(AppError::Secret(format!("{source} is empty")));
    }
    Ok(SecretString::from(value.to_owned()))
}
