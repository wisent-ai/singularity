use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use super::{SurfaceError, SurfaceResult};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFile {
    pub repositories: BTreeMap<String, RepoPolicy>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepoPolicy {
    pub root: PathBuf,
    pub remote: String,
    pub base_branch: String,
    pub branch_prefix: String,
    pub github_repository: String,
    pub github_head_owner: String,
    pub allowed_paths: Vec<PathBuf>,
    pub checks: BTreeMap<String, CheckPolicy>,
    pub required_checks: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckPolicy {
    pub kind: CheckKind,
    pub timeout_secs: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckKind {
    GitDiffCheck,
}

impl PolicyFile {
    pub fn load(path: &Path) -> SurfaceResult<Self> {
        require_owner_only_file(path)?;
        let bytes =
            fs::read(path).map_err(|e| SurfaceError::policy(format!("cannot read policy: {e}")))?;
        let mut policy: Self = serde_json::from_slice(&bytes)
            .map_err(|e| SurfaceError::policy(format!("invalid policy JSON: {e}")))?;
        if policy.repositories.is_empty() {
            return Err(SurfaceError::policy("policy has no repositories"));
        }
        for (id, repo) in &mut policy.repositories {
            validate_id("repository id", id)?;
            repo.validate()?;
        }
        Ok(policy)
    }
}

impl RepoPolicy {
    fn validate(&mut self) -> SurfaceResult<()> {
        if !self.root.is_absolute() {
            return Err(SurfaceError::policy("repository root must be absolute"));
        }
        self.root = self.root.canonicalize().map_err(|e| {
            SurfaceError::policy(format!("cannot canonicalize repository root: {e}"))
        })?;
        if !self.root.join(".git").exists() {
            return Err(SurfaceError::policy(
                "repository root is not a git checkout",
            ));
        }
        validate_token("remote", &self.remote)?;
        validate_branch("base_branch", &self.base_branch)?;
        validate_branch_prefix(&self.branch_prefix)?;
        validate_github_repository(&self.github_repository)?;
        validate_github_component("github_head_owner", &self.github_head_owner)?;
        let repository_owner = self
            .github_repository
            .split_once('/')
            .expect("validated GitHub repository contains one slash")
            .0;
        if self.github_head_owner != repository_owner {
            return Err(SurfaceError::policy(
                "github_head_owner must match the owner of github_repository",
            ));
        }
        if is_protected_branch(&self.branch_prefix) {
            return Err(SurfaceError::policy(
                "branch_prefix cannot be a protected branch",
            ));
        }
        if self.allowed_paths.is_empty() {
            return Err(SurfaceError::policy("allowed_paths must not be empty"));
        }
        for path in &self.allowed_paths {
            validate_relative_path(path)?;
        }
        let required: BTreeSet<_> = self.required_checks.iter().collect();
        if required.len() != self.required_checks.len() {
            return Err(SurfaceError::policy("required_checks contains duplicates"));
        }
        for check in &self.required_checks {
            if !self.checks.contains_key(check) {
                return Err(SurfaceError::policy(format!(
                    "required check {check:?} is not defined"
                )));
            }
        }
        for (name, check) in &self.checks {
            validate_id("check name", name)?;
            if !matches!(check.kind, CheckKind::GitDiffCheck) {
                return Err(SurfaceError::policy(format!(
                    "check {name:?} has unsupported kind"
                )));
            }
            if check.timeout_secs == 0 || check.timeout_secs > 3600 {
                return Err(SurfaceError::policy(format!(
                    "check {name:?} timeout_secs must be 1..=3600"
                )));
            }
        }
        Ok(())
    }

    pub fn path_allowed(&self, path: &Path) -> bool {
        validate_relative_path(path).is_ok()
            && self
                .allowed_paths
                .iter()
                .any(|allowed| path == allowed || path.starts_with(allowed))
    }
}

fn validate_github_repository(value: &str) -> SurfaceResult<()> {
    let Some((owner, repository)) = value.split_once('/') else {
        return Err(SurfaceError::policy("github_repository must be owner/name"));
    };
    if repository.contains('/') {
        return Err(SurfaceError::policy(
            "github_repository must contain exactly one slash",
        ));
    }
    validate_github_component("GitHub owner", owner)?;
    validate_github_component("GitHub repository", repository)
}

fn validate_github_component(kind: &str, value: &str) -> SurfaceResult<()> {
    if value.is_empty()
        || value.len() > 100
        || value.starts_with('-')
        || value.starts_with('.')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(SurfaceError::policy(format!("invalid {kind}")));
    }
    Ok(())
}

pub fn validate_id(kind: &str, value: &str) -> SurfaceResult<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        return Err(SurfaceError::invalid(format!("invalid {kind}")));
    }
    Ok(())
}

pub fn validate_token(kind: &str, value: &str) -> SurfaceResult<()> {
    if value.is_empty()
        || value.len() > 256
        || value.starts_with('-')
        || value.contains('\0')
        || value.chars().any(char::is_whitespace)
    {
        return Err(SurfaceError::policy(format!("invalid {kind}")));
    }
    Ok(())
}

pub fn validate_branch(kind: &str, value: &str) -> SurfaceResult<()> {
    validate_token(kind, value)?;
    if value.contains("..")
        || value.contains("@{")
        || value.ends_with('.')
        || value.ends_with('/')
        || value.contains("//")
        || value.contains('~')
        || value.contains('^')
        || value.contains(':')
        || value.contains('?')
        || value.contains('*')
        || value.contains('[')
        || value.contains('\\')
    {
        return Err(SurfaceError::policy(format!("invalid {kind}")));
    }
    Ok(())
}

fn validate_branch_prefix(value: &str) -> SurfaceResult<()> {
    let prefix = value
        .strip_suffix('/')
        .ok_or_else(|| SurfaceError::policy("branch_prefix must end with /"))?;
    validate_branch("branch_prefix", prefix)
}

pub fn is_protected_branch(value: &str) -> bool {
    matches!(value.trim_end_matches('/'), "main" | "master")
}

pub fn validate_relative_path(path: &Path) -> SurfaceResult<()> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(SurfaceError::invalid("path must be non-empty and relative"));
    }
    for component in path.components() {
        match component {
            Component::Normal(name) if name != ".git" => {}
            _ => return Err(SurfaceError::invalid("path contains a forbidden component")),
        }
    }
    Ok(())
}

#[cfg(unix)]
pub fn require_owner_only_file(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::policy(format!("cannot stat policy: {e}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(SurfaceError::policy(
            "policy must be a regular, non-symlink file",
        ));
    }
    if metadata.uid() != unsafe { libc_geteuid() } || metadata.mode() & 0o077 != 0 {
        return Err(SurfaceError::policy(
            "policy must be owned by the current user and mode 0600 (or stricter)",
        ));
    }
    Ok(())
}

#[cfg(unix)]
unsafe extern "C" {
    fn geteuid() -> u32;
}
#[cfg(unix)]
unsafe fn libc_geteuid() -> u32 {
    unsafe { geteuid() }
}

#[cfg(not(unix))]
pub fn require_owner_only_file(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy(
        "owner-only policy enforcement requires Unix",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("wisent-policy-test-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(path.join(".git")).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn warsztat_contract_policy_binds_head_owner_to_repository_owner() {
        let root = TestDirectory::new();
        let mut policy = RepoPolicy {
            root: root.0.clone(),
            remote: "origin".into(),
            base_branch: "main".into(),
            branch_prefix: "proposal/".into(),
            github_repository: "wisent-ai/singularity".into(),
            github_head_owner: "attacker".into(),
            allowed_paths: vec![PathBuf::from("src")],
            checks: BTreeMap::new(),
            required_checks: Vec::new(),
        };

        let error = policy.validate().unwrap_err();

        assert_eq!(
            error.to_string(),
            "policy_denied: github_head_owner must match the owner of github_repository"
        );
    }
}
