//! Reading and writing the repository: git invocations, the jail, the paths a patch touches.
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use super::*;
use crate::repo_surface::command::git;
use crate::repo_surface::policy::{validate_relative_path, RepoPolicy};
use crate::repo_surface::state::WorkspaceState;
use crate::repo_surface::{SurfaceError, SurfaceResult};

pub(super) fn parse<T: for<'de> Deserialize<'de>>(value: Value) -> SurfaceResult<T> {
    serde_json::from_value(value)
        .map_err(|e| SurfaceError::invalid(format!("invalid tool arguments: {e}")))
}
pub(super) fn successful(
    output: crate::repo_surface::command::CommandOutput,
    operation: &str,
) -> SurfaceResult<crate::repo_surface::command::CommandOutput> {
    if output.success {
        Ok(output)
    } else {
        Err(SurfaceError::command(format!(
            "{operation} failed (exit {:?}): {}",
            output.code, output.stderr
        )))
    }
}
pub(super) fn successful_ref<'a>(
    output: &'a crate::repo_surface::command::CommandOutput,
    operation: &str,
) -> SurfaceResult<&'a crate::repo_surface::command::CommandOutput> {
    if output.success {
        Ok(output)
    } else {
        Err(SurfaceError::command(format!(
            "{operation} failed (exit {:?}): {}",
            output.code, output.stderr
        )))
    }
}
pub(super) fn ensure_mutable(state: &WorkspaceState) -> SurfaceResult<()> {
    if state.commit.is_some() {
        Err(SurfaceError::conflict("workspace is already committed"))
    } else {
        Ok(())
    }
}
pub(super) fn request_fingerprint(operation: &str, value: &Value) -> SurfaceResult<String> {
    let bytes = serde_json::to_vec(&(operation, value))
        .map_err(|e| SurfaceError::internal(e.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}
pub(super) fn status_json(s: &WorkspaceState) -> Value {
    json!({"workspace_id":s.id,"repo_id":s.repo_id,"branch":s.branch,"base_commit":s.base_commit,"sealed_fingerprint":s.sealed_fingerprint,"checks":s.checks,"commit":s.commit,"published":s.published,"pull_request_url":s.pull_request_url,"final_gate":"external CI and human review"})
}

pub(super) fn ensure_allowed(repo: &RepoPolicy, path: &Path) -> SurfaceResult<()> {
    validate_relative_path(path)?;
    if repo.path_allowed(path) {
        Ok(())
    } else {
        Err(SurfaceError::policy("path is outside allowed_paths"))
    }
}
pub(super) fn jailed_path(
    root: &Path,
    relative: &Path,
    must_exist: bool,
) -> SurfaceResult<PathBuf> {
    validate_relative_path(relative)?;
    let joined = root.join(relative);
    if must_exist {
        let canonical = joined
            .canonicalize()
            .map_err(|e| SurfaceError::invalid(format!("path does not exist: {e}")))?;
        let root = root
            .canonicalize()
            .map_err(|e| SurfaceError::state(format!("invalid worktree: {e}")))?;
        if !canonical.starts_with(&root) {
            return Err(SurfaceError::policy("path escapes workspace"));
        }
        let metadata =
            fs::symlink_metadata(&joined).map_err(|e| SurfaceError::invalid(e.to_string()))?;
        if metadata.file_type().is_symlink() {
            return Err(SurfaceError::policy("symlinks are not allowed"));
        }
        Ok(canonical)
    } else {
        Ok(joined)
    }
}

pub(super) async fn changed_paths(worktree: &Path) -> SurfaceResult<Vec<PathBuf>> {
    let out = successful(
        git(
            worktree,
            &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
            None,
            30,
        )
        .await?,
        "inspect workspace changes",
    )?;
    let bytes = out.stdout.as_bytes();
    let mut at = 0;
    let mut paths = Vec::new();
    while at < bytes.len() {
        let end = bytes[at..]
            .iter()
            .position(|b| *b == 0)
            .map(|n| at + n)
            .ok_or_else(|| SurfaceError::command("malformed git status"))?;
        let entry = &bytes[at..end];
        if entry.len() < MIN_STATUS_ENTRY_BYTES || entry[2] != b' ' {
            return Err(SurfaceError::command("malformed git status entry"));
        }
        let status = &entry[..2];
        let path = std::str::from_utf8(&entry[STATUS_PATH_OFFSET..])
            .map_err(|_| SurfaceError::invalid("non-UTF-8 repository path"))?;
        paths.push(PathBuf::from(path));
        at = end + 1;
        if status.iter().any(|b| matches!(*b, b'R' | b'C')) {
            let next = bytes[at..]
                .iter()
                .position(|b| *b == 0)
                .map(|n| at + n)
                .ok_or_else(|| SurfaceError::command("malformed rename status"))?;
            let old = std::str::from_utf8(&bytes[at..next])
                .map_err(|_| SurfaceError::invalid("non-UTF-8 repository path"))?;
            paths.push(PathBuf::from(old));
            at = next + 1;
        }
    }
    Ok(paths)
}
pub(super) async fn enforce_changed_paths(repo: &RepoPolicy, worktree: &Path) -> SurfaceResult<()> {
    for path in changed_paths(worktree).await? {
        ensure_allowed(repo, &path)?;
        reject_symlink_components(worktree, &path)?;
    }
    Ok(())
}
pub(super) fn reject_symlink_components(root: &Path, relative: &Path) -> SurfaceResult<()> {
    let mut p = root.to_owned();
    for c in relative.components() {
        p.push(c);
        if let Ok(m) = fs::symlink_metadata(&p) {
            if m.file_type().is_symlink() {
                return Err(SurfaceError::policy("changed path contains a symlink"));
            }
        }
    }
    Ok(())
}

pub(super) fn patch_paths(patch: &str) -> SurfaceResult<BTreeSet<PathBuf>> {
    let mut paths = BTreeSet::new();
    let mut saw_diff = false;
    for line in patch.lines() {
        if let Some(raw) = line.strip_prefix("diff --git ") {
            saw_diff = true;
            if raw.contains('\t') {
                return Err(SurfaceError::invalid("malformed diff --git header"));
            }
            let Some((old, new)) = raw.split_once(' ') else {
                return Err(SurfaceError::invalid("malformed diff --git header"));
            };
            if old.is_empty() || new.is_empty() || new.contains(' ') {
                return Err(SurfaceError::invalid(
                    "quoted or spaced patch paths are not accepted",
                ));
            }
            if old == "/dev/null" || new == "/dev/null" {
                return Err(SurfaceError::invalid(
                    "diff --git paths may not be /dev/null",
                ));
            }
            insert_patch_path(&mut paths, old, Some("a/"))?;
            insert_patch_path(&mut paths, new, Some("b/"))?;
        } else if let Some(raw) = line.strip_prefix("--- ") {
            insert_patch_path(&mut paths, raw, Some("a/"))?;
        } else if let Some(raw) = line.strip_prefix("+++ ") {
            insert_patch_path(&mut paths, raw, Some("b/"))?;
        } else if let Some(raw) = line
            .strip_prefix("rename from ")
            .or_else(|| line.strip_prefix("rename to "))
            .or_else(|| line.strip_prefix("copy from "))
            .or_else(|| line.strip_prefix("copy to "))
        {
            insert_patch_path(&mut paths, raw, None)?;
        }
    }
    if !saw_diff || paths.is_empty() {
        return Err(SurfaceError::invalid(
            "patch has no valid diff --git headers",
        ));
    }
    Ok(paths)
}

pub(super) fn insert_patch_path(
    paths: &mut BTreeSet<PathBuf>,
    raw: &str,
    required_prefix: Option<&str>,
) -> SurfaceResult<()> {
    if raw == "/dev/null" {
        return if required_prefix.is_some() {
            Ok(())
        } else {
            Err(SurfaceError::invalid(
                "rename/copy path may not be /dev/null",
            ))
        };
    }
    if raw.is_empty()
        || raw.starts_with('"')
        || raw.contains('\\')
        || raw.chars().any(char::is_whitespace)
    {
        return Err(SurfaceError::invalid(
            "quoted, spaced, or malformed patch paths are not accepted",
        ));
    }
    let value = if let Some(prefix) = required_prefix {
        raw.strip_prefix(prefix)
            .ok_or_else(|| SurfaceError::invalid("patch path has an invalid Git prefix"))?
    } else {
        raw
    };
    let path = PathBuf::from(value);
    validate_relative_path(&path)?;
    paths.insert(path);
    Ok(())
}

pub(super) async fn bounded_diff(worktree: &Path) -> SurfaceResult<String> {
    let tracked_diff = successful(
        git(
            worktree,
            &[
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
            ],
            None,
            60,
        )
        .await?,
        "generate diff",
    )?;
    if tracked_diff.truncated {
        return Err(SurfaceError::conflict("diff exceeds command output limit"));
    }
    let mut out = tracked_diff.stdout;
    for path in changed_paths(worktree).await? {
        if worktree.join(&path).is_file() {
            let text = path
                .to_str()
                .ok_or_else(|| SurfaceError::invalid("non-UTF-8 path"))?;
            let tracked = git(
                worktree,
                &["ls-files", "--error-unmatch", "--", text],
                None,
                30,
            )
            .await?;
            if !tracked.success {
                let diff = git(
                    worktree,
                    &[
                        "diff",
                        "--no-index",
                        "--binary",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--",
                        "/dev/null",
                        text,
                    ],
                    None,
                    60,
                )
                .await?;
                if diff.code != Some(1) && !diff.success {
                    return Err(SurfaceError::command(format!(
                        "generate untracked diff failed: {}",
                        diff.stderr
                    )));
                }
                if diff.truncated {
                    return Err(SurfaceError::conflict("diff exceeds command output limit"));
                }
                out.push_str(&diff.stdout);
            }
        }
        if out.len() > DIFF_CAP {
            return Err(SurfaceError::conflict("diff exceeds 1048576-byte limit"));
        }
    }
    Ok(out)
}
