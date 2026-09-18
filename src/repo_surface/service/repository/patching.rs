//! The paths a patch touches, and the diff a workspace answers with.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::*;
use crate::repo_surface::command::git;
use crate::repo_surface::policy::validate_relative_path;
use crate::repo_surface::{SurfaceError, SurfaceResult};
pub(crate) fn patch_paths(patch: &str) -> SurfaceResult<BTreeSet<PathBuf>> {
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

pub(crate) fn insert_patch_path(
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

pub(crate) async fn bounded_diff(worktree: &Path) -> SurfaceResult<String> {
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
