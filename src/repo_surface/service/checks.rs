//! Staging, sealing and the checks a commit and a pull request must pass.
use serde_json::Value;
use std::path::Path;

use super::repository::*;
use super::*;
use crate::repo_surface::command::git;
use crate::repo_surface::policy::{is_protected_branch, RepoPolicy};
use crate::repo_surface::state::WorkspaceState;
use crate::repo_surface::{SurfaceError, SurfaceResult};

pub(super) async fn stage_allowed(repo: &RepoPolicy, worktree: &Path) -> SurfaceResult<()> {
    for path in changed_paths(worktree).await? {
        let text = path
            .to_str()
            .ok_or_else(|| SurfaceError::invalid("non-UTF-8 path"))?;
        let attribute = successful(
            git(worktree, &["check-attr", "filter", "--", text], None, 30).await?,
            "inspect clean filter policy",
        )?;
        if attribute.truncated
            || !attribute
                .stdout
                .trim_end()
                .ends_with(": filter: unspecified")
        {
            return Err(SurfaceError::policy(
                "changed paths with Git clean filters cannot be sealed",
            ));
        }
    }
    let roots: Vec<String> = repo
        .allowed_paths
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    let mut args = vec!["add".to_owned(), "--".to_owned()];
    args.extend(roots);
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    successful(
        git(worktree, &refs, None, 60).await?,
        "stage allowed changes",
    )?;
    Ok(())
}

pub(super) async fn write_tree(worktree: &Path) -> SurfaceResult<String> {
    let tree = successful(
        git(worktree, &["write-tree"], None, 30).await?,
        "write sealed tree",
    )?
    .stdout
    .trim()
    .to_owned();
    if !matches!(tree.len(), SHA1_HEX_CHARS | SHA256_HEX_CHARS)
        || !tree.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(SurfaceError::command(
            "git write-tree returned an invalid object id",
        ));
    }
    Ok(tree)
}

pub(super) async fn fresh_seal(state: &WorkspaceState) -> SurfaceResult<String> {
    let sealed = state
        .sealed_fingerprint
        .clone()
        .ok_or_else(|| SurfaceError::conflict("workspace is not sealed"))?;
    let current = write_tree(&state.worktree).await?;
    if current != sealed {
        return Err(SurfaceError::conflict("staged tree changed after seal"));
    }
    let unstaged = git(
        &state.worktree,
        &["diff", "--quiet", "--no-ext-diff", "--no-textconv", "--"],
        None,
        30,
    )
    .await?;
    if unstaged.code == Some(1) {
        return Err(SurfaceError::conflict("worktree changed after seal"));
    }
    successful(unstaged, "verify sealed worktree")?;
    let untracked = successful(
        git(
            &state.worktree,
            &["ls-files", "--others", "--exclude-standard"],
            None,
            30,
        )
        .await?,
        "verify sealed untracked files",
    )?;
    if untracked.truncated || !untracked.stdout.is_empty() {
        return Err(SurfaceError::conflict(
            "untracked files appeared after seal",
        ));
    }
    Ok(sealed)
}
pub(super) fn commit_ready(state: &WorkspaceState, repo: &RepoPolicy) -> SurfaceResult<String> {
    let commit = state
        .commit
        .clone()
        .ok_or_else(|| SurfaceError::conflict("workspace must be committed first"))?;
    if state.branch == repo.base_branch
        || is_protected_branch(&state.branch)
        || !state.branch.starts_with(&repo.branch_prefix)
    {
        return Err(SurfaceError::policy("workspace branch is not publishable"));
    }
    Ok(commit)
}
pub(super) async fn committed_head(
    state: &WorkspaceState,
    repo: &RepoPolicy,
) -> SurfaceResult<String> {
    let commit = commit_ready(state, repo)?;
    let status = successful(
        git(
            &state.worktree,
            &["status", "--porcelain=v1", "--untracked-files=normal"],
            None,
            30,
        )
        .await?,
        "verify committed workspace",
    )?;
    if !status.stdout.is_empty() {
        return Err(SurfaceError::conflict(
            "committed workspace is no longer clean",
        ));
    }
    let head = successful(
        git(&state.worktree, &["rev-parse", "HEAD"], None, 30).await?,
        "verify committed HEAD",
    )?
    .stdout
    .trim()
    .to_owned();
    if head != commit {
        return Err(SurfaceError::conflict(
            "workspace HEAD no longer matches recorded commit",
        ));
    }
    Ok(commit)
}
pub(super) fn validated_pull_request_url(
    value: &Value,
    repo: &RepoPolicy,
    state: &WorkspaceState,
    commit: &str,
) -> SurfaceResult<String> {
    let head_owner = value
        .get("headRepositoryOwner")
        .and_then(|owner| owner.get("login"))
        .and_then(Value::as_str);
    if value.get("baseRefName").and_then(Value::as_str) != Some(&repo.base_branch)
        || value.get("headRefName").and_then(Value::as_str) != Some(&state.branch)
        || value.get("headRefOid").and_then(Value::as_str) != Some(commit)
        || head_owner != Some(repo.github_head_owner.as_str())
    {
        return Err(SurfaceError::conflict(
            "pull request does not match the policy repository, branches, and commit",
        ));
    }
    if value.get("state").and_then(Value::as_str) != Some("OPEN") {
        return Err(SurfaceError::conflict("proposal pull request is not open"));
    }
    value
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| SurfaceError::command("gh response omitted pull request URL"))
}

pub(super) fn validate_commit_message(v: &str) -> SurfaceResult<()> {
    if v.trim() != v
        || v.is_empty()
        || v.len() > MAX_COMMIT_MESSAGE_BYTES
        || v.contains('\0')
        || v.contains('\n')
        || v.starts_with('-')
    {
        Err(SurfaceError::invalid(&format!(
            "commit message must be a single 1..={MAX_COMMIT_MESSAGE_BYTES} character line"
        )))
    } else {
        Ok(())
    }
}
pub(super) fn validate_pr_text(title: &str, body: &str) -> SurfaceResult<()> {
    if title.trim() != title
        || title.is_empty()
        || title.len() > MAX_PR_TITLE_BYTES
        || title.contains('\0')
        || title.contains('\n')
        || title.starts_with('-')
        || body.len() > MAX_PR_BODY_BYTES
        || body.contains('\0')
    {
        Err(SurfaceError::invalid("invalid pull request title or body"))
    } else {
        Ok(())
    }
}
