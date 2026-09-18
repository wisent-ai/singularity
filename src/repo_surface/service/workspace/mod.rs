//! The workspace tools: creating one, reading a file, applying a patch, the diff, the seal, the checks.
use chrono::Utc;
use serde_json::{json, Value};
use std::fs;

use super::checks::*;
use super::repository::*;
use super::*;
use crate::repo_surface::command::git;
use crate::repo_surface::policy::{is_protected_branch, validate_branch, validate_id};
use crate::repo_surface::state::WorkspaceState;
use crate::repo_surface::{SurfaceError, SurfaceResult};

impl RepoService {
    pub(super) async fn workspace_create(&self, input: WorkspaceCreate) -> SurfaceResult<Value> {
        validate_id("repository id", &input.repo_id)?;
        validate_id("workspace id", &input.workspace_id)?;
        validate_id("request_id", &input.request_id)?;
        let _request_lock = self.state.lock_request(&input.request_id)?;
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let fp = request_fingerprint(
            "workspace_create",
            &json!({"repo_id":input.repo_id,"workspace_id":input.workspace_id}),
        )?;
        if let Some(value) = self.replay(
            &input.request_id,
            "workspace_create",
            &input.workspace_id,
            &fp,
        )? {
            return Ok(value);
        }
        let repo = self
            .policy
            .repositories
            .get(&input.repo_id)
            .ok_or_else(|| SurfaceError::policy("repository is not allowlisted"))?;
        let filters = git(
            &repo.root,
            &[
                "config",
                "--local",
                "--get-regexp",
                "^filter\\..*\\.(clean|smudge|process)$",
            ],
            None,
            30,
        )
        .await?;
        if filters.success && !filters.stdout.trim().is_empty() {
            return Err(SurfaceError::policy(
                "repository config contains executable Git filters",
            ));
        }
        if !filters.success && filters.code != Some(1) {
            return Err(SurfaceError::command(format!(
                "cannot inspect Git filters: {}",
                filters.stderr
            )));
        }
        let status = successful(
            git(
                &repo.root,
                &["status", "--porcelain=v1", "--untracked-files=normal"],
                None,
                30,
            )
            .await?,
            "inspect source repository",
        )?;
        if !status.stdout.is_empty() {
            return Err(SurfaceError::conflict("source repository is not clean"));
        }
        let worktree = self.state.worktree_path(&input.workspace_id)?;
        if worktree.exists() {
            return Err(SurfaceError::conflict("workspace already exists"));
        }
        let branch = format!("{}{}", repo.branch_prefix, input.workspace_id);
        validate_branch("generated branch", &branch)?;
        if is_protected_branch(&branch) || branch == repo.base_branch {
            return Err(SurfaceError::policy("generated branch is protected"));
        }
        let base_ref = format!("{}/{}", repo.remote, repo.base_branch);
        let base = successful(
            git(
                &repo.root,
                &["rev-parse", "--verify", &format!("{base_ref}^{{commit}}")],
                None,
                30,
            )
            .await?,
            "resolve base branch",
        )?
        .stdout
        .trim()
        .to_owned();
        successful(
            git(
                &repo.root,
                &[
                    "worktree",
                    "add",
                    "-b",
                    &branch,
                    worktree
                        .to_str()
                        .ok_or_else(|| SurfaceError::invalid("non-UTF-8 state path"))?,
                    &base,
                ],
                None,
                120,
            )
            .await?,
            "create isolated worktree",
        )?;
        let state = WorkspaceState {
            id: input.workspace_id.clone(),
            repo_id: input.repo_id,
            branch,
            base_commit: base,
            worktree,
            created_at: Utc::now().to_rfc3339(),
            sealed_fingerprint: None,
            checks: Default::default(),
            commit: None,
            published: false,
            pull_request_url: None,
        };
        self.state.save_workspace(&state)?;
        let response = status_json(&state);
        self.record(
            &input.request_id,
            "workspace_create",
            &state.id,
            fp,
            &response,
        )?;
        Ok(response)
    }

    pub(super) async fn workspace_read(&self, input: WorkspaceRead) -> SurfaceResult<Value> {
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        ensure_allowed(repo, &input.path)?;
        reject_symlink_components(&state.worktree, &input.path)?;
        let path = jailed_path(&state.worktree, &input.path, true)?;
        let metadata = fs::symlink_metadata(&path)
            .map_err(|e| SurfaceError::invalid(format!("cannot stat requested file: {e}")))?;
        if !metadata.is_file() || metadata.len() > READ_CAP as u64 {
            return Err(SurfaceError::invalid(
                "requested path is not a bounded regular file",
            ));
        }
        let bytes = fs::read(path)
            .map_err(|e| SurfaceError::state(format!("cannot read workspace file: {e}")))?;
        let content = String::from_utf8(bytes)
            .map_err(|_| SurfaceError::invalid("workspace_read only returns UTF-8 text"))?;
        Ok(json!({"path":input.path,"content":content}))
    }

    pub(super) async fn workspace_apply_patch(
        &self,
        input: WorkspacePatch,
    ) -> SurfaceResult<Value> {
        validate_id("request_id", &input.request_id)?;
        let _request_lock = self.state.lock_request(&input.request_id)?;
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        if input.patch.is_empty() || input.patch.len() > PATCH_CAP {
            return Err(SurfaceError::invalid("patch must be 1..=1048576 bytes"));
        }
        let fp = request_fingerprint(
            "workspace_apply_patch",
            &json!({"workspace_id":input.workspace_id,"patch":input.patch}),
        )?;
        if let Some(value) = self.replay(
            &input.request_id,
            "workspace_apply_patch",
            &input.workspace_id,
            &fp,
        )? {
            return Ok(value);
        }
        let mut state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        ensure_mutable(&state)?;
        let paths = patch_paths(&input.patch)?;
        for path in &paths {
            ensure_allowed(repo, path)?;
            reject_symlink_components(&state.worktree, path)?;
        }
        successful(
            git(
                &state.worktree,
                &[
                    "apply",
                    "--check",
                    "--recount",
                    "--whitespace=error-all",
                    "-",
                ],
                Some(input.patch.as_bytes()),
                60,
            )
            .await?,
            "validate patch",
        )?;
        successful(
            git(
                &state.worktree,
                &["apply", "--recount", "--whitespace=error-all", "-"],
                Some(input.patch.as_bytes()),
                60,
            )
            .await?,
            "apply patch",
        )?;
        if let Err(policy_error) = enforce_changed_paths(repo, &state.worktree).await {
            let rollback = git(
                &state.worktree,
                &[
                    "apply",
                    "--reverse",
                    "--recount",
                    "--whitespace=nowarn",
                    "-",
                ],
                Some(input.patch.as_bytes()),
                60,
            )
            .await?;
            if !rollback.success {
                return Err(SurfaceError::state(format!(
                    "rejected patch could not be rolled back: {}; original error: {}",
                    rollback.stderr, policy_error
                )));
            }
            return Err(policy_error);
        }
        state.sealed_fingerprint = None;
        state.checks.clear();
        self.state.save_workspace(&state)?;
        let response = json!({"workspace_id":state.id,"applied":true});
        self.record(
            &input.request_id,
            "workspace_apply_patch",
            &state.id,
            fp,
            &response,
        )?;
        Ok(response)
    }
}

mod reporting;
