//! Exact commits and non-forced publication from the canonical main checkout.
use serde_json::{Value, json};

use super::checks::*;
use super::repository::*;
use super::*;
use crate::repo_surface::command::{git, git_network};
use crate::repo_surface::policy::validate_id;
use crate::repo_surface::{SurfaceError, SurfaceResult};

impl RepoService {
    pub(super) async fn commit_create(&self, input: CommitCreate) -> SurfaceResult<Value> {
        validate_id("request_id", &input.request_id)?;
        let _request_lock = self.state.lock_request(&input.request_id)?;
        validate_commit_message(&input.message)?;
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let fp = request_fingerprint(
            "commit_create",
            &json!({"workspace_id":input.workspace_id,"message":input.message}),
        )?;
        if let Some(value) =
            self.replay(&input.request_id, "commit_create", &input.workspace_id, &fp)?
        {
            return Ok(value);
        }
        let mut state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        enforce_changed_paths(repo, &state.worktree).await?;
        let sealed = fresh_seal(&state).await?;
        for required in &repo.required_checks {
            let evidence = state.checks.get(required).ok_or_else(|| {
                SurfaceError::conflict(format!("required check {required:?} has not run"))
            })?;
            if !evidence.succeeded || evidence.fingerprint != sealed {
                return Err(SurfaceError::conflict(format!(
                    "required check {required:?} lacks successful exact evidence"
                )));
            }
        }
        let commit = reconcile_commit(&state, &input.message, &input.request_id, &sealed).await?;
        let clean = successful(
            git(
                &state.worktree,
                &["status", "--porcelain=v1", "--untracked-files=normal"],
                None,
                30,
            )
            .await?,
            "verify committed workspace",
        )?;
        if !clean.stdout.is_empty() {
            return Err(SurfaceError::conflict(
                "workspace is not clean after commit",
            ));
        }
        state.commit = Some(commit.clone());
        self.state.save_workspace(&state)?;
        let response = json!({"workspace_id":state.id,"commit":commit,"branch":state.branch});
        self.record(&input.request_id, "commit_create", &state.id, fp, &response)?;
        Ok(response)
    }

    pub(super) async fn branch_publish(&self, input: Publish) -> SurfaceResult<Value> {
        validate_id("request_id", &input.request_id)?;
        let _request_lock = self.state.lock_request(&input.request_id)?;
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let fp = request_fingerprint(
            "branch_publish",
            &json!({"workspace_id":input.workspace_id}),
        )?;
        if let Some(value) = self.replay(
            &input.request_id,
            "branch_publish",
            &input.workspace_id,
            &fp,
        )? {
            return Ok(value);
        }
        let mut state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        let commit = committed_head(&state, repo).await?;
        let remote_ref = format!("refs/heads/{}", state.branch);
        let existing = git_network(
            &state.worktree,
            &["ls-remote", "--heads", &repo.remote, &remote_ref],
            60,
        )
        .await?;
        successful_ref(&existing, "reconcile remote branch")?;
        let remote_commit = existing.stdout.split_whitespace().next();
        if remote_commit != Some(commit.as_str()) {
            if remote_commit != Some(state.base_commit.as_str()) {
                return Err(SurfaceError::conflict("remote main differs from the workspace base; no force push was attempted"));
            }
            let refspec = format!("{commit}:{remote_ref}");
            successful(
                git_network(&state.worktree, &["push", "--porcelain", &repo.remote, &refspec], 180).await?,
                "publish canonical main",
            )?;
        }
        let readback = git_network(&state.worktree, &["ls-remote", "--heads", &repo.remote, &remote_ref], 60).await?;
        successful_ref(&readback, "verify published canonical main")?;
        if readback.stdout.split_whitespace().next() != Some(commit.as_str()) {
            return Err(SurfaceError::conflict("remote main does not contain the exact published commit"));
        }
        state.published = true;
        self.state.save_workspace(&state)?;
        let response =
            json!({"workspace_id":state.id,"branch":state.branch,"commit":commit,"published":true});
        self.record(
            &input.request_id,
            "branch_publish",
            &state.id,
            fp,
            &response,
        )?;
        Ok(response)
    }


    pub(super) async fn proposal_status(&self, input: WorkspaceOnly) -> SurfaceResult<Value> {
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let state = self.state.load_workspace(&input.workspace_id)?;
        Ok(status_json(&state))
    }
}
