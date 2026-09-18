//! The workspace tools that report on a workspace: its diff, its seal and its checks.
use chrono::Utc;
use serde_json::{json, Value};

use super::*;
use crate::repo_surface::command::git;
use crate::repo_surface::state::CheckEvidence;
use crate::repo_surface::{SurfaceError, SurfaceResult};
impl RepoService {
    pub(crate) async fn workspace_diff(&self, input: WorkspaceOnly) -> SurfaceResult<Value> {
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        enforce_changed_paths(repo, &state.worktree).await?;
        let diff = bounded_diff(&state.worktree).await?;
        Ok(json!({"workspace_id":state.id,"diff":diff}))
    }

    pub(crate) async fn workspace_seal(&self, input: WorkspaceOnly) -> SurfaceResult<Value> {
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let mut state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        ensure_mutable(&state)?;
        enforce_changed_paths(repo, &state.worktree).await?;
        let diff = bounded_diff(&state.worktree).await?;
        if diff.is_empty() {
            return Err(SurfaceError::conflict("cannot seal an empty diff"));
        }
        stage_allowed(repo, &state.worktree).await?;
        let fingerprint = write_tree(&state.worktree).await?;
        state.sealed_fingerprint = Some(fingerprint.clone());
        state.checks.clear();
        self.state.save_workspace(&state)?;
        Ok(json!({"workspace_id":state.id,"fingerprint":fingerprint,"diff":diff}))
    }

    pub(crate) async fn workspace_check(&self, input: WorkspaceCheck) -> SurfaceResult<Value> {
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let mut state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        ensure_mutable(&state)?;
        let sealed = fresh_seal(&state).await?;
        let check = repo
            .checks
            .get(&input.check)
            .ok_or_else(|| SurfaceError::policy("check is not allowlisted"))?;
        let output = git(
            &state.worktree,
            &[
                "diff",
                "--cached",
                "--check",
                "--no-ext-diff",
                "--no-textconv",
            ],
            None,
            check.timeout_secs,
        )
        .await?;
        let after = fresh_seal(&state).await?;
        if after != sealed {
            return Err(SurfaceError::conflict("check modified the sealed index"));
        }
        enforce_changed_paths(repo, &state.worktree).await?;
        let exit_code = output.code.ok_or_else(|| {
            SurfaceError::conflict(format!(
                "check {} was terminated by a signal and reported no exit code",
                input.check
            ))
        })?;
        let evidence = CheckEvidence {
            fingerprint: sealed,
            exit_code,
            succeeded: output.success,
            checked_at: Utc::now().to_rfc3339(),
            stdout: output.stdout,
            stderr: output.stderr,
            truncated: output.truncated,
        };
        state.checks.insert(input.check.clone(), evidence.clone());
        self.state.save_workspace(&state)?;
        Ok(json!({"workspace_id":state.id,"check":input.check,"evidence":evidence}))
    }
}
