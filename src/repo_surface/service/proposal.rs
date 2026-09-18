//! The proposal tools: the commit, the published branch, the pull request, the status.
use serde_json::{json, Value};

use super::checks::*;
use super::repository::*;
use super::*;
use crate::repo_surface::command::{gh, git, git_network};
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
        ensure_mutable(&state)?;
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
        successful(
            git(
                &state.worktree,
                &[
                    "commit",
                    "--no-verify",
                    "--no-gpg-sign",
                    "-m",
                    &input.message,
                ],
                None,
                120,
            )
            .await?,
            "create commit",
        )?;
        let committed_tree = successful(
            git(&state.worktree, &["rev-parse", "HEAD^{tree}"], None, 30).await?,
            "resolve committed tree",
        )?
        .stdout
        .trim()
        .to_owned();
        if committed_tree != sealed {
            return Err(SurfaceError::conflict(
                "committed tree does not match sealed tree",
            ));
        }
        let commit = successful(
            git(&state.worktree, &["rev-parse", "HEAD"], None, 30).await?,
            "resolve commit",
        )?
        .stdout
        .trim()
        .to_owned();
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
        if let Some(remote_commit) = remote_commit {
            if remote_commit != commit {
                return Err(SurfaceError::conflict(
                    "remote branch exists at a different commit",
                ));
            }
        } else {
            let refspec = format!("{commit}:{remote_ref}");
            successful(
                git_network(
                    &state.worktree,
                    &["push", "--porcelain", &repo.remote, &refspec],
                    180,
                )
                .await?,
                "publish branch",
            )?;
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

    pub(super) async fn pull_request_open(&self, input: PullRequest) -> SurfaceResult<Value> {
        validate_id("request_id", &input.request_id)?;
        let _request_lock = self.state.lock_request(&input.request_id)?;
        validate_pr_text(&input.title, &input.body)?;
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let fp = request_fingerprint(
            "pull_request_open",
            &json!({"workspace_id":input.workspace_id,"title":input.title,"body":input.body}),
        )?;
        if let Some(value) = self.replay(
            &input.request_id,
            "pull_request_open",
            &input.workspace_id,
            &fp,
        )? {
            return Ok(value);
        }
        let mut state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        let commit = committed_head(&state, repo).await?;
        if !state.published {
            return Err(SurfaceError::conflict(
                "branch must be published before opening a pull request",
            ));
        }
        let remote_ref = format!("refs/heads/{}", state.branch);
        let remote = git_network(
            &state.worktree,
            &["ls-remote", "--heads", &repo.remote, &remote_ref],
            60,
        )
        .await?;
        successful_ref(&remote, "reconcile proposal branch before pull request")?;
        if remote.stdout.split_whitespace().next() != Some(commit.as_str()) {
            return Err(SurfaceError::conflict(
                "published proposal branch no longer matches the recorded commit",
            ));
        }
        let list_args = vec![
            "pr".into(),
            "list".into(),
            "--repo".into(),
            repo.github_repository.clone(),
            "--head".into(),
            state.branch.clone(),
            "--state".into(),
            "all".into(),
            "--limit".into(),
            "2".into(),
            "--json".into(),
            "url,state,baseRefName,headRefName,headRefOid,headRepositoryOwner".into(),
        ];
        let listed = successful(
            gh(&state.worktree, &list_args, 60).await?,
            "reconcile pull request",
        )?;
        let matches: Vec<Value> = serde_json::from_str(&listed.stdout)
            .map_err(|e| SurfaceError::command(format!("invalid gh response: {e}")))?;
        if matches.len() > 1 {
            return Err(SurfaceError::conflict(
                "multiple pull requests exist for the proposal branch",
            ));
        }
        let url = if let Some(value) = matches.first() {
            validated_pull_request_url(value, repo, &state, &commit)?
        } else {
            let create_args = vec![
                "pr".into(),
                "create".into(),
                "--repo".into(),
                repo.github_repository.clone(),
                "--base".into(),
                repo.base_branch.clone(),
                "--head".into(),
                state.branch.clone(),
                "--title".into(),
                input.title,
                "--body".into(),
                input.body,
            ];
            let created_url = successful(
                gh(&state.worktree, &create_args, 120).await?,
                "open pull request",
            )?
            .stdout
            .trim()
            .to_owned();
            if created_url.is_empty() {
                return Err(SurfaceError::command("pull request URL is empty"));
            }
            let view_args = vec![
                "pr".into(),
                "view".into(),
                created_url,
                "--repo".into(),
                repo.github_repository.clone(),
                "--json".into(),
                "url,state,baseRefName,headRefName,headRefOid,headRepositoryOwner".into(),
            ];
            let viewed = successful(
                gh(&state.worktree, &view_args, 60).await?,
                "verify created pull request",
            )?;
            let value: Value = serde_json::from_str(&viewed.stdout)
                .map_err(|error| SurfaceError::command(format!("invalid gh response: {error}")))?;
            validated_pull_request_url(&value, repo, &state, &commit)?
        };
        if url.is_empty() {
            return Err(SurfaceError::command("pull request URL is empty"));
        }
        state.pull_request_url = Some(url.clone());
        self.state.save_workspace(&state)?;
        let response = json!({"workspace_id":state.id,"pull_request_url":url,"final_gate":"external CI and human review"});
        self.record(
            &input.request_id,
            "pull_request_open",
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
