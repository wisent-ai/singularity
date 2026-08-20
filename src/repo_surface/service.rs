use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use super::command::{gh, git, git_network};
use super::policy::{
    PolicyFile, RepoPolicy, is_protected_branch, validate_branch, validate_id,
    validate_relative_path,
};
use super::state::{CheckEvidence, RequestRecord, StateStore, WorkspaceState};
use super::{SurfaceError, SurfaceResult};

const PATCH_CAP: usize = 1024 * 1024;
const DIFF_CAP: usize = 1024 * 1024;
const READ_CAP: usize = 256 * 1024;

#[derive(Clone)]
pub struct RepoService {
    policy: PolicyFile,
    state: StateStore,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceCreate {
    repo_id: String,
    workspace_id: String,
    request_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceRead {
    workspace_id: String,
    path: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspacePatch {
    workspace_id: String,
    patch: String,
    request_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceOnly {
    workspace_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceCheck {
    workspace_id: String,
    check: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommitCreate {
    workspace_id: String,
    message: String,
    request_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Publish {
    workspace_id: String,
    request_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PullRequest {
    workspace_id: String,
    title: String,
    body: String,
    request_id: String,
}

impl RepoService {
    pub fn new(policy: PolicyFile, state: StateStore) -> Self {
        Self { policy, state }
    }

    pub async fn call(&self, name: &str, arguments: Value) -> SurfaceResult<Value> {
        match name {
            "workspace_create" => self.workspace_create(parse(arguments)?).await,
            "workspace_read" => self.workspace_read(parse(arguments)?).await,
            "workspace_apply_patch" => self.workspace_apply_patch(parse(arguments)?).await,
            "workspace_diff" => self.workspace_diff(parse(arguments)?).await,
            "workspace_seal" => self.workspace_seal(parse(arguments)?).await,
            "workspace_check" => self.workspace_check(parse(arguments)?).await,
            "commit_create" => self.commit_create(parse(arguments)?).await,
            "branch_publish" => self.branch_publish(parse(arguments)?).await,
            "pull_request_open" => self.pull_request_open(parse(arguments)?).await,
            "proposal_status" => self.proposal_status(parse(arguments)?).await,
            _ => Err(SurfaceError::invalid("unknown tool")),
        }
    }

    fn repo<'a>(&'a self, state: &WorkspaceState) -> SurfaceResult<&'a RepoPolicy> {
        self.policy
            .repositories
            .get(&state.repo_id)
            .ok_or_else(|| SurfaceError::policy("workspace repository is no longer allowed"))
    }

    async fn workspace_create(&self, input: WorkspaceCreate) -> SurfaceResult<Value> {
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

    async fn workspace_read(&self, input: WorkspaceRead) -> SurfaceResult<Value> {
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

    async fn workspace_apply_patch(&self, input: WorkspacePatch) -> SurfaceResult<Value> {
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

    async fn workspace_diff(&self, input: WorkspaceOnly) -> SurfaceResult<Value> {
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let state = self.state.load_workspace(&input.workspace_id)?;
        let repo = self.repo(&state)?;
        enforce_changed_paths(repo, &state.worktree).await?;
        let diff = bounded_diff(&state.worktree).await?;
        Ok(json!({"workspace_id":state.id,"diff":diff}))
    }

    async fn workspace_seal(&self, input: WorkspaceOnly) -> SurfaceResult<Value> {
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

    async fn workspace_check(&self, input: WorkspaceCheck) -> SurfaceResult<Value> {
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
        let evidence = CheckEvidence {
            fingerprint: sealed,
            exit_code: output.code.unwrap_or(-1),
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

    async fn commit_create(&self, input: CommitCreate) -> SurfaceResult<Value> {
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

    async fn branch_publish(&self, input: Publish) -> SurfaceResult<Value> {
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

    async fn pull_request_open(&self, input: PullRequest) -> SurfaceResult<Value> {
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

    async fn proposal_status(&self, input: WorkspaceOnly) -> SurfaceResult<Value> {
        let _workspace_lock = self.state.lock_workspace(&input.workspace_id)?;
        let state = self.state.load_workspace(&input.workspace_id)?;
        Ok(status_json(&state))
    }

    fn replay(
        &self,
        request_id: &str,
        operation: &str,
        workspace_id: &str,
        fingerprint: &str,
    ) -> SurfaceResult<Option<Value>> {
        let Some(record) = self.state.load_request(request_id)? else {
            return Ok(None);
        };
        if record.operation != operation
            || record.workspace_id != workspace_id
            || record.input_fingerprint != fingerprint
        {
            return Err(SurfaceError::conflict(
                "request_id was already used for different input",
            ));
        }
        Ok(Some(record.response))
    }
    fn record(
        &self,
        request_id: &str,
        operation: &str,
        workspace_id: &str,
        input_fingerprint: String,
        response: &Value,
    ) -> SurfaceResult<()> {
        self.state.save_request(
            request_id,
            &RequestRecord {
                operation: operation.into(),
                workspace_id: workspace_id.into(),
                input_fingerprint,
                response: response.clone(),
            },
        )
    }
}

fn parse<T: for<'de> Deserialize<'de>>(value: Value) -> SurfaceResult<T> {
    serde_json::from_value(value)
        .map_err(|e| SurfaceError::invalid(format!("invalid tool arguments: {e}")))
}
fn successful(
    output: super::command::CommandOutput,
    operation: &str,
) -> SurfaceResult<super::command::CommandOutput> {
    if output.success {
        Ok(output)
    } else {
        Err(SurfaceError::command(format!(
            "{operation} failed (exit {:?}): {}",
            output.code, output.stderr
        )))
    }
}
fn successful_ref<'a>(
    output: &'a super::command::CommandOutput,
    operation: &str,
) -> SurfaceResult<&'a super::command::CommandOutput> {
    if output.success {
        Ok(output)
    } else {
        Err(SurfaceError::command(format!(
            "{operation} failed (exit {:?}): {}",
            output.code, output.stderr
        )))
    }
}
fn ensure_mutable(state: &WorkspaceState) -> SurfaceResult<()> {
    if state.commit.is_some() {
        Err(SurfaceError::conflict("workspace is already committed"))
    } else {
        Ok(())
    }
}
fn request_fingerprint(operation: &str, value: &Value) -> SurfaceResult<String> {
    let bytes = serde_json::to_vec(&(operation, value))
        .map_err(|e| SurfaceError::internal(e.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}
fn status_json(s: &WorkspaceState) -> Value {
    json!({"workspace_id":s.id,"repo_id":s.repo_id,"branch":s.branch,"base_commit":s.base_commit,"sealed_fingerprint":s.sealed_fingerprint,"checks":s.checks,"commit":s.commit,"published":s.published,"pull_request_url":s.pull_request_url,"final_gate":"external CI and human review"})
}

fn ensure_allowed(repo: &RepoPolicy, path: &Path) -> SurfaceResult<()> {
    validate_relative_path(path)?;
    if repo.path_allowed(path) {
        Ok(())
    } else {
        Err(SurfaceError::policy("path is outside allowed_paths"))
    }
}
fn jailed_path(root: &Path, relative: &Path, must_exist: bool) -> SurfaceResult<PathBuf> {
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

async fn changed_paths(worktree: &Path) -> SurfaceResult<Vec<PathBuf>> {
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
        if entry.len() < 4 || entry[2] != b' ' {
            return Err(SurfaceError::command("malformed git status entry"));
        }
        let status = &entry[..2];
        let path = std::str::from_utf8(&entry[3..])
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
async fn enforce_changed_paths(repo: &RepoPolicy, worktree: &Path) -> SurfaceResult<()> {
    for path in changed_paths(worktree).await? {
        ensure_allowed(repo, &path)?;
        reject_symlink_components(worktree, &path)?;
    }
    Ok(())
}
fn reject_symlink_components(root: &Path, relative: &Path) -> SurfaceResult<()> {
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

fn patch_paths(patch: &str) -> SurfaceResult<BTreeSet<PathBuf>> {
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

fn insert_patch_path(
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

async fn bounded_diff(worktree: &Path) -> SurfaceResult<String> {
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

async fn stage_allowed(repo: &RepoPolicy, worktree: &Path) -> SurfaceResult<()> {
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

async fn write_tree(worktree: &Path) -> SurfaceResult<String> {
    let tree = successful(
        git(worktree, &["write-tree"], None, 30).await?,
        "write sealed tree",
    )?
    .stdout
    .trim()
    .to_owned();
    if !matches!(tree.len(), 40 | 64) || !tree.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SurfaceError::command(
            "git write-tree returned an invalid object id",
        ));
    }
    Ok(tree)
}

async fn fresh_seal(state: &WorkspaceState) -> SurfaceResult<String> {
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
fn commit_ready(state: &WorkspaceState, repo: &RepoPolicy) -> SurfaceResult<String> {
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
async fn committed_head(state: &WorkspaceState, repo: &RepoPolicy) -> SurfaceResult<String> {
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
fn validated_pull_request_url(
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

fn validate_commit_message(v: &str) -> SurfaceResult<()> {
    if v.trim() != v
        || v.is_empty()
        || v.len() > 200
        || v.contains('\0')
        || v.contains('\n')
        || v.starts_with('-')
    {
        Err(SurfaceError::invalid(
            "commit message must be a single 1..=200 character line",
        ))
    } else {
        Ok(())
    }
}
fn validate_pr_text(title: &str, body: &str) -> SurfaceResult<()> {
    if title.trim() != title
        || title.is_empty()
        || title.len() > 256
        || title.contains('\0')
        || title.contains('\n')
        || title.starts_with('-')
        || body.len() > 32 * 1024
        || body.contains('\0')
    {
        Err(SurfaceError::invalid("invalid pull request title or body"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("wisent-service-test-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    async fn git_ok(cwd: &Path, args: &[&str]) -> super::super::command::CommandOutput {
        let output = git(cwd, args, None, 30).await.unwrap();
        assert!(
            output.success,
            "git {args:?} failed with stderr: {}",
            output.stderr
        );
        output
    }

    #[tokio::test]
    async fn warsztat_contract_staged_tree_change_invalidates_seal() {
        let directory = TestDirectory::new();
        git_ok(&directory.0, &["init", "--quiet"]).await;
        let tracked = directory.0.join("proposal.txt");
        fs::write(&tracked, "sealed\n").unwrap();
        git_ok(&directory.0, &["add", "--", "proposal.txt"]).await;
        let sealed = write_tree(&directory.0).await.unwrap();
        let state = WorkspaceState {
            id: "workspace".into(),
            repo_id: "repo".into(),
            branch: "proposal/workspace".into(),
            base_commit: String::new(),
            worktree: directory.0.clone(),
            created_at: String::new(),
            sealed_fingerprint: Some(sealed),
            checks: BTreeMap::new(),
            commit: None,
            published: false,
            pull_request_url: None,
        };
        fs::write(tracked, "changed after seal\n").unwrap();
        git_ok(&directory.0, &["add", "--", "proposal.txt"]).await;

        let error = fresh_seal(&state).await.unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid_state: staged tree changed after seal"
        );
    }

    #[test]
    fn warsztat_contract_request_replay_requires_identical_input() {
        let directory = TestDirectory::new();
        let state = StateStore::open(directory.0.join("state")).unwrap();
        let service = RepoService::new(
            PolicyFile {
                repositories: BTreeMap::new(),
            },
            state.clone(),
        );
        let response = json!({"workspace_id":"workspace","applied":true});
        state
            .save_request(
                "request",
                &RequestRecord {
                    operation: "workspace_apply_patch".into(),
                    workspace_id: "workspace".into(),
                    input_fingerprint: "fingerprint".into(),
                    response: response.clone(),
                },
            )
            .unwrap();

        assert_eq!(
            service
                .replay(
                    "request",
                    "workspace_apply_patch",
                    "workspace",
                    "fingerprint"
                )
                .unwrap(),
            Some(response)
        );
        let error = service
            .replay(
                "request",
                "workspace_apply_patch",
                "workspace",
                "different-fingerprint",
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid_state: request_id was already used for different input"
        );
    }
}
