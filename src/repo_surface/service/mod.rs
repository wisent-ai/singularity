use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;

use crate::repo_surface::policy::{PolicyFile, RepoPolicy};
use crate::repo_surface::state::{RequestRecord, StateStore, WorkspaceState};
use crate::repo_surface::{SurfaceError, SurfaceResult};

const PATCH_CAP: usize = 1024 * 1024;
const DIFF_CAP: usize = 1024 * 1024;
const READ_CAP: usize = 256 * 1024;
/// A porcelain status entry is `XY path`: two status bytes, a space, at least one byte.
const MIN_STATUS_ENTRY_BYTES: usize = 4;
const STATUS_PATH_OFFSET: usize = 3;
/// A git object id is 40 hex characters (SHA-1) or 64 (SHA-256).
const SHA1_HEX_CHARS: usize = 40;
const SHA256_HEX_CHARS: usize = 64;
/// A commit message is one line of at most 200 characters.
const MAX_COMMIT_MESSAGE_BYTES: usize = 200;

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
pub(crate) struct WorkspaceOnly {
    workspace_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkspaceCheck {
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
            "proposal_status" => self.proposal_status(parse(arguments)?).await,
            _ => Err(SurfaceError::invalid("unknown tool")),
        }
    }

    pub(super) fn repo<'a>(&'a self, state: &WorkspaceState) -> SurfaceResult<&'a RepoPolicy> {
        let repo = self.policy.repositories.get(&state.repo_id)
            .ok_or_else(|| SurfaceError::policy("workspace repository is no longer allowed"))?;
        if state.worktree != repo.root || state.branch != repo.base_branch {
            return Err(SurfaceError::policy("workspace is not the canonical main checkout; legacy isolated workspaces are not adopted"));
        }
        self.state.require_repository_owner(&state.repo_id, &state.id)?;
        Ok(repo)
    }

    pub(super) fn replay(
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
    pub(super) fn record(
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

mod checks;

use repository::{parse, status_json};
mod proposal;
mod repository;
mod workspace;

#[cfg(test)]
#[path = "../../../tests/repo_surface/service.rs"]
mod tests;
