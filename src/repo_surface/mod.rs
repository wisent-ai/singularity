mod command;
mod policy;
mod service;
mod state;

use serde_json::{Value, json};
use std::fmt;
use std::path::PathBuf;

pub use policy::PolicyFile;
pub use service::RepoService;
pub use state::StateStore;

pub type SurfaceResult<T> = Result<T, SurfaceError>;

#[derive(Debug)]
pub struct SurfaceError {
    kind: &'static str,
    message: String,
}
impl SurfaceError {
    pub fn invalid(v: impl Into<String>) -> Self {
        Self {
            kind: "invalid_arguments",
            message: v.into(),
        }
    }
    pub fn policy(v: impl Into<String>) -> Self {
        Self {
            kind: "policy_denied",
            message: v.into(),
        }
    }
    pub fn conflict(v: impl Into<String>) -> Self {
        Self {
            kind: "invalid_state",
            message: v.into(),
        }
    }
    pub fn state(v: impl Into<String>) -> Self {
        Self {
            kind: "state_error",
            message: v.into(),
        }
    }
    pub fn command(v: impl Into<String>) -> Self {
        Self {
            kind: "command_failed",
            message: v.into(),
        }
    }
    pub fn internal(v: impl Into<String>) -> Self {
        Self {
            kind: "internal_error",
            message: v.into(),
        }
    }
    pub fn tool_result(&self) -> Value {
        json!({"content":[{"type":"text","text":format!("{}: {}",self.kind,self.message)}],"isError":true})
    }
}
impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}
impl std::error::Error for SurfaceError {}

pub fn load_from_environment() -> SurfaceResult<RepoService> {
    let policy_path = required_absolute_env("JEDEN_REPO_POLICY_FILE")?;
    let state_dir = required_absolute_env("JEDEN_REPO_STATE_DIR")?;
    Ok(RepoService::new(
        PolicyFile::load(&policy_path)?,
        StateStore::open(state_dir)?,
    ))
}
fn required_absolute_env(name: &str) -> SurfaceResult<PathBuf> {
    let value = std::env::var_os(name)
        .ok_or_else(|| SurfaceError::policy(format!("{name} is required")))?;
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(SurfaceError::policy(format!("{name} must be absolute")));
    }
    Ok(path)
}

pub fn tools() -> Value {
    let object = |properties: Value, required: Value| json!({"type":"object","properties":properties,"required":required,"additionalProperties":false});
    json!([
      {"name":"workspace_create","description":"Create an isolated policy-bound worktree from a clean source repository.","inputSchema":object(json!({"repo_id":{"type":"string"},"workspace_id":{"type":"string"},"request_id":{"type":"string"}}),json!(["repo_id","workspace_id","request_id"]))},
      {"name":"workspace_read","description":"Read one bounded UTF-8 file inside allowed_paths.","inputSchema":object(json!({"workspace_id":{"type":"string"},"path":{"type":"string"}}),json!(["workspace_id","path"]))},
      {"name":"workspace_apply_patch","description":"Apply one bounded unified diff restricted to allowed_paths.","inputSchema":object(json!({"workspace_id":{"type":"string"},"patch":{"type":"string"},"request_id":{"type":"string"}}),json!(["workspace_id","patch","request_id"]))},
      {"name":"workspace_diff","description":"Return the bounded unified diff for the workspace.","inputSchema":object(json!({"workspace_id":{"type":"string"}}),json!(["workspace_id"]))},
      {"name":"workspace_seal","description":"Stage allowed roots and seal the exact Git tree object and bounded diff.","inputSchema":object(json!({"workspace_id":{"type":"string"}}),json!(["workspace_id"]))},
      {"name":"workspace_check","description":"Run the fixed git_diff_check against the exact sealed index and retain tree-bound evidence.","inputSchema":object(json!({"workspace_id":{"type":"string"},"check":{"type":"string"}}),json!(["workspace_id","check"]))},
      {"name":"commit_create","description":"Commit the sealed index without restaging, only after exact successful required evidence.","inputSchema":object(json!({"workspace_id":{"type":"string"},"message":{"type":"string"},"request_id":{"type":"string"}}),json!(["workspace_id","message","request_id"]))},
      {"name":"branch_publish","description":"Reconcile then publish the committed proposal branch without force.","inputSchema":object(json!({"workspace_id":{"type":"string"},"request_id":{"type":"string"}}),json!(["workspace_id","request_id"]))},
      {"name":"pull_request_open","description":"Reconcile then open a pull request to the policy base branch; external CI and human review remain final gates.","inputSchema":object(json!({"workspace_id":{"type":"string"},"title":{"type":"string"},"body":{"type":"string"},"request_id":{"type":"string"}}),json!(["workspace_id","title","body","request_id"]))},
      {"name":"proposal_status","description":"Return durable proposal lifecycle state without changing it.","inputSchema":object(json!({"workspace_id":{"type":"string"}}),json!(["workspace_id"]))}
    ])
}
