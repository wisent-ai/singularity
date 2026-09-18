use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use super::policy::validate_id;
use super::{SurfaceError, SurfaceResult};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceState {
    pub id: String,
    pub repo_id: String,
    pub branch: String,
    pub base_commit: String,
    pub worktree: PathBuf,
    pub created_at: String,
    pub sealed_fingerprint: Option<String>,
    pub checks: BTreeMap<String, CheckEvidence>,
    pub commit: Option<String>,
    pub published: bool,
    pub pull_request_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckEvidence {
    pub fingerprint: String,
    pub exit_code: i32,
    pub succeeded: bool,
    pub checked_at: String,
    pub stdout: String,
    pub stderr: String,
    pub truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestRecord {
    pub operation: String,
    pub workspace_id: String,
    pub input_fingerprint: String,
    pub response: Value,
}

pub struct WorkspaceLock {
    file: fs::File,
}

#[cfg(unix)]
impl Drop for WorkspaceLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        unsafe {
            flock(self.file.as_raw_fd(), LOCK_UN);
        }
    }
}

#[derive(Clone)]
pub struct StateStore {
    root: PathBuf,
}

impl StateStore {
    pub fn open(root: PathBuf) -> SurfaceResult<Self> {
        if !root.is_absolute() {
            return Err(SurfaceError::policy(
                "JEDEN_REPO_STATE_DIR must be absolute",
            ));
        }
        if root.exists() {
            require_owner_dir(&root)?;
        } else {
            create_owner_dir(&root)?;
        }
        for child in [
            "workspaces",
            "records",
            "requests",
            "locks",
            "request-locks",
        ] {
            let path = root.join(child);
            if path.exists() {
                require_owner_dir(&path)?;
            } else {
                create_owner_dir(&path)?;
            }
        }
        Ok(Self { root })
    }

    pub fn worktree_path(&self, id: &str) -> SurfaceResult<PathBuf> {
        validate_id("workspace id", id)?;
        Ok(self.root.join("workspaces").join(id))
    }

    #[cfg(unix)]
    pub fn lock_workspace(&self, id: &str) -> SurfaceResult<WorkspaceLock> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        validate_id("workspace id", id)?;
        let path = self.root.join("locks").join(format!("{id}.lock"));
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
            .map_err(|error| SurfaceError::state(format!("cannot open workspace lock: {error}")))?;
        if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
            return Err(SurfaceError::state(format!(
                "cannot acquire workspace lock: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(WorkspaceLock { file })
    }

    #[cfg(not(unix))]
    pub fn lock_workspace(&self, _id: &str) -> SurfaceResult<WorkspaceLock> {
        Err(SurfaceError::policy("workspace locking requires Unix"))
    }

    #[cfg(unix)]
    pub fn lock_request(&self, request_id: &str) -> SurfaceResult<WorkspaceLock> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        validate_id("request_id", request_id)?;
        let path = self
            .root
            .join("request-locks")
            .join(format!("{request_id}.lock"));
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
            .map_err(|error| SurfaceError::state(format!("cannot open request lock: {error}")))?;
        if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
            return Err(SurfaceError::state(format!(
                "cannot acquire request lock: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(WorkspaceLock { file })
    }

    #[cfg(not(unix))]
    pub fn lock_request(&self, _request_id: &str) -> SurfaceResult<WorkspaceLock> {
        Err(SurfaceError::policy("request locking requires Unix"))
    }

    fn record_path(&self, id: &str) -> SurfaceResult<PathBuf> {
        validate_id("workspace id", id)?;
        Ok(self.root.join("records").join(format!("{id}.json")))
    }

    fn request_path(&self, request_id: &str) -> SurfaceResult<PathBuf> {
        validate_id("request_id", request_id)?;
        Ok(self
            .root
            .join("requests")
            .join(format!("{request_id}.json")))
    }

    pub fn load_workspace(&self, id: &str) -> SurfaceResult<WorkspaceState> {
        read_owner_json(&self.record_path(id)?, "workspace")
    }

    pub fn save_workspace(&self, state: &WorkspaceState) -> SurfaceResult<()> {
        atomic_owner_json(&self.record_path(&state.id)?, state)
    }

    pub fn load_request(&self, request_id: &str) -> SurfaceResult<Option<RequestRecord>> {
        let path = self.request_path(request_id)?;
        if !path.exists() {
            return Ok(None);
        }
        read_owner_json(&path, "request ledger").map(Some)
    }

    pub fn save_request(&self, request_id: &str, record: &RequestRecord) -> SurfaceResult<()> {
        let path = self.request_path(request_id)?;
        if path.exists() {
            return Err(SurfaceError::conflict("request_id already exists"));
        }
        atomic_owner_json_new(&path, record)
    }
}

mod locking;

use locking::*;
