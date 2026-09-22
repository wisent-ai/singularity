use chrono::{DateTime, Utc};
use std::fs;
use std::path::PathBuf;

use crate::finance_surface::{SurfaceError, SurfaceResult};

#[derive(Clone)]
pub struct StateStore {
    root: PathBuf,
}

impl StateStore {
    pub fn open(root: PathBuf) -> SurfaceResult<Self> {
        if !root.is_absolute() {
            return Err(SurfaceError::policy(
                "SINGULARITY_FINANCE_STATE_DIR must be absolute",
            ));
        }
        ensure_owner_dir(&root)?;
        for child in [
            "transactions",
            "requests",
            "audit",
            "commits",
            "commit-applied",
        ] {
            ensure_owner_dir(&root.join(child))?;
        }
        let store = Self { root };
        {
            let _lock = store.lock()?;
            store.repair_audit_head()?;
            store.validate_audit()?;
            store.recover_commits()?;
        }
        Ok(store)
    }

    #[cfg(unix)]
    pub fn lock(&self) -> SurfaceResult<StoreLock> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(self.root.join("finance.lock"))
            .map_err(|e| SurfaceError::state(format!("cannot open finance lock: {e}")))?;
        if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
            return Err(SurfaceError::state(format!(
                "cannot acquire finance lock: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(StoreLock { file })
    }
    #[cfg(not(unix))]
    pub fn lock(&self) -> SurfaceResult<StoreLock> {
        Err(SurfaceError::policy("finance locking requires Unix"))
    }

    pub fn bind_policy(
        &self,
        policy_id: &str,
        version: u64,
        document_hash: &str,
    ) -> SurfaceResult<()> {
        let path = self.root.join("policy-anchor.json");
        let next = PolicyAnchor {
            policy_id: policy_id.into(),
            version,
            document_hash: document_hash.into(),
        };
        if path.exists() {
            let current: PolicyAnchor = read_json(&path, "policy anchor")?;
            if current.policy_id != next.policy_id
                || version < current.version
                || (version == current.version && current.document_hash != next.document_hash)
            {
                return Err(SurfaceError::policy(
                    "signed policy rollback or equivocation detected",
                ));
            }
            if version == current.version {
                return Ok(());
            }
        }
        atomic_json(&path, &next, false)
    }

    pub fn bind_lease(
        &self,
        lease_id: &str,
        issued_at: DateTime<Utc>,
        document_hash: &str,
    ) -> SurfaceResult<()> {
        let path = self.root.join("lease-anchor.json");
        let next = LeaseAnchor {
            lease_id: lease_id.into(),
            issued_at,
            document_hash: document_hash.into(),
        };
        if path.exists() {
            let current: LeaseAnchor = read_json(&path, "lease anchor")?;
            if issued_at < current.issued_at
                || (issued_at == current.issued_at
                    && (current.lease_id != next.lease_id
                        || current.document_hash != next.document_hash))
            {
                return Err(SurfaceError::policy(
                    "signed enable lease rollback or equivocation detected",
                ));
            }
            if issued_at == current.issued_at {
                return Ok(());
            }
        }
        atomic_json(&path, &next, false)
    }
}

mod audit;
mod entries;
mod files;
mod records;

pub(crate) use files::*;
use files::{LOCK_EX, LOCK_UN, flock};
pub use records::*;
use records::{LeaseAnchor, PolicyAnchor};

#[cfg(test)]
#[path = "../../../tests/finance_surface/state.rs"]
mod tests;
