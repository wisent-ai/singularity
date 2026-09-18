//! Reading and writing one transaction, one request and one commit.
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use super::records::CommitRecord;
use super::*;
use crate::finance_surface::policy::validate_id;
use crate::finance_surface::{SurfaceError, SurfaceResult};

impl StateStore {
    pub(super) fn tx_path(&self, id: &str) -> SurfaceResult<PathBuf> {
        validate_id("transaction_id", id)?;
        Ok(self.root.join("transactions").join(format!("{id}.json")))
    }
    pub(super) fn request_path(&self, id: &str) -> SurfaceResult<PathBuf> {
        validate_id("request_id", id)?;
        Ok(self.root.join("requests").join(format!("{id}.json")))
    }
    pub fn load_transaction(&self, id: &str) -> SurfaceResult<Transaction> {
        read_json(&self.tx_path(id)?, "transaction")
    }
    pub fn save_transaction(&self, tx: &Transaction) -> SurfaceResult<()> {
        atomic_json(&self.tx_path(&tx.transaction_id)?, tx, false)
    }
    pub fn transaction_exists(&self, id: &str) -> SurfaceResult<bool> {
        let path = self.tx_path(id)?;
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                crate::finance_surface::policy::require_owner_only_file(&path)?;
                Ok(true)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(SurfaceError::state(format!(
                "cannot inspect transaction: {error}"
            ))),
        }
    }
    pub fn load_request(&self, id: &str) -> SurfaceResult<Option<RequestRecord>> {
        let path = self.request_path(id)?;
        if !path.exists() {
            return Ok(None);
        }
        read_json(&path, "request record").map(Some)
    }
    pub fn save_request(&self, id: &str, record: &RequestRecord) -> SurfaceResult<()> {
        atomic_json(&self.request_path(id)?, record, true)
    }
    pub fn commit(
        &self,
        commit_id: &str,
        transaction: &Transaction,
        request: Option<(&str, &RequestRecord)>,
        mut audit_event: Value,
    ) -> SurfaceResult<()> {
        validate_id("commit_id", commit_id)?;
        if let Value::Object(map) = &mut audit_event {
            map.insert("commit_id".into(), Value::String(commit_id.into()));
        } else {
            return Err(SurfaceError::internal("audit event must be an object"));
        }
        let record = CommitRecord {
            commit_id: commit_id.into(),
            transaction: transaction.clone(),
            request: request.map(|(id, value)| (id.into(), value.clone())),
            audit_event,
        };
        let path = self.root.join("commits").join(format!("{commit_id}.json"));
        if path.exists() {
            let existing: CommitRecord = read_json(&path, "commit journal")?;
            if canonical_bytes(&existing)? != canonical_bytes(&record)? {
                return Err(SurfaceError::conflict(
                    "commit id reused with different content",
                ));
            }
        } else {
            atomic_json(&path, &record, true)?;
        }
        self.apply_commit(&record)
    }

    pub(super) fn recover_commits(&self) -> SurfaceResult<()> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(self.root.join("commits"))
            .map_err(|e| SurfaceError::state(format!("cannot list commit journal: {e}")))?
        {
            let path = entry
                .map_err(|e| SurfaceError::state(format!("cannot inspect commit journal: {e}")))?
                .path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                return Err(SurfaceError::state("unknown file in commit journal"));
            }
            paths.push(path);
        }
        paths.sort();
        for path in paths {
            let record: CommitRecord = read_json(&path, "commit journal")?;
            self.apply_commit(&record)?;
        }
        Ok(())
    }

    pub(super) fn apply_commit(&self, record: &CommitRecord) -> SurfaceResult<()> {
        let marker = self
            .root
            .join("commit-applied")
            .join(format!("{}.json", record.commit_id));
        if marker.exists() {
            let applied: String = read_json(&marker, "commit marker")?;
            if applied != record.commit_id {
                return Err(SurfaceError::state("invalid commit marker"));
            }
            return Ok(());
        }
        self.append_audit_once(&record.commit_id, record.audit_event.clone())?;
        self.save_transaction(&record.transaction)?;
        if let Some((id, request)) = &record.request {
            match self.load_request(id)? {
                Some(existing) if canonical_bytes(&existing)? == canonical_bytes(request)? => {}
                Some(_) => {
                    return Err(SurfaceError::state(
                        "request ledger conflicts with commit journal",
                    ));
                }
                None => self.save_request(id, request)?,
            }
        }
        atomic_json(&marker, &record.commit_id, true)
    }

    pub fn all_transactions(&self) -> SurfaceResult<Vec<Transaction>> {
        let mut values = Vec::new();
        for entry in fs::read_dir(self.root.join("transactions"))
            .map_err(|e| SurfaceError::state(format!("cannot list transactions: {e}")))?
        {
            let entry = entry
                .map_err(|e| SurfaceError::state(format!("cannot inspect transaction: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                return Err(SurfaceError::state("unknown file in transaction store"));
            }
            values.push(read_json(&path, "transaction")?);
        }
        Ok(values)
    }
}
