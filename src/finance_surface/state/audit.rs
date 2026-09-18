//! The audit chain: its head, its validation, and the WORM sink beside it.
use chrono::Utc;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

use super::records::{AuditRecord, CommitRecord};
use super::*;
use crate::finance_surface::{SurfaceError, SurfaceResult};

impl StateStore {
    pub(super) fn repair_audit_head(&self) -> SurfaceResult<()> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(self.root.join("audit"))
            .map_err(|e| SurfaceError::state(format!("cannot list audit chain: {e}")))?
        {
            paths.push(
                entry
                    .map_err(|e| SurfaceError::state(format!("cannot inspect audit chain: {e}")))?
                    .path(),
            );
        }
        paths.sort();
        let Some(path) = paths.last() else {
            return Ok(());
        };
        let last: AuditRecord = read_json(path, "audit record")?;
        let head_path = self.root.join("audit-head.json");
        let head_sequence = if head_path.exists() {
            let head: AuditRecord = read_json(&head_path, "audit head")?;
            if head.sequence == last.sequence && head.hash == last.hash {
                return Ok(());
            }
            if head.sequence >= last.sequence || head.sequence.checked_add(1) != Some(last.sequence)
            {
                return Err(SurfaceError::state("audit head cannot be safely recovered"));
            }
            head.sequence
        } else {
            if last.sequence != 1 {
                return Err(SurfaceError::state(
                    "missing audit head cannot be safely recovered",
                ));
            }
            0
        };
        let _ = head_sequence;
        let commit_id = last
            .event
            .get("commit_id")
            .and_then(Value::as_str)
            .ok_or_else(|| SurfaceError::state("orphan audit record has no journal commit"))?;
        let journal_path = self.root.join("commits").join(format!("{commit_id}.json"));
        let journal: CommitRecord = read_json(&journal_path, "commit journal")?;
        if journal.audit_event != last.event
            || audit_hash(
                last.sequence,
                last.timestamp,
                &last.event,
                &last.previous_hash,
            )? != last.hash
        {
            return Err(SurfaceError::state(
                "orphan audit record does not match commit journal",
            ));
        }
        atomic_json(&head_path, &last, false)
    }

    pub(super) fn validate_audit(&self) -> SurfaceResult<()> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(self.root.join("audit"))
            .map_err(|e| SurfaceError::state(format!("cannot list audit chain: {e}")))?
        {
            let path = entry
                .map_err(|e| SurfaceError::state(format!("cannot inspect audit chain: {e}")))?
                .path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                return Err(SurfaceError::state("unknown file in audit chain"));
            }
            paths.push(path);
        }
        paths.sort();
        let mut previous = "0".repeat(64);
        let mut last: Option<AuditRecord> = None;
        for (index, path) in paths.iter().enumerate() {
            let record: AuditRecord = read_json(path, "audit record")?;
            let expected_sequence = u64::try_from(index + 1)
                .map_err(|_| SurfaceError::state("audit sequence overflow"))?;
            let expected_name = format!("{:020}-{}.json", record.sequence, record.hash);
            if record.sequence != expected_sequence
                || record.previous_hash != previous
                || path.file_name().and_then(|v| v.to_str()) != Some(&expected_name)
                || audit_hash(
                    record.sequence,
                    record.timestamp,
                    &record.event,
                    &record.previous_hash,
                )? != record.hash
            {
                return Err(SurfaceError::state("audit hash chain validation failed"));
            }
            previous = record.hash.clone();
            last = Some(record);
        }
        let head_path = self.root.join("audit-head.json");
        match (last, head_path.exists()) {
            (None, false) => Ok(()),
            (Some(last), true) => {
                let head: AuditRecord = read_json(&head_path, "audit head")?;
                if head.sequence != last.sequence
                    || head.hash != last.hash
                    || head.previous_hash != last.previous_hash
                {
                    return Err(SurfaceError::state(
                        "audit head does not match complete chain",
                    ));
                }
                Ok(())
            }
            _ => Err(SurfaceError::state("audit head/chain presence mismatch")),
        }
    }

    pub(super) fn append_audit_once(&self, commit_id: &str, event: Value) -> SurfaceResult<String> {
        for entry in fs::read_dir(self.root.join("audit"))
            .map_err(|e| SurfaceError::state(format!("cannot list audit chain: {e}")))?
        {
            let path = entry
                .map_err(|e| SurfaceError::state(format!("cannot inspect audit chain: {e}")))?
                .path();
            let record: AuditRecord = read_json(&path, "audit record")?;
            if record.event.get("commit_id").and_then(Value::as_str) == Some(commit_id) {
                if record.event != event {
                    return Err(SurfaceError::state(
                        "audit commit id conflicts with journal",
                    ));
                }
                return Ok(record.hash);
            }
        }
        self.append_audit(event)
    }

    pub fn append_audit(&self, event: Value) -> SurfaceResult<String> {
        self.validate_audit()?;
        let head_path = self.root.join("audit-head.json");
        let (sequence, previous_hash) = if head_path.exists() {
            let head: AuditRecord = read_json(&head_path, "audit head")?;
            let record_path = self
                .root
                .join("audit")
                .join(format!("{:020}-{}.json", head.sequence, head.hash));
            let persisted: AuditRecord = read_json(&record_path, "audit record")?;
            if persisted.hash != head.hash
                || audit_hash(
                    persisted.sequence,
                    persisted.timestamp,
                    &persisted.event,
                    &persisted.previous_hash,
                )? != persisted.hash
            {
                return Err(SurfaceError::state("audit chain head mismatch"));
            }
            (
                head.sequence
                    .checked_add(1)
                    .ok_or_else(|| SurfaceError::state("audit sequence overflow"))?,
                head.hash,
            )
        } else {
            (1, "0".repeat(64))
        };
        let timestamp = Utc::now();
        let hash = audit_hash(sequence, timestamp, &event, &previous_hash)?;
        let record = AuditRecord {
            sequence,
            timestamp,
            event,
            previous_hash,
            hash: hash.clone(),
        };
        let record_path = self
            .root
            .join("audit")
            .join(format!("{sequence:020}-{hash}.json"));
        atomic_json(&record_path, &record, true)?;
        atomic_json(&head_path, &record, false)?;
        Ok(hash)
    }

    pub fn append_worm(&self, sink: &Path, event: &Value) -> SurfaceResult<()> {
        if !sink.is_absolute() {
            return Err(SurfaceError::policy("WORM sink must be absolute"));
        }
        require_owner_dir(sink)?;
        let bytes = crate::finance_surface::policy::canonical_json(event)?;
        let hash = hex::encode(Sha256::digest(&bytes));
        let path = sink.join(format!("{hash}.json"));
        if path.exists() {
            crate::finance_surface::policy::require_owner_only_file(&path)?;
            let existing = fs::read(&path)
                .map_err(|e| SurfaceError::state(format!("cannot read WORM receipt copy: {e}")))?;
            if existing == bytes {
                return Ok(());
            }
            return Err(SurfaceError::state("WORM receipt copy hash collision"));
        }
        create_new_bytes(&path, &bytes)
            .map_err(|e| SurfaceError::state(format!("external WORM sink unavailable: {e}")))
    }
}
