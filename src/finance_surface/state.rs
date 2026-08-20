use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::policy::validate_id;
use super::{SurfaceError, SurfaceResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionStatus {
    Proposed,
    PolicyAccepted,
    Simulated,
    ApprovalPending,
    Approved,
    Timelocked,
    Ready,
    Signed,
    Submitted,
    Confirmed,
    Rejected,
    Cancelled,
    Expired,
    Failed,
    Indeterminate,
    Quarantined,
}

impl TransactionStatus {
    pub fn reserves_funds(self) -> bool {
        matches!(
            self,
            Self::Proposed
                | Self::PolicyAccepted
                | Self::Simulated
                | Self::ApprovalPending
                | Self::Approved
                | Self::Timelocked
                | Self::Ready
                | Self::Signed
                | Self::Submitted
                | Self::Indeterminate
                | Self::Quarantined
        )
    }
    pub fn terminal(self) -> bool {
        matches!(
            self,
            Self::Confirmed | Self::Rejected | Self::Cancelled | Self::Expired | Self::Failed
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalIntent {
    pub beneficiary_id: String,
    pub asset: String,
    pub amount_minor: i64,
    pub purpose: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransition {
    pub status: TransactionStatus,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub evidence_hash: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transaction {
    pub transaction_id: String,
    pub request_id: String,
    pub policy_id: String,
    pub policy_version: u64,
    pub lease_id: String,
    pub intent: CanonicalIntent,
    pub intent_hash: String,
    pub created_at: DateTime<Utc>,
    pub status: TransactionStatus,
    pub transitions: Vec<StateTransition>,
    pub approvals: BTreeMap<String, String>,
    pub simulation_evidence_hash: Option<String>,
    pub approval_deadline: DateTime<Utc>,
    pub timelock_until: DateTime<Utc>,
    pub reconciliation_required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestRecord {
    pub operation: String,
    pub input_hash: String,
    pub transaction_id: String,
    pub response: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditRecord {
    sequence: u64,
    timestamp: DateTime<Utc>,
    event: Value,
    previous_hash: String,
    hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommitRecord {
    commit_id: String,
    transaction: Transaction,
    request: Option<(String, RequestRecord)>,
    audit_event: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyAnchor {
    policy_id: String,
    version: u64,
    document_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LeaseAnchor {
    lease_id: String,
    issued_at: DateTime<Utc>,
    document_hash: String,
}

pub struct StoreLock {
    file: fs::File,
}
#[cfg(unix)]
impl Drop for StoreLock {
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

    fn tx_path(&self, id: &str) -> SurfaceResult<PathBuf> {
        validate_id("transaction_id", id)?;
        Ok(self.root.join("transactions").join(format!("{id}.json")))
    }
    fn request_path(&self, id: &str) -> SurfaceResult<PathBuf> {
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
                super::policy::require_owner_only_file(&path)?;
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

    fn recover_commits(&self) -> SurfaceResult<()> {
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

    fn apply_commit(&self, record: &CommitRecord) -> SurfaceResult<()> {
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
    fn repair_audit_head(&self) -> SurfaceResult<()> {
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

    fn validate_audit(&self) -> SurfaceResult<()> {
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

    fn append_audit_once(&self, commit_id: &str, event: Value) -> SurfaceResult<String> {
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
        let bytes = super::policy::canonical_json(event)?;
        let hash = hex::encode(Sha256::digest(&bytes));
        let path = sink.join(format!("{hash}.json"));
        if path.exists() {
            super::policy::require_owner_only_file(&path)?;
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

fn canonical_bytes<T: Serialize>(value: &T) -> SurfaceResult<Vec<u8>> {
    let value = serde_json::to_value(value)
        .map_err(|e| SurfaceError::internal(format!("cannot canonicalize journal: {e}")))?;
    super::policy::canonical_json(&value)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, kind: &str) -> SurfaceResult<T> {
    super::policy::require_owner_only_file(path)?;
    let bytes =
        fs::read(path).map_err(|e| SurfaceError::state(format!("cannot read {kind}: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| SurfaceError::state(format!("invalid {kind}: {e}")))
}

#[cfg(unix)]
fn ensure_owner_dir(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    if !path.exists() {
        let mut b = fs::DirBuilder::new();
        b.recursive(true).mode(0o700);
        b.create(path)
            .map_err(|e| SurfaceError::state(format!("cannot create state directory: {e}")))?;
    }
    let m = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::state(format!("cannot stat state directory: {e}")))?;
    if m.file_type().is_symlink()
        || !m.is_dir()
        || m.uid() != unsafe { geteuid() }
        || m.mode() & 0o077 != 0
    {
        return Err(SurfaceError::policy(
            "state and WORM directories must be owner-only, current-user-owned, and not symlinks",
        ));
    }
    Ok(())
}
#[cfg(not(unix))]
fn ensure_owner_dir(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}
#[cfg(unix)]
fn require_owner_dir(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::MetadataExt;
    let m = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::state(format!("external WORM sink is not provisioned: {e}")))?;
    if m.file_type().is_symlink()
        || !m.is_dir()
        || m.uid() != unsafe { geteuid() }
        || m.mode() & 0o077 != 0
    {
        return Err(SurfaceError::policy(
            "external WORM sink must be owner-only, current-user-owned, and not a symlink",
        ));
    }
    Ok(())
}
#[cfg(not(unix))]
fn require_owner_dir(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}

#[cfg(unix)]
fn create_new_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    sync_parent(path)
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("path has no parent"))?;
    fs::File::open(parent)?.sync_all()
}

fn audit_hash(
    sequence: u64,
    timestamp: DateTime<Utc>,
    event: &Value,
    previous_hash: &str,
) -> SurfaceResult<String> {
    let value = serde_json::json!({"event":event,"previous_hash":previous_hash,"sequence":sequence,"timestamp":timestamp});
    Ok(hex::encode(Sha256::digest(super::policy::canonical_json(
        &value,
    )?)))
}

#[cfg(unix)]
fn atomic_json<T: Serialize>(path: &Path, value: &T, new_only: bool) -> SurfaceResult<()> {
    let bytes = serde_json::to_vec(value)
        .map_err(|e| SurfaceError::internal(format!("cannot serialize state: {e}")))?;
    if new_only {
        return create_new_bytes(path, &bytes).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                SurfaceError::conflict("record already exists")
            } else {
                SurfaceError::state(format!("cannot persist state: {e}"))
            }
        });
    }
    let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    create_new_bytes(&tmp, &bytes)
        .map_err(|e| SurfaceError::state(format!("cannot persist temporary state: {e}")))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        SurfaceError::state(format!("cannot install state: {e}"))
    })?;
    sync_parent(path).map_err(|e| SurfaceError::state(format!("cannot sync state directory: {e}")))
}
#[cfg(not(unix))]
fn atomic_json<T: Serialize>(_path: &Path, _value: &T, _new_only: bool) -> SurfaceResult<()> {
    Err(SurfaceError::policy("owner-only state requires Unix"))
}

#[cfg(unix)]
const LOCK_EX: i32 = 2;
#[cfg(unix)]
const LOCK_UN: i32 = 8;
#[cfg(unix)]
unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
    fn geteuid() -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "wisent-finance-state-test-{}",
                uuid::Uuid::new_v4()
            ));
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn transaction(id: &str) -> Transaction {
        let now = Utc::now();
        Transaction {
            transaction_id: id.into(),
            request_id: format!("request-{id}"),
            policy_id: "policy".into(),
            policy_version: 2,
            lease_id: "lease".into(),
            intent: CanonicalIntent {
                beneficiary_id: "beneficiary".into(),
                asset: "USD".into(),
                amount_minor: 10,
                purpose: "invoice".into(),
                expires_at: now + chrono::Duration::minutes(10),
            },
            intent_hash: "a".repeat(64),
            created_at: now,
            status: TransactionStatus::ApprovalPending,
            transitions: vec![],
            approvals: BTreeMap::new(),
            simulation_evidence_hash: None,
            approval_deadline: now + chrono::Duration::minutes(10),
            timelock_until: now + chrono::Duration::minutes(1),
            reconciliation_required: false,
        }
    }

    #[test]
    fn finance_contract_rejects_policy_and_lease_rollback_or_equivocation() {
        let directory = TestDirectory::new();
        let store = StateStore::open(directory.0.clone()).unwrap();
        let now = Utc::now();
        store.bind_policy("policy", 2, "hash-v2").unwrap();
        store.bind_lease("lease-2", now, "lease-hash-2").unwrap();

        let policy_rollback = store.bind_policy("policy", 1, "hash-v1").unwrap_err();
        let policy_equivocation = store.bind_policy("policy", 2, "different").unwrap_err();
        let lease_rollback = store
            .bind_lease("lease-1", now - chrono::Duration::seconds(1), "old")
            .unwrap_err();
        let lease_equivocation = store
            .bind_lease("other-lease", now, "different")
            .unwrap_err();

        assert_eq!(
            policy_rollback.to_string(),
            "policy_denied: signed policy rollback or equivocation detected"
        );
        assert_eq!(policy_equivocation.to_string(), policy_rollback.to_string());
        assert_eq!(
            lease_rollback.to_string(),
            "policy_denied: signed enable lease rollback or equivocation detected"
        );
        assert_eq!(lease_equivocation.to_string(), lease_rollback.to_string());
    }

    #[test]
    fn finance_contract_reopen_rejects_a_corrupted_audit_record() {
        let directory = TestDirectory::new();
        let store = StateStore::open(directory.0.clone()).unwrap();
        store
            .append_audit(serde_json::json!({"type":"proposal"}))
            .unwrap();
        let audit_path = fs::read_dir(directory.0.join("audit"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        fs::write(audit_path, b"{}").unwrap();

        let error = match StateStore::open(directory.0.clone()) {
            Ok(_) => panic!("corrupted audit record was accepted"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .starts_with("state_error: invalid audit record:")
        );
    }

    #[test]
    fn finance_contract_reopen_recovers_an_unapplied_commit_journal() {
        let directory = TestDirectory::new();
        let store = StateStore::open(directory.0.clone()).unwrap();
        let transaction = transaction("crash-recovery");
        let response = serde_json::json!({"transaction_id":"crash-recovery"});
        let request = RequestRecord {
            operation: "finance_propose".into(),
            input_hash: "input-hash".into(),
            transaction_id: transaction.transaction_id.clone(),
            response: response.clone(),
        };
        let commit = CommitRecord {
            commit_id: "crash-commit".into(),
            transaction: transaction.clone(),
            request: Some(("crash-request".into(), request)),
            audit_event: serde_json::json!({
                "type":"proposal_created",
                "commit_id":"crash-commit"
            }),
        };
        atomic_json(
            &directory.0.join("commits/crash-commit.json"),
            &commit,
            true,
        )
        .unwrap();
        drop(store);

        let reopened = StateStore::open(directory.0.clone()).unwrap();

        assert_eq!(
            reopened
                .load_transaction("crash-recovery")
                .unwrap()
                .transaction_id,
            transaction.transaction_id
        );
        assert_eq!(
            reopened
                .load_request("crash-request")
                .unwrap()
                .unwrap()
                .response,
            response
        );
        assert!(
            directory
                .0
                .join("commit-applied/crash-commit.json")
                .exists()
        );
    }
}
