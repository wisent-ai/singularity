//! The records the store holds and the states a transaction moves through.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;

use super::*;

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
    pub parameters: Option<Value>,
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
pub(super) struct AuditRecord {
    pub(super) sequence: u64,
    pub(super) timestamp: DateTime<Utc>,
    pub(super) event: Value,
    pub(super) previous_hash: String,
    pub(super) hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CommitRecord {
    pub(super) commit_id: String,
    pub(super) transaction: Transaction,
    pub(super) request: Option<(String, RequestRecord)>,
    pub(super) audit_event: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PolicyAnchor {
    pub(super) policy_id: String,
    pub(super) version: u64,
    pub(super) document_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LeaseAnchor {
    pub(super) lease_id: String,
    pub(super) issued_at: DateTime<Utc>,
    pub(super) document_hash: String,
}

pub struct StoreLock {
    pub(super) file: fs::File,
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
