use chrono::Utc;
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

use crate::finance_surface::policy::{EnableLease, PolicyFile};
use crate::finance_surface::state::StateStore;
use crate::finance_surface::{SurfaceError, SurfaceResult};

/// A purpose is at most 512 printable characters; an executor answer at most 64 KiB;
/// an owner event counts from a day back to five minutes ahead; parameters are at most
/// 16 KiB and eight levels deep; an evidence hash is 64 hex characters.
const MAX_PURPOSE_CHARS: usize = 512;
const MAX_EXECUTOR_RESPONSE_BYTES: usize = 64 * 1024;
const OWNER_EVENT_MAX_AGE_HOURS: i64 = 24;
const OWNER_EVENT_MAX_SKEW_MINUTES: i64 = 5;
const MAX_PARAMETER_BYTES: usize = 16 * 1024;
const MAX_PARAMETER_DEPTH: usize = 8;
const HASH_HEX_CHARS: usize = 64;

#[derive(Clone)]
pub struct FinanceService {
    policy: PolicyFile,
    state: StateStore,
    lease_path: PathBuf,
    executor: PathBuf,
    verifying_key: VerifyingKey,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Propose {
    request_id: String,
    beneficiary_id: String,
    asset: String,
    amount_minor: i64,
    purpose: String,
    #[serde(default)]
    parameters: Option<Value>,
    ttl_seconds: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Status {
    transaction_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Cancel {
    transaction_id: String,
    request_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Execute {
    transaction_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionResponse {
    executor_id: String,
    executor_reference_hash: String,
    executor_signature_hex: String,
    worm_receipt_file: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerEvent {
    pub event_id: String,
    pub transaction_id: String,
    pub intent_hash: String,
    pub occurred_at: chrono::DateTime<Utc>,
    pub action: OwnerAction,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum OwnerAction {
    SimulationAccepted {
        evidence_hash: String,
        simulator_id: String,
        simulator_signature_hex: String,
    },
    ApprovalGranted {
        approver_id: String,
        approval_signature_hex: String,
    },
    Signed {
        signer_attestation_hash: String,
        signer_id: String,
        signer_signature_hex: String,
    },
    Submitted {
        executor_reference_hash: String,
        executor_id: String,
        executor_signature_hex: String,
        worm_receipt_file: PathBuf,
    },
    Confirmed {
        reconciliation_hash: String,
        reconciler_id: String,
        reconciler_signature_hex: String,
        worm_receipt_file: PathBuf,
    },
    Rejected {
        reason_code: String,
    },
    Failed {
        reason_code: String,
    },
    Indeterminate {
        reason_code: String,
        executor_id: String,
        executor_signature_hex: String,
    },
    ReconciledNotSubmitted {
        reconciliation_hash: String,
        reconciler_id: String,
        reconciler_signature_hex: String,
    },
    Quarantined {
        reason_code: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WormReceipt {
    sink_id: String,
    receipt_id: String,
    event_kind: String,
    transaction_id: String,
    intent_hash: String,
    reference_hash: String,
    recorded_at: chrono::DateTime<Utc>,
}

impl FinanceService {
    pub fn new(
        policy: PolicyFile,
        state: StateStore,
        lease_path: PathBuf,
        verifying_key: VerifyingKey,
        executor: PathBuf,
    ) -> Self {
        Self {
            policy,
            state,
            lease_path,
            verifying_key,
            executor,
        }
    }
    pub async fn call(&self, name: &str, arguments: Value) -> SurfaceResult<Value> {
        match name {
            "finance_propose" => self.propose(parse(arguments)?),
            "finance_status" => self.status(parse(arguments)?),
            "finance_cancel" => self.cancel(parse(arguments)?),
            "finance_execute" => self.execute(parse(arguments)?).await,
            _ => Err(SurfaceError::invalid("unknown tool")),
        }
    }
    fn active_lease(&self) -> SurfaceResult<EnableLease> {
        let now = Utc::now();
        self.policy.require_active(now)?;
        let lease =
            EnableLease::load_active(&self.lease_path, &self.verifying_key, &self.policy, now)?;
        self.state.bind_lease(
            &lease.lease_id,
            lease.issued_at,
            &crate::finance_surface::policy::document_hash(&lease)?,
        )?;
        Ok(lease)
    }
}

mod events;
mod execution;
mod proposals;
mod verify;

pub(crate) use verify::*;

#[cfg(test)]
#[path = "../../../tests/finance_surface/service.rs"]
mod tests;
