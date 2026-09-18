//! Proposing a transaction, reading its status, and cancelling it.
use chrono::{Duration, Utc};
use serde_json::{Value, json};

use super::{
    FinanceService, MAX_PURPOSE_CHARS, hash_value, public_status, transition,
    validate_execution_parameters,
};
use crate::finance_surface::policy::{validate_asset, validate_id};
use crate::finance_surface::state::{
    CanonicalIntent, RequestRecord, StateTransition, Transaction, TransactionStatus,
};
use crate::finance_surface::{SurfaceError, SurfaceResult};

use super::{Cancel, Propose, Status};
impl FinanceService {
    pub(super) fn propose(&self, input: Propose) -> SurfaceResult<Value> {
        validate_id("request_id", &input.request_id)?;
        validate_id("beneficiary_id", &input.beneficiary_id)?;
        validate_asset(&input.asset)?;
        if input.amount_minor <= 0 {
            return Err(SurfaceError::invalid(
                "amount_minor must be a positive integer",
            ));
        }
        if input.purpose.is_empty()
            || input.purpose.len() > MAX_PURPOSE_CHARS
            || input.purpose.chars().any(char::is_control)
        {
            return Err(SurfaceError::invalid(
                "purpose must be 1..=512 printable characters",
            ));
        }
        if input.ttl_seconds == 0
            || input.ttl_seconds > self.policy.approval.proposal_ttl_max_seconds
        {
            return Err(SurfaceError::policy("proposal TTL exceeds signed policy"));
        }
        let _lock = self.state.lock()?;
        let lease = self.active_lease()?;
        let now = Utc::now();
        let beneficiary = self
            .policy
            .beneficiaries
            .get(&input.beneficiary_id)
            .ok_or_else(|| SurfaceError::policy("beneficiary is not in signed policy"))?;
        if !beneficiary.enabled
            || now < beneficiary.valid_from
            || now >= beneficiary.expires_at
            || !beneficiary.allowed_assets.iter().any(|v| v == &input.asset)
            || !beneficiary
                .allowed_purposes
                .iter()
                .any(|v| v == &input.purpose)
        {
            return Err(SurfaceError::policy(
                "beneficiary is disabled, outside its validity window, or disallows the asset or purpose",
            ));
        }
        let limits = self
            .policy
            .assets
            .get(&input.asset)
            .ok_or_else(|| SurfaceError::policy("asset is not in signed policy"))?;
        if input.amount_minor > limits.per_transaction_minor
            || input.amount_minor > beneficiary.per_transaction_minor
        {
            return Err(SurfaceError::policy("per-transaction limit exceeded"));
        }
        validate_execution_parameters(&input.parameters)?;
        let input_hash = hash_value(
            &json!({"beneficiary_id":input.beneficiary_id,"asset":input.asset,"amount_minor":input.amount_minor,"purpose":input.purpose,"parameters":input.parameters,"ttl_seconds":input.ttl_seconds}),
        )?;
        if let Some(record) = self.state.load_request(&input.request_id)? {
            if record.operation != "finance_propose" || record.input_hash != input_hash {
                return Err(SurfaceError::conflict(
                    "request_id was already used with different intent",
                ));
            }
            return Ok(record.response);
        }
        let expires_at = now
            .checked_add_signed(Duration::seconds(
                i64::try_from(input.ttl_seconds)
                    .map_err(|_| SurfaceError::invalid("TTL overflow"))?,
            ))
            .ok_or_else(|| SurfaceError::invalid("TTL overflow"))?;
        if expires_at > beneficiary.expires_at {
            return Err(SurfaceError::policy(
                "proposal validity exceeds beneficiary validity",
            ));
        }
        let intent = CanonicalIntent {
            beneficiary_id: input.beneficiary_id,
            asset: input.asset,
            amount_minor: input.amount_minor,
            purpose: input.purpose,
            parameters: input.parameters,
            expires_at,
        };
        let intent_hash = hash_value(&intent)?;
        self.enforce_limits(&intent, now)?;
        let transaction_id = format!("fin_{}", &intent_hash[..32]);
        if self.state.transaction_exists(&transaction_id)? {
            return Err(SurfaceError::conflict(
                "intent already has a different idempotency owner",
            ));
        }
        let timelock_until = expires_at;
        let mut tx = Transaction {
            transaction_id: transaction_id.clone(),
            request_id: input.request_id.clone(),
            policy_id: self.policy.policy_id.clone(),
            policy_version: self.policy.version,
            lease_id: lease.lease_id,
            intent,
            intent_hash,
            created_at: now,
            status: TransactionStatus::ApprovalPending,
            transitions: Vec::new(),
            approvals: std::collections::BTreeMap::new(),
            simulation_evidence_hash: None,
            approval_deadline: expires_at,
            timelock_until,
            reconciliation_required: false,
        };
        for status in [
            TransactionStatus::Proposed,
            TransactionStatus::PolicyAccepted,
            TransactionStatus::ApprovalPending,
        ] {
            tx.transitions.push(StateTransition {
                status,
                at: now,
                actor: "finance_monitor".into(),
                evidence_hash: None,
            });
        }
        let response = public_status(&tx);
        let request = RequestRecord {
            operation: "finance_propose".into(),
            input_hash,
            transaction_id,
            response: response.clone(),
        };
        let commit_id = format!("proposal_{}", &hash_value(&input.request_id)?[..40]);
        self.state.commit(&commit_id,&tx,Some((&input.request_id,&request)),json!({"type":"proposal_created","transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash,"policy_id":tx.policy_id,"policy_version":tx.policy_version}))?;
        Ok(response)
    }

    pub(super) fn status(&self, input: Status) -> SurfaceResult<Value> {
        validate_id("transaction_id", &input.transaction_id)?;
        let _lock = self.state.lock()?;
        let mut tx = self.state.load_transaction(&input.transaction_id)?;
        self.refresh_time(&mut tx)?;
        Ok(public_status(&tx))
    }
    pub(super) fn cancel(&self, input: Cancel) -> SurfaceResult<Value> {
        validate_id("transaction_id", &input.transaction_id)?;
        validate_id("request_id", &input.request_id)?;
        let _lock = self.state.lock()?;
        let input_hash = hash_value(&json!({"transaction_id":input.transaction_id}))?;
        if let Some(r) = self.state.load_request(&input.request_id)? {
            if r.operation != "finance_cancel" || r.input_hash != input_hash {
                return Err(SurfaceError::conflict(
                    "request_id was already used differently",
                ));
            }
            return Ok(r.response);
        }
        let mut tx = self.state.load_transaction(&input.transaction_id)?;
        self.refresh_time(&mut tx)?;
        if Utc::now() >= tx.approval_deadline {
            return Err(SurfaceError::conflict("cancellation deadline has passed"));
        }
        if matches!(
            tx.status,
            TransactionStatus::Signed
                | TransactionStatus::Submitted
                | TransactionStatus::Confirmed
                | TransactionStatus::Indeterminate
                | TransactionStatus::Quarantined
        ) || tx.status.terminal()
        {
            return Err(SurfaceError::conflict(
                "transaction can no longer be cancelled",
            ));
        }
        transition(&mut tx, TransactionStatus::Cancelled, "model", None);
        let response = public_status(&tx);
        let request = RequestRecord {
            operation: "finance_cancel".into(),
            input_hash,
            transaction_id: tx.transaction_id.clone(),
            response: response.clone(),
        };
        let commit_id = format!("cancel_{}", &hash_value(&input.request_id)?[..40]);
        self.state.commit(&commit_id,&tx,Some((&input.request_id,&request)),json!({"type":"proposal_cancelled","transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash}))?;
        Ok(response)
    }
}
