//! What a transaction and its evidence must satisfy: parameters, hashes, roles, approvals, receipts.
use chrono::Utc;
use ed25519_dalek::{Signature, Verifier};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::finance_surface::policy::{
    load_signed, validate_id, verifying_key_from_hex, PolicyFile,
};
use crate::finance_surface::state::{StateTransition, Transaction, TransactionStatus};
use crate::finance_surface::{SurfaceError, SurfaceResult};

use super::{WormReceipt, HASH_HEX_CHARS, MAX_PARAMETER_BYTES, MAX_PARAMETER_DEPTH};

/// A proposal that gave no parameters is a proposal with nothing to validate; a `null` in place
/// of an object is the same absence written out, and anything else is refused.
pub(crate) fn validate_execution_parameters(value: Option<&Value>) -> SurfaceResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_null() && !value.is_object() {
        return Err(SurfaceError::invalid("parameters must be a JSON object"));
    }
    if serde_json::to_vec(value)
        .map_err(|error| SurfaceError::invalid(format!("invalid parameters: {error}")))?
        .len()
        > MAX_PARAMETER_BYTES
    {
        return Err(SurfaceError::invalid("parameters exceed size limit"));
    }
    fn walk(value: &Value, depth: usize) -> SurfaceResult<()> {
        if depth > MAX_PARAMETER_DEPTH {
            return Err(SurfaceError::invalid("parameters exceed depth limit"));
        }
        match value {
            Value::Object(map) => {
                for (key, nested) in map {
                    let normalized = key.to_ascii_lowercase().replace('-', "_");
                    if [
                        "destination",
                        "recipient",
                        "beneficiary_id",
                        "asset",
                        "amount_minor",
                        "purpose",
                        "private_key",
                        "secret",
                        "token",
                        "authorization",
                    ]
                    .contains(&normalized.as_str())
                    {
                        return Err(SurfaceError::policy(
                            "parameters cannot override protected intent fields",
                        ));
                    }
                    walk(nested, depth + 1)?;
                }
            }
            Value::Array(values) => {
                for nested in values {
                    walk(nested, depth + 1)?;
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
        Ok(())
    }
    walk(value, 0)
}

pub(crate) fn parse<T: for<'de> Deserialize<'de>>(v: Value) -> SurfaceResult<T> {
    serde_json::from_value(v).map_err(|e| SurfaceError::invalid(format!("invalid arguments: {e}")))
}
pub(crate) fn hash_value<T: Serialize>(v: &T) -> SurfaceResult<String> {
    let value = serde_json::to_value(v)
        .map_err(|e| SurfaceError::internal(format!("cannot canonicalize value: {e}")))?;
    let b = crate::finance_surface::policy::canonical_json(&value)?;
    Ok(hex::encode(Sha256::digest(b)))
}
pub(crate) fn checked_add(a: i64, b: i64) -> SurfaceResult<i64> {
    a.checked_add(b)
        .ok_or_else(|| SurfaceError::policy("financial total overflow"))
}
pub(crate) fn require_hash(v: &str) -> SurfaceResult<()> {
    if v.len() != HASH_HEX_CHARS || !v.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(SurfaceError::invalid(
            "evidence hash must be 64 hexadecimal characters",
        ));
    }
    Ok(())
}
pub(crate) fn transition(
    tx: &mut Transaction,
    status: TransactionStatus,
    actor: &str,
    evidence_hash: Option<String>,
) {
    tx.status = status;
    tx.transitions.push(StateTransition {
        status,
        at: Utc::now(),
        actor: actor.into(),
        evidence_hash,
    });
}
pub(crate) fn execution_time(tx: &Transaction) -> Option<chrono::DateTime<Utc>> {
    tx.transitions
        .iter()
        .rev()
        .find(|v| {
            matches!(
                v.status,
                TransactionStatus::Confirmed | TransactionStatus::Submitted
            )
        })
        .map(|v| v.at)
}
pub(crate) fn pre_signing_state(status: TransactionStatus) -> bool {
    matches!(
        status,
        TransactionStatus::Proposed
            | TransactionStatus::PolicyAccepted
            | TransactionStatus::Simulated
            | TransactionStatus::ApprovalPending
            | TransactionStatus::Approved
            | TransactionStatus::Timelocked
            | TransactionStatus::Ready
    )
}
pub(crate) fn verify_worm_receipt(
    policy: &PolicyFile,
    tx: &Transaction,
    event_kind: &str,
    reference_hash: &str,
    occurred_at: &chrono::DateTime<Utc>,
    path: &Path,
) -> SurfaceResult<String> {
    if !path.is_absolute() {
        return Err(SurfaceError::policy("WORM receipt file must be absolute"));
    }
    let key = verifying_key_from_hex(&policy.worm_receipt_key_hex)?;
    let receipt: WormReceipt = load_signed(path, &key, "external WORM receipt")?;
    validate_id("receipt_id", &receipt.receipt_id)?;
    if receipt.sink_id != policy.worm_sink_id
        || receipt.event_kind != event_kind
        || receipt.transaction_id != tx.transaction_id
        || receipt.intent_hash != tx.intent_hash
        || receipt.reference_hash != reference_hash
        || receipt.recorded_at != *occurred_at
    {
        return Err(SurfaceError::policy(
            "external WORM receipt is not bound to the exact execution event",
        ));
    }
    hash_value(&receipt)
}
pub(crate) fn verify_role(
    authorities: &std::collections::BTreeMap<String, String>,
    role: &str,
    policy: &PolicyFile,
    tx: &Transaction,
    authority_id: &str,
    reference: &str,
    signature_hex: &str,
) -> SurfaceResult<()> {
    validate_id("custody authority id", authority_id)?;
    let key_hex = authorities.get(authority_id).ok_or_else(|| {
        SurfaceError::policy("custody authority is not authorized by signed policy")
    })?;
    let key = verifying_key_from_hex(key_hex)?;
    let bytes = hex::decode(signature_hex)
        .map_err(|_| SurfaceError::policy("invalid custody signature encoding"))?;
    let signature = Signature::from_slice(&bytes)
        .map_err(|_| SurfaceError::policy("invalid custody signature"))?;
    let message = format!(
        "singularity-finance-{role}-v1:{}:{}:{}:{}:{reference}",
        policy.policy_id, policy.version, tx.transaction_id, tx.intent_hash
    );
    key.verify(message.as_bytes(), &signature)
        .map_err(|_| SurfaceError::policy("independent custody signature verification failed"))
}
pub(crate) fn verify_approval(
    policy: &PolicyFile,
    tx: &Transaction,
    approver_id: &str,
    signature_hex: &str,
) -> SurfaceResult<()> {
    let key_hex = policy
        .approval
        .approver_keys
        .get(approver_id)
        .ok_or_else(|| SurfaceError::policy("approver is not authorized by signed policy"))?;
    let key = verifying_key_from_hex(key_hex)?;
    let bytes = hex::decode(signature_hex)
        .map_err(|_| SurfaceError::policy("invalid approval signature encoding"))?;
    let signature = Signature::from_slice(&bytes)
        .map_err(|_| SurfaceError::policy("invalid approval signature"))?;
    let simulation_evidence_hash = tx
        .simulation_evidence_hash
        .as_deref()
        .ok_or_else(|| SurfaceError::policy("approval requires accepted simulation evidence"))?;
    let message = format!(
        "singularity-finance-approval-v1:{}:{}:{}:{}:{}",
        policy.policy_id,
        policy.version,
        tx.transaction_id,
        tx.intent_hash,
        simulation_evidence_hash
    );
    key.verify(message.as_bytes(), &signature).map_err(|_| {
        SurfaceError::policy("exact-intent and simulation approval signature verification failed")
    })
}
pub(crate) fn public_status(tx: &Transaction) -> Value {
    json!({"transaction_id":tx.transaction_id,"status":tx.status,"intent_hash":tx.intent_hash,"beneficiary_id":tx.intent.beneficiary_id,"asset":tx.intent.asset,"amount_minor":tx.intent.amount_minor,"purpose":tx.intent.purpose,"parameters":tx.intent.parameters,"expires_at":tx.intent.expires_at,"approval_count":tx.approvals.len(),"timelock_until":tx.timelock_until,"reconciliation_required":tx.reconciliation_required,"policy_id":tx.policy_id,"policy_version":tx.policy_version})
}
