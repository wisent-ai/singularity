//! The owner events the surface ingests, and the checks every one of them passes.
//!
//! The custody events that move a transaction forward are in `custody`; the ones that
//! end or suspend it are in `outcome`. The match below is the only place that decides
//! which of them an event is, and it stays exhaustive over `OwnerAction`.
use chrono::{Duration, Utc};
use serde_json::{Value, json};
use std::path::Path;

use crate::finance_surface::policy::{load_signed, validate_id};
use crate::finance_surface::state::RequestRecord;
use crate::finance_surface::{SurfaceError, SurfaceResult};

use super::{FinanceService, OwnerAction, OwnerEvent, hash_value, public_status};
use super::{OWNER_EVENT_MAX_AGE_HOURS, OWNER_EVENT_MAX_SKEW_MINUTES};

/// The ledger identifier of an owner event is the first forty characters of the hash of
/// its own identifier, so a replayed event answers with the transaction it already wrote.
const LEDGER_ID_HEX_CHARS: usize = 40;

impl FinanceService {
    pub fn ingest_owner_event(&self, path: &Path) -> SurfaceResult<Value> {
        let event: OwnerEvent = load_signed(path, &self.verifying_key, "owner event")?;
        validate_id("event_id", &event.event_id)?;
        validate_id("transaction_id", &event.transaction_id)?;
        let _lock = self.state.lock()?;
        let event_hash = hash_value(&event)?;
        let event_id_hash = hash_value(&event.event_id)?;
        let ledger_id = format!("owner_{}", &event_id_hash[..LEDGER_ID_HEX_CHARS]);
        if let Some(r) = self.state.load_request(&ledger_id)? {
            if r.input_hash != event_hash {
                return Err(SurfaceError::conflict("owner event id reused"));
            }
            return Ok(r.response);
        }
        let mut tx = self.state.load_transaction(&event.transaction_id)?;
        self.refresh_time(&mut tx)?;
        if matches!(
            &event.action,
            OwnerAction::SimulationAccepted { .. }
                | OwnerAction::ApprovalGranted { .. }
                | OwnerAction::Signed { .. }
                | OwnerAction::Submitted { .. }
        ) {
            self.active_lease()?;
        }
        if tx.intent_hash != event.intent_hash {
            return Err(SurfaceError::policy(
                "owner event does not approve exact intent hash",
            ));
        }
        if event.occurred_at < Utc::now() - Duration::hours(OWNER_EVENT_MAX_AGE_HOURS)
            || event.occurred_at > Utc::now() + Duration::minutes(OWNER_EVENT_MAX_SKEW_MINUTES)
        {
            return Err(SurfaceError::policy(
                "owner event timestamp outside acceptance window",
            ));
        }
        if tx.status.terminal() {
            return Err(SurfaceError::conflict(
                "terminal transaction state is immutable",
            ));
        }
        match &event.action {
            OwnerAction::SimulationAccepted {
                evidence_hash,
                simulator_id,
                simulator_signature_hex,
            } => self.accept_simulation(
                &mut tx,
                evidence_hash,
                simulator_id,
                simulator_signature_hex,
            )?,
            OwnerAction::ApprovalGranted {
                approver_id,
                approval_signature_hex,
            } => self.grant_approval(&mut tx, approver_id, approval_signature_hex, &event_hash)?,
            OwnerAction::Signed {
                signer_attestation_hash,
                signer_id,
                signer_signature_hex,
            } => self.record_signature(
                &mut tx,
                signer_attestation_hash,
                signer_id,
                signer_signature_hex,
            )?,
            OwnerAction::Submitted {
                executor_reference_hash,
                executor_id,
                executor_signature_hex,
                worm_receipt_file,
            } => self.record_submission(
                &mut tx,
                executor_reference_hash,
                executor_id,
                executor_signature_hex,
                worm_receipt_file,
                &event,
            )?,
            OwnerAction::Confirmed {
                reconciliation_hash,
                reconciler_id,
                reconciler_signature_hex,
                worm_receipt_file,
            } => self.record_confirmation(
                &mut tx,
                reconciliation_hash,
                reconciler_id,
                reconciler_signature_hex,
                worm_receipt_file,
                &event,
            )?,
            OwnerAction::Rejected { reason_code } => {
                self.record_rejection(&mut tx, reason_code, &event_hash)?
            }
            OwnerAction::Failed { reason_code } => {
                self.record_failure(&mut tx, reason_code, &event_hash)?
            }
            OwnerAction::Indeterminate {
                reason_code,
                executor_id,
                executor_signature_hex,
            } => self.record_indeterminate(
                &mut tx,
                reason_code,
                executor_id,
                executor_signature_hex,
                &event_hash,
            )?,
            OwnerAction::ReconciledNotSubmitted {
                reconciliation_hash,
                reconciler_id,
                reconciler_signature_hex,
            } => self.record_not_submitted(
                &mut tx,
                reconciliation_hash,
                reconciler_id,
                reconciler_signature_hex,
            )?,
            OwnerAction::Quarantined { reason_code } => {
                self.record_quarantine(&mut tx, reason_code, &event_hash)?
            }
        };
        let response = public_status(&tx);
        let request = RequestRecord {
            operation: "owner_event".into(),
            input_hash: event_hash.clone(),
            transaction_id: tx.transaction_id.clone(),
            response: response.clone(),
        };
        self.state.commit(&ledger_id,&tx,Some((&ledger_id,&request)),json!({"type":"owner_event","event_id":event.event_id,"transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash,"event_hash":event_hash,"status":tx.status}))?;
        Ok(response)
    }
}

mod custody;
mod outcome;
