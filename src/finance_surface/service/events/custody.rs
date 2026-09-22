//! The custody events that move a transaction forward: an accepted simulation, an
//! approval, a signature, a submission, a confirmation. Each returns the evidence hash
//! the ledger records, and each refuses a state its own step cannot follow.
use chrono::{Duration, Utc};
use serde_json::json;

use std::path::Path;

use crate::finance_surface::policy::validate_id;
use crate::finance_surface::state::{Transaction, TransactionStatus};
use crate::finance_surface::{SurfaceError, SurfaceResult};

use super::super::{
    FinanceService, OwnerEvent, require_hash, transition, verify_approval, verify_role,
    verify_worm_receipt,
};

impl FinanceService {
    pub(super) fn accept_simulation(
        &self,
        tx: &mut Transaction,
        evidence_hash: &str,
        simulator_id: &str,
        simulator_signature_hex: &str,
    ) -> SurfaceResult<String> {
        require_hash(evidence_hash)?;
        verify_role(
            &self.policy.custody_authorities.simulators,
            "simulation",
            &self.policy,
            tx,
            simulator_id,
            evidence_hash,
            simulator_signature_hex,
        )?;
        if tx.status != TransactionStatus::ApprovalPending || tx.simulation_evidence_hash.is_some()
        {
            return Err(SurfaceError::conflict(
                "simulation event invalid in current state",
            ));
        }
        tx.simulation_evidence_hash = Some(evidence_hash.to_string());
        transition(
            tx,
            TransactionStatus::Simulated,
            "external_simulator",
            Some(evidence_hash.to_string()),
        );
        transition(
            tx,
            TransactionStatus::ApprovalPending,
            "finance_monitor",
            None,
        );
        Ok(evidence_hash.to_string())
    }

    pub(super) fn grant_approval(
        &self,
        tx: &mut Transaction,
        approver_id: &str,
        approval_signature_hex: &str,
        event_hash: &str,
    ) -> SurfaceResult<String> {
        validate_id("approver_id", approver_id)?;
        if tx.status != TransactionStatus::ApprovalPending || tx.simulation_evidence_hash.is_none()
        {
            return Err(SurfaceError::conflict(
                "approval requires an independently accepted simulation",
            ));
        }
        if tx.approvals.contains_key(approver_id) {
            return Err(SurfaceError::conflict(
                "approver already approved this exact intent",
            ));
        }
        verify_approval(&self.policy, tx, approver_id, approval_signature_hex)?;
        tx.approvals
            .insert(approver_id.to_string(), event_hash.to_string());
        if tx.approvals.len() >= self.policy.approval.required_approvals as usize {
            transition(
                tx,
                TransactionStatus::Approved,
                "operator",
                Some(event_hash.to_string()),
            );
            tx.timelock_until = Utc::now()
                .checked_add_signed(Duration::seconds(
                    i64::try_from(self.policy.approval.timelock_seconds)
                        .map_err(|_| SurfaceError::policy("timelock overflow"))?,
                ))
                .ok_or_else(|| SurfaceError::policy("timelock overflow"))?;
            transition(tx, TransactionStatus::Timelocked, "finance_monitor", None);
        }
        Ok(event_hash.to_string())
    }

    pub(super) fn record_signature(
        &self,
        tx: &mut Transaction,
        signer_attestation_hash: &str,
        signer_id: &str,
        signer_signature_hex: &str,
    ) -> SurfaceResult<String> {
        require_hash(signer_attestation_hash)?;
        verify_role(
            &self.policy.custody_authorities.signers,
            "signing",
            &self.policy,
            tx,
            signer_id,
            signer_attestation_hash,
            signer_signature_hex,
        )?;
        if tx.status != TransactionStatus::Ready {
            return Err(SurfaceError::conflict(
                "signing requires completed timelock and ready state",
            ));
        }
        transition(
            tx,
            TransactionStatus::Signed,
            "external_signer",
            Some(signer_attestation_hash.to_string()),
        );
        Ok(signer_attestation_hash.to_string())
    }

    pub(super) fn record_submission(
        &self,
        tx: &mut Transaction,
        executor_reference_hash: &str,
        executor_id: &str,
        executor_signature_hex: &str,
        worm_receipt_file: &Path,
        event: &OwnerEvent,
    ) -> SurfaceResult<String> {
        require_hash(executor_reference_hash)?;
        verify_role(
            &self.policy.custody_authorities.executors,
            "submission",
            &self.policy,
            tx,
            executor_id,
            executor_reference_hash,
            executor_signature_hex,
        )?;
        if tx.status != TransactionStatus::Signed || tx.reconciliation_required {
            return Err(SurfaceError::conflict(
                "submission requires signed state and completed reconciliation",
            ));
        }
        let receipt_hash = verify_worm_receipt(
            &self.policy,
            tx,
            "submitted",
            executor_reference_hash,
            &event.occurred_at,
            worm_receipt_file,
        )?;
        let worm = json!({"type":"submission","worm_sink_id":self.policy.worm_sink_id,"worm_receipt_hash":receipt_hash,"transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash,"executor_reference_hash":executor_reference_hash,"at":event.occurred_at});
        self.state.append_worm(&self.policy.worm_sink_dir, &worm)?;
        transition(
            tx,
            TransactionStatus::Submitted,
            "external_executor",
            Some(executor_reference_hash.to_string()),
        );
        Ok(executor_reference_hash.to_string())
    }

    pub(super) fn record_confirmation(
        &self,
        tx: &mut Transaction,
        reconciliation_hash: &str,
        reconciler_id: &str,
        reconciler_signature_hex: &str,
        worm_receipt_file: &Path,
        event: &OwnerEvent,
    ) -> SurfaceResult<String> {
        require_hash(reconciliation_hash)?;
        verify_role(
            &self.policy.custody_authorities.reconcilers,
            "confirmation",
            &self.policy,
            tx,
            reconciler_id,
            reconciliation_hash,
            reconciler_signature_hex,
        )?;
        if !matches!(
            tx.status,
            TransactionStatus::Submitted
                | TransactionStatus::Indeterminate
                | TransactionStatus::Quarantined
        ) {
            return Err(SurfaceError::conflict(
                "confirmation requires submitted, indeterminate, or quarantined state",
            ));
        }
        let receipt_hash = verify_worm_receipt(
            &self.policy,
            tx,
            "confirmed",
            reconciliation_hash,
            &event.occurred_at,
            worm_receipt_file,
        )?;
        let worm = json!({"type":"confirmation","worm_sink_id":self.policy.worm_sink_id,"worm_receipt_hash":receipt_hash,"transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash,"reconciliation_hash":reconciliation_hash,"at":event.occurred_at});
        self.state.append_worm(&self.policy.worm_sink_dir, &worm)?;
        tx.reconciliation_required = false;
        transition(
            tx,
            TransactionStatus::Confirmed,
            "external_reconciler",
            Some(reconciliation_hash.to_string()),
        );
        Ok(reconciliation_hash.to_string())
    }
}
