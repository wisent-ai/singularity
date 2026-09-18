//! The owner events that end or suspend a transaction: a rejection, a failure, an
//! indeterminate effect, a reconciliation that found nothing submitted, a quarantine.
use crate::finance_surface::policy::validate_id;
use crate::finance_surface::state::{Transaction, TransactionStatus};
use crate::finance_surface::{SurfaceError, SurfaceResult};

use super::{FinanceService, pre_signing_state, require_hash, transition, verify_role};

impl FinanceService {
    pub(super) fn record_rejection(
        &self,
        tx: &mut Transaction,
        reason_code: &str,
        event_hash: &str,
    ) -> SurfaceResult<String> {
        validate_id("reason_code", reason_code)?;
        if !pre_signing_state(tx.status) {
            return Err(SurfaceError::conflict(
                "rejection is allowed only before signing",
            ));
        }
        transition(
            tx,
            TransactionStatus::Rejected,
            "operator",
            Some(event_hash.to_string()),
        );
        Ok(event_hash.to_string())
    }

    pub(super) fn record_failure(
        &self,
        tx: &mut Transaction,
        reason_code: &str,
        event_hash: &str,
    ) -> SurfaceResult<String> {
        validate_id("reason_code", reason_code)?;
        if !pre_signing_state(tx.status) {
            return Err(SurfaceError::conflict(
                "failure is allowed only before signing",
            ));
        }
        transition(
            tx,
            TransactionStatus::Failed,
            "owner",
            Some(event_hash.to_string()),
        );
        Ok(event_hash.to_string())
    }

    pub(super) fn record_indeterminate(
        &self,
        tx: &mut Transaction,
        reason_code: &str,
        executor_id: &str,
        executor_signature_hex: &str,
        event_hash: &str,
    ) -> SurfaceResult<String> {
        validate_id("reason_code", reason_code)?;
        verify_role(
            &self.policy.custody_authorities.executors,
            "indeterminate",
            &self.policy,
            tx,
            executor_id,
            reason_code,
            executor_signature_hex,
        )?;
        if !matches!(
            tx.status,
            TransactionStatus::Signed | TransactionStatus::Submitted
        ) {
            return Err(SurfaceError::conflict(
                "indeterminate only follows signing/submission ambiguity",
            ));
        }
        tx.reconciliation_required = true;
        transition(
            tx,
            TransactionStatus::Indeterminate,
            "external_executor",
            Some(event_hash.to_string()),
        );
        Ok(event_hash.to_string())
    }

    pub(super) fn record_not_submitted(
        &self,
        tx: &mut Transaction,
        reconciliation_hash: &str,
        reconciler_id: &str,
        reconciler_signature_hex: &str,
    ) -> SurfaceResult<String> {
        require_hash(reconciliation_hash)?;
        verify_role(
            &self.policy.custody_authorities.reconcilers,
            "not_submitted",
            &self.policy,
            tx,
            reconciler_id,
            reconciliation_hash,
            reconciler_signature_hex,
        )?;
        if !matches!(
            tx.status,
            TransactionStatus::Indeterminate | TransactionStatus::Quarantined
        ) {
            return Err(SurfaceError::conflict(
                "reconciliation requires indeterminate or quarantined state",
            ));
        }
        tx.reconciliation_required = false;
        transition(
            tx,
            TransactionStatus::Signed,
            "external_reconciler",
            Some(reconciliation_hash.to_string()),
        );
        Ok(reconciliation_hash.to_string())
    }

    pub(super) fn record_quarantine(
        &self,
        tx: &mut Transaction,
        reason_code: &str,
        event_hash: &str,
    ) -> SurfaceResult<String> {
        validate_id("reason_code", reason_code)?;
        transition(
            tx,
            TransactionStatus::Quarantined,
            "owner",
            Some(event_hash.to_string()),
        );
        Ok(event_hash.to_string())
    }
}
