//! Executing an approved transaction, refreshing its clock, and the limits it is held to.
use chrono::{Duration, Utc};
use serde_json::{Value, json};
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::{
    FinanceService, MAX_EXECUTOR_RESPONSE_BYTES, checked_add, execution_time, public_status,
    require_hash, transition, verify_role, verify_worm_receipt,
};
use crate::finance_surface::policy::validate_id;
use crate::finance_surface::state::{CanonicalIntent, Transaction, TransactionStatus};
use crate::finance_surface::{SurfaceError, SurfaceResult};

use super::{Execute, ExecutionResponse};
impl FinanceService {
    pub(super) async fn execute(&self, input: Execute) -> SurfaceResult<Value> {
        validate_id("transaction_id", &input.transaction_id)?;
        self.active_lease()?;
        let request = {
            let _lock = self.state.lock()?;
            let mut tx = self.state.load_transaction(&input.transaction_id)?;
            self.refresh_time(&mut tx)?;
            if tx.status != TransactionStatus::Signed || tx.reconciliation_required {
                return Err(SurfaceError::conflict(
                    "execution requires signed state and completed reconciliation",
                ));
            }
            let beneficiary = self
                .policy
                .beneficiaries
                .get(&tx.intent.beneficiary_id)
                .ok_or_else(|| SurfaceError::policy("beneficiary is absent from policy"))?;
            let request = json!({
                "version": "singularity.finance.execute.v1",
                "transaction_id": tx.transaction_id,
                "intent_hash": tx.intent_hash,
                "beneficiary_id": tx.intent.beneficiary_id,
                "destination": beneficiary.destination,
                "asset": tx.intent.asset,
                "amount_minor": tx.intent.amount_minor,
                "purpose": tx.intent.purpose,
                "parameters": tx.intent.parameters,
                "expires_at": tx.intent.expires_at,
                "policy_id": tx.policy_id,
                "policy_version": tx.policy_version,
            });
            tx.reconciliation_required = true;
            transition(
                &mut tx,
                TransactionStatus::Indeterminate,
                "finance_executor_dispatch",
                None,
            );
            let commit_id = format!("execute_dispatch_{}", tx.transaction_id);
            self.state.commit(
                &commit_id,
                &tx,
                None,
                json!({"type":"execution_dispatched","transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash}),
            )?;
            request
        };
        let mut child = Command::new(&self.executor)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| SurfaceError::internal(format!("cannot start executor: {error}")))?;
        let bytes = serde_json::to_vec(&request)
            .map_err(|error| SurfaceError::internal(format!("cannot encode execution: {error}")))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| SurfaceError::internal("executor stdin is unavailable"))?;
        stdin.write_all(&bytes).await.map_err(|error| {
            SurfaceError::internal(format!("cannot write executor request: {error}"))
        })?;
        drop(stdin);
        let output = child
            .wait_with_output()
            .await
            .map_err(|error| SurfaceError::internal(format!("executor wait failed: {error}")))?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr);
            return Err(SurfaceError::internal(format!(
                "executor refused: {}",
                detail.chars().take(512).collect::<String>()
            )));
        }
        if output.stdout.len() > MAX_EXECUTOR_RESPONSE_BYTES {
            return Err(SurfaceError::internal(
                "executor response exceeds size limit",
            ));
        }
        let response: ExecutionResponse =
            serde_json::from_slice(&output.stdout).map_err(|error| {
                SurfaceError::internal(format!("executor returned invalid JSON: {error}"))
            })?;
        require_hash(&response.executor_reference_hash)?;
        let occurred_at = Utc::now();
        let _lock = self.state.lock()?;
        let mut tx = self.state.load_transaction(&input.transaction_id)?;
        if tx.status != TransactionStatus::Indeterminate || !tx.reconciliation_required {
            return Err(SurfaceError::conflict(
                "execution response does not match dispatched state",
            ));
        }
        verify_role(
            &self.policy.custody_authorities.executors,
            "submission",
            &self.policy,
            &tx,
            &response.executor_id,
            &response.executor_reference_hash,
            &response.executor_signature_hex,
        )?;
        let receipt_hash = verify_worm_receipt(
            &self.policy,
            &tx,
            "submitted",
            &response.executor_reference_hash,
            &occurred_at,
            &response.worm_receipt_file,
        )?;
        let worm = json!({"type":"submission","worm_sink_id":self.policy.worm_sink_id,"worm_receipt_hash":receipt_hash,"transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash,"executor_reference_hash":response.executor_reference_hash,"at":occurred_at});
        self.state.append_worm(&self.policy.worm_sink_dir, &worm)?;
        tx.reconciliation_required = false;
        transition(
            &mut tx,
            TransactionStatus::Submitted,
            "isolated_executor",
            Some(response.executor_reference_hash),
        );
        let commit_id = format!("execute_submitted_{}", tx.transaction_id);
        self.state.commit(
            &commit_id,
            &tx,
            None,
            json!({"type":"execution_submitted","transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash}),
        )?;
        Ok(public_status(&tx))
    }

    pub(super) fn refresh_time(&self, tx: &mut Transaction) -> SurfaceResult<()> {
        let now = Utc::now();
        let before = tx.status;
        if !tx.status.terminal()
            && now >= tx.approval_deadline
            && !matches!(
                tx.status,
                TransactionStatus::Signed
                    | TransactionStatus::Submitted
                    | TransactionStatus::Indeterminate
                    | TransactionStatus::Quarantined
            )
        {
            transition(tx, TransactionStatus::Expired, "finance_monitor", None);
        } else if tx.status == TransactionStatus::Timelocked && now >= tx.timelock_until {
            transition(tx, TransactionStatus::Ready, "finance_monitor", None);
        }
        if tx.status != before {
            let commit_id = format!("auto_{}_{}", tx.transaction_id, tx.transitions.len());
            self.state.commit(&commit_id,tx,None,json!({"type":"automatic_transition","transaction_id":tx.transaction_id,"from":before,"to":tx.status}))?;
        }
        Ok(())
    }

    pub(super) fn enforce_limits(
        &self,
        intent: &CanonicalIntent,
        now: chrono::DateTime<Utc>,
    ) -> SurfaceResult<()> {
        let limits = self
            .policy
            .assets
            .get(&intent.asset)
            .ok_or_else(|| SurfaceError::policy("asset missing"))?;
        let beneficiary = self
            .policy
            .beneficiaries
            .get(&intent.beneficiary_id)
            .ok_or_else(|| SurfaceError::policy("beneficiary missing"))?;
        if !beneficiary.enabled
            || now < beneficiary.valid_from
            || now >= beneficiary.expires_at
            || intent.expires_at > beneficiary.expires_at
            || !beneficiary
                .allowed_assets
                .iter()
                .any(|v| v == &intent.asset)
            || !beneficiary
                .allowed_purposes
                .iter()
                .any(|v| v == &intent.purpose)
        {
            return Err(SurfaceError::policy(
                "beneficiary policy does not permit the intent",
            ));
        }
        if intent.amount_minor > limits.per_transaction_minor
            || intent.amount_minor > beneficiary.per_transaction_minor
        {
            return Err(SurfaceError::policy("per-transaction limit exceeded"));
        }
        let rolling_start = now
            - Duration::seconds(
                i64::try_from(limits.rolling_window_seconds)
                    .map_err(|_| SurfaceError::policy("rolling window overflow"))?,
            );
        let beneficiary_rolling_start = now
            - Duration::seconds(
                i64::try_from(beneficiary.rolling_window_seconds)
                    .map_err(|_| SurfaceError::policy("beneficiary rolling window overflow"))?,
            );
        let mut reserved = 0i64;
        let mut rolling = 0i64;
        let mut daily = 0i64;
        let mut lifetime = 0i64;
        let mut beneficiary_rolling = 0i64;
        let mut beneficiary_daily = 0i64;
        let mut beneficiary_lifetime = 0i64;
        for mut tx in self.state.all_transactions()? {
            self.refresh_time(&mut tx)?;
            if tx.intent.asset != intent.asset {
                continue;
            }
            let counted = tx.status.reserves_funds()
                || tx.status == TransactionStatus::Submitted
                || tx.status == TransactionStatus::Confirmed;
            if !counted {
                continue;
            }
            reserved = checked_add(reserved, tx.intent.amount_minor)?;
            lifetime = checked_add(lifetime, tx.intent.amount_minor)?;
            let window_time = if tx.status.reserves_funds() {
                now
            } else {
                execution_time(&tx).ok_or_else(|| {
                    SurfaceError::state("executed transaction lacks execution timestamp")
                })?
            };
            if tx.status.reserves_funds() || window_time >= rolling_start {
                rolling = checked_add(rolling, tx.intent.amount_minor)?;
            }
            if tx.status.reserves_funds() || window_time.date_naive() == now.date_naive() {
                daily = checked_add(daily, tx.intent.amount_minor)?;
            }
            if tx.intent.beneficiary_id == intent.beneficiary_id {
                beneficiary_lifetime = checked_add(beneficiary_lifetime, tx.intent.amount_minor)?;
                if tx.status.reserves_funds() || window_time >= beneficiary_rolling_start {
                    beneficiary_rolling = checked_add(beneficiary_rolling, tx.intent.amount_minor)?;
                }
                if tx.status.reserves_funds() || window_time.date_naive() == now.date_naive() {
                    beneficiary_daily = checked_add(beneficiary_daily, tx.intent.amount_minor)?;
                }
            }
        }
        let amount = intent.amount_minor;
        if checked_add(rolling, amount)? > limits.rolling_limit_minor
            || checked_add(daily, amount)? > limits.daily_limit_minor
            || checked_add(lifetime, amount)? > limits.lifetime_limit_minor
        {
            return Err(SurfaceError::policy(
                "rolling, daily, or lifetime limit exceeded",
            ));
        }
        if checked_add(beneficiary_rolling, amount)? > beneficiary.rolling_limit_minor
            || checked_add(beneficiary_daily, amount)? > beneficiary.daily_limit_minor
            || checked_add(beneficiary_lifetime, amount)? > beneficiary.lifetime_limit_minor
        {
            return Err(SurfaceError::policy(
                "beneficiary rolling, daily, or lifetime limit exceeded",
            ));
        }
        let available = limits
            .spendable_balance_minor
            .checked_sub(limits.protected_reserve_minor)
            .and_then(|v| v.checked_sub(reserved))
            .ok_or_else(|| SurfaceError::policy("protected reserve arithmetic failed"))?;
        if amount > available {
            return Err(SurfaceError::policy("protected reserve would be breached"));
        }
        Ok(())
    }
}
