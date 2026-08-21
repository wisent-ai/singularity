use chrono::{Duration, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::policy::{
    EnableLease, PolicyFile, load_signed, validate_asset, validate_id, verifying_key_from_hex,
};
use super::state::{
    CanonicalIntent, RequestRecord, StateStore, StateTransition, Transaction, TransactionStatus,
};
use super::{SurfaceError, SurfaceResult};

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
    parameters: Value,
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
            &super::policy::document_hash(&lease)?,
        )?;
        Ok(lease)
    }

    fn propose(&self, input: Propose) -> SurfaceResult<Value> {
        validate_id("request_id", &input.request_id)?;
        validate_id("beneficiary_id", &input.beneficiary_id)?;
        validate_asset(&input.asset)?;
        if input.amount_minor <= 0 {
            return Err(SurfaceError::invalid(
                "amount_minor must be a positive integer",
            ));
        }
        if input.purpose.is_empty()
            || input.purpose.len() > 512
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

    fn status(&self, input: Status) -> SurfaceResult<Value> {
        validate_id("transaction_id", &input.transaction_id)?;
        let _lock = self.state.lock()?;
        let mut tx = self.state.load_transaction(&input.transaction_id)?;
        self.refresh_time(&mut tx)?;
        Ok(public_status(&tx))
    }
    fn cancel(&self, input: Cancel) -> SurfaceResult<Value> {
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

    async fn execute(&self, input: Execute) -> SurfaceResult<Value> {
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
            json!({
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
            })
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
        if output.stdout.len() > 64 * 1024 {
            return Err(SurfaceError::internal(
                "executor response exceeds size limit",
            ));
        }
        let response: Value = serde_json::from_slice(&output.stdout).map_err(|error| {
            SurfaceError::internal(format!("executor returned invalid JSON: {error}"))
        })?;
        let event_file = response
            .get("signed_owner_event_file")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .ok_or_else(|| {
                SurfaceError::internal(
                    "executor response must name an absolute signed_owner_event_file",
                )
            })?;
        self.ingest_owner_event(&event_file)
    }

    fn refresh_time(&self, tx: &mut Transaction) -> SurfaceResult<()> {
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

    fn enforce_limits(
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

    pub fn ingest_owner_event(&self, path: &Path) -> SurfaceResult<Value> {
        let event: OwnerEvent = load_signed(path, &self.verifying_key, "owner event")?;
        validate_id("event_id", &event.event_id)?;
        validate_id("transaction_id", &event.transaction_id)?;
        let _lock = self.state.lock()?;
        let event_hash = hash_value(&event)?;
        let event_id_hash = hash_value(&event.event_id)?;
        let ledger_id = format!("owner_{}", &event_id_hash[..40]);
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
        if event.occurred_at < Utc::now() - Duration::hours(24)
            || event.occurred_at > Utc::now() + Duration::minutes(5)
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
        let evidence = match &event.action {
            OwnerAction::SimulationAccepted {
                evidence_hash,
                simulator_id,
                simulator_signature_hex,
            } => {
                require_hash(evidence_hash)?;
                verify_role(
                    &self.policy.custody_authorities.simulators,
                    "simulation",
                    &self.policy,
                    &tx,
                    simulator_id,
                    evidence_hash,
                    simulator_signature_hex,
                )?;
                if tx.status != TransactionStatus::ApprovalPending
                    || tx.simulation_evidence_hash.is_some()
                {
                    return Err(SurfaceError::conflict(
                        "simulation event invalid in current state",
                    ));
                }
                tx.simulation_evidence_hash = Some(evidence_hash.clone());
                transition(
                    &mut tx,
                    TransactionStatus::Simulated,
                    "external_simulator",
                    Some(evidence_hash.clone()),
                );
                transition(
                    &mut tx,
                    TransactionStatus::ApprovalPending,
                    "finance_monitor",
                    None,
                );
                Some(evidence_hash.clone())
            }
            OwnerAction::ApprovalGranted {
                approver_id,
                approval_signature_hex,
            } => {
                validate_id("approver_id", approver_id)?;
                if tx.status != TransactionStatus::ApprovalPending
                    || tx.simulation_evidence_hash.is_none()
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
                verify_approval(&self.policy, &tx, approver_id, approval_signature_hex)?;
                tx.approvals.insert(approver_id.clone(), event_hash.clone());
                if tx.approvals.len() >= self.policy.approval.required_approvals as usize {
                    transition(
                        &mut tx,
                        TransactionStatus::Approved,
                        "operator",
                        Some(event_hash.clone()),
                    );
                    tx.timelock_until = Utc::now()
                        .checked_add_signed(Duration::seconds(
                            i64::try_from(self.policy.approval.timelock_seconds)
                                .map_err(|_| SurfaceError::policy("timelock overflow"))?,
                        ))
                        .ok_or_else(|| SurfaceError::policy("timelock overflow"))?;
                    transition(
                        &mut tx,
                        TransactionStatus::Timelocked,
                        "finance_monitor",
                        None,
                    );
                }
                Some(event_hash.clone())
            }
            OwnerAction::Signed {
                signer_attestation_hash,
                signer_id,
                signer_signature_hex,
            } => {
                require_hash(signer_attestation_hash)?;
                verify_role(
                    &self.policy.custody_authorities.signers,
                    "signing",
                    &self.policy,
                    &tx,
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
                    &mut tx,
                    TransactionStatus::Signed,
                    "external_signer",
                    Some(signer_attestation_hash.clone()),
                );
                Some(signer_attestation_hash.clone())
            }
            OwnerAction::Submitted {
                executor_reference_hash,
                executor_id,
                executor_signature_hex,
                worm_receipt_file,
            } => {
                require_hash(executor_reference_hash)?;
                verify_role(
                    &self.policy.custody_authorities.executors,
                    "submission",
                    &self.policy,
                    &tx,
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
                    &tx,
                    "submitted",
                    executor_reference_hash,
                    &event.occurred_at,
                    worm_receipt_file,
                )?;
                let worm = json!({"type":"submission","worm_sink_id":self.policy.worm_sink_id,"worm_receipt_hash":receipt_hash,"transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash,"executor_reference_hash":executor_reference_hash,"at":event.occurred_at});
                self.state.append_worm(&self.policy.worm_sink_dir, &worm)?;
                transition(
                    &mut tx,
                    TransactionStatus::Submitted,
                    "external_executor",
                    Some(executor_reference_hash.clone()),
                );
                Some(executor_reference_hash.clone())
            }
            OwnerAction::Confirmed {
                reconciliation_hash,
                reconciler_id,
                reconciler_signature_hex,
                worm_receipt_file,
            } => {
                require_hash(reconciliation_hash)?;
                verify_role(
                    &self.policy.custody_authorities.reconcilers,
                    "confirmation",
                    &self.policy,
                    &tx,
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
                    &tx,
                    "confirmed",
                    reconciliation_hash,
                    &event.occurred_at,
                    worm_receipt_file,
                )?;
                let worm = json!({"type":"confirmation","worm_sink_id":self.policy.worm_sink_id,"worm_receipt_hash":receipt_hash,"transaction_id":tx.transaction_id,"intent_hash":tx.intent_hash,"reconciliation_hash":reconciliation_hash,"at":event.occurred_at});
                self.state.append_worm(&self.policy.worm_sink_dir, &worm)?;
                tx.reconciliation_required = false;
                transition(
                    &mut tx,
                    TransactionStatus::Confirmed,
                    "external_reconciler",
                    Some(reconciliation_hash.clone()),
                );
                Some(reconciliation_hash.clone())
            }
            OwnerAction::Rejected { reason_code } => {
                validate_id("reason_code", reason_code)?;
                if !pre_signing_state(tx.status) {
                    return Err(SurfaceError::conflict(
                        "rejection is allowed only before signing",
                    ));
                }
                transition(
                    &mut tx,
                    TransactionStatus::Rejected,
                    "operator",
                    Some(event_hash.clone()),
                );
                Some(event_hash.clone())
            }
            OwnerAction::Failed { reason_code } => {
                validate_id("reason_code", reason_code)?;
                if !pre_signing_state(tx.status) {
                    return Err(SurfaceError::conflict(
                        "failure is allowed only before signing",
                    ));
                }
                transition(
                    &mut tx,
                    TransactionStatus::Failed,
                    "owner",
                    Some(event_hash.clone()),
                );
                Some(event_hash.clone())
            }
            OwnerAction::Indeterminate {
                reason_code,
                executor_id,
                executor_signature_hex,
            } => {
                validate_id("reason_code", reason_code)?;
                verify_role(
                    &self.policy.custody_authorities.executors,
                    "indeterminate",
                    &self.policy,
                    &tx,
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
                    &mut tx,
                    TransactionStatus::Indeterminate,
                    "external_executor",
                    Some(event_hash.clone()),
                );
                Some(event_hash.clone())
            }
            OwnerAction::ReconciledNotSubmitted {
                reconciliation_hash,
                reconciler_id,
                reconciler_signature_hex,
            } => {
                require_hash(reconciliation_hash)?;
                verify_role(
                    &self.policy.custody_authorities.reconcilers,
                    "not_submitted",
                    &self.policy,
                    &tx,
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
                    &mut tx,
                    TransactionStatus::Signed,
                    "external_reconciler",
                    Some(reconciliation_hash.clone()),
                );
                Some(reconciliation_hash.clone())
            }
            OwnerAction::Quarantined { reason_code } => {
                validate_id("reason_code", reason_code)?;
                transition(
                    &mut tx,
                    TransactionStatus::Quarantined,
                    "owner",
                    Some(event_hash.clone()),
                );
                Some(event_hash.clone())
            }
        };
        let _ = evidence;
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

fn validate_execution_parameters(value: &Value) -> SurfaceResult<()> {
    if !value.is_null() && !value.is_object() {
        return Err(SurfaceError::invalid("parameters must be a JSON object"));
    }
    if serde_json::to_vec(value)
        .map_err(|error| SurfaceError::invalid(format!("invalid parameters: {error}")))?
        .len()
        > 16 * 1024
    {
        return Err(SurfaceError::invalid("parameters exceed size limit"));
    }
    fn walk(value: &Value, depth: usize) -> SurfaceResult<()> {
        if depth > 8 {
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

fn parse<T: for<'de> Deserialize<'de>>(v: Value) -> SurfaceResult<T> {
    serde_json::from_value(v).map_err(|e| SurfaceError::invalid(format!("invalid arguments: {e}")))
}
fn hash_value<T: Serialize>(v: &T) -> SurfaceResult<String> {
    let value = serde_json::to_value(v)
        .map_err(|e| SurfaceError::internal(format!("cannot canonicalize value: {e}")))?;
    let b = super::policy::canonical_json(&value)?;
    Ok(hex::encode(Sha256::digest(b)))
}
fn checked_add(a: i64, b: i64) -> SurfaceResult<i64> {
    a.checked_add(b)
        .ok_or_else(|| SurfaceError::policy("financial total overflow"))
}
fn require_hash(v: &str) -> SurfaceResult<()> {
    if v.len() != 64 || !v.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(SurfaceError::invalid(
            "evidence hash must be 64 hexadecimal characters",
        ));
    }
    Ok(())
}
fn transition(
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
fn execution_time(tx: &Transaction) -> Option<chrono::DateTime<Utc>> {
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
fn pre_signing_state(status: TransactionStatus) -> bool {
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
fn verify_worm_receipt(
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
fn verify_role(
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
fn verify_approval(
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
fn public_status(tx: &Transaction) -> Value {
    json!({"transaction_id":tx.transaction_id,"status":tx.status,"intent_hash":tx.intent_hash,"beneficiary_id":tx.intent.beneficiary_id,"asset":tx.intent.asset,"amount_minor":tx.intent.amount_minor,"purpose":tx.intent.purpose,"parameters":tx.intent.parameters,"expires_at":tx.intent.expires_at,"approval_count":tx.approvals.len(),"timelock_until":tx.timelock_until,"reconciliation_required":tx.reconciliation_required,"policy_id":tx.policy_id,"policy_version":tx.policy_version})
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Barrier};

    use ed25519_dalek::{Signer, SigningKey};

    use super::*;
    use crate::finance_surface::policy::{
        ApprovalPolicy, AssetPolicy, Beneficiary, CustodyAuthorities, SignedDocument,
    };

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "wisent-finance-service-test-{}",
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn key_hex(key: &SigningKey) -> String {
        hex::encode(key.verifying_key().to_bytes())
    }

    fn write_signed<T: Serialize>(path: &Path, document: &T, signer: &SigningKey) {
        use std::os::unix::fs::OpenOptionsExt;

        let value = serde_json::to_value(document).unwrap();
        let signature = signer.sign(&super::super::policy::canonical_json(&value).unwrap());
        let envelope = SignedDocument {
            document: value,
            signature_hex: hex::encode(signature.to_bytes()),
        };
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true).mode(0o600);
        serde_json::to_writer(options.open(path).unwrap(), &envelope).unwrap();
    }

    struct Fixture {
        _directory: TestDirectory,
        service: FinanceService,
        state: StateStore,
        document_key: SigningKey,
        approver_key: SigningKey,
        executor_key: SigningKey,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = TestDirectory::new();
            let document_key = key(1);
            let approver_key = key(2);
            let simulator_key = key(3);
            let signer_key = key(4);
            let executor_key = key(5);
            let reconciler_key = key(6);
            let worm_key = key(7);
            let now = Utc::now();
            let worm = directory.0.join("worm");
            std::fs::create_dir(&worm).unwrap();
            let policy = PolicyFile {
                policy_id: "policy".into(),
                version: 1,
                valid_from: now - Duration::hours(1),
                expires_at: now + Duration::hours(1),
                beneficiaries: BTreeMap::from([(
                    "beneficiary".into(),
                    Beneficiary {
                        destination: "vault-address".into(),
                        allowed_assets: vec!["USD".into()],
                        allowed_purposes: vec!["invoice".into()],
                        valid_from: now - Duration::hours(1),
                        expires_at: now + Duration::hours(1),
                        per_transaction_minor: 1_000,
                        rolling_window_seconds: 3_600,
                        rolling_limit_minor: 10_000,
                        daily_limit_minor: 10_000,
                        lifetime_limit_minor: 10_000,
                        enabled: true,
                    },
                )]),
                assets: BTreeMap::from([(
                    "USD".into(),
                    AssetPolicy {
                        per_transaction_minor: 1_000,
                        rolling_window_seconds: 3_600,
                        rolling_limit_minor: 1_000,
                        daily_limit_minor: 2_000,
                        lifetime_limit_minor: 10_000,
                        spendable_balance_minor: 1_000,
                        protected_reserve_minor: 100,
                    },
                )]),
                approval: ApprovalPolicy {
                    required_approvals: 1,
                    approver_keys: BTreeMap::from([("approver".into(), key_hex(&approver_key))]),
                    timelock_seconds: 60,
                    proposal_ttl_max_seconds: 3_600,
                },
                custody_authorities: CustodyAuthorities {
                    simulators: BTreeMap::from([("simulator".into(), key_hex(&simulator_key))]),
                    signers: BTreeMap::from([("signer".into(), key_hex(&signer_key))]),
                    executors: BTreeMap::from([("executor".into(), key_hex(&executor_key))]),
                    reconcilers: BTreeMap::from([("reconciler".into(), key_hex(&reconciler_key))]),
                },
                worm_sink_dir: worm,
                worm_sink_id: "worm".into(),
                worm_receipt_key_hex: key_hex(&worm_key),
            };
            let lease = EnableLease {
                policy_id: policy.policy_id.clone(),
                policy_version: policy.version,
                lease_id: "lease".into(),
                issued_at: now - Duration::minutes(1),
                expires_at: now + Duration::minutes(30),
                enabled: true,
                kill_switch: false,
            };
            let lease_path = directory.0.join("lease.json");
            write_signed(&lease_path, &lease, &document_key);
            let state = StateStore::open(directory.0.join("state")).unwrap();
            let service = FinanceService::new(
                policy,
                state.clone(),
                lease_path,
                document_key.verifying_key(),
                PathBuf::from("/usr/bin/false"),
            );
            Self {
                _directory: directory,
                service,
                state,
                document_key,
                approver_key,
                executor_key,
            }
        }

        fn transaction(&self, id: &str, amount: i64, status: TransactionStatus) -> Transaction {
            let now = Utc::now();
            Transaction {
                transaction_id: id.into(),
                request_id: format!("request-{id}"),
                policy_id: self.service.policy.policy_id.clone(),
                policy_version: self.service.policy.version,
                lease_id: "lease".into(),
                intent: CanonicalIntent {
                    beneficiary_id: "beneficiary".into(),
                    asset: "USD".into(),
                    amount_minor: amount,
                    purpose: "invoice".into(),
                    parameters: json!({}),
                    expires_at: now + Duration::minutes(10),
                },
                intent_hash: format!("{id:0<64}"),
                created_at: now,
                status,
                transitions: vec![],
                approvals: BTreeMap::new(),
                simulation_evidence_hash: None,
                approval_deadline: now + Duration::minutes(10),
                timelock_until: now + Duration::minutes(1),
                reconciliation_required: false,
            }
        }
    }

    fn proposal(request_id: &str, amount: i64) -> Value {
        json!({
            "request_id": request_id,
            "beneficiary_id": "beneficiary",
            "asset": "USD",
            "amount_minor": amount,
            "purpose": "invoice",
            "ttl_seconds": 600
        })
    }

    fn make_caps_permissive(fixture: &mut Fixture) {
        let asset = fixture.service.policy.assets.get_mut("USD").unwrap();
        asset.per_transaction_minor = 1_000_000;
        asset.rolling_limit_minor = 1_000_000;
        asset.daily_limit_minor = 1_000_000;
        asset.lifetime_limit_minor = 1_000_000;
        asset.spendable_balance_minor = 1_000_000;
        asset.protected_reserve_minor = 0;

        let beneficiary = fixture
            .service
            .policy
            .beneficiaries
            .get_mut("beneficiary")
            .unwrap();
        beneficiary.per_transaction_minor = 1_000_000;
        beneficiary.rolling_limit_minor = 1_000_000;
        beneficiary.daily_limit_minor = 1_000_000;
        beneficiary.lifetime_limit_minor = 1_000_000;
    }

    fn record_execution_at(transaction: &mut Transaction, at: chrono::DateTime<Utc>) {
        transaction.transitions.push(StateTransition {
            status: transaction.status,
            at,
            actor: "executor".into(),
            evidence_hash: Some("evidence".into()),
        });
    }

    #[test]
    fn finance_contract_rejects_raw_destinations_and_unknown_arguments() {
        let mut arguments = proposal("request", 1);
        arguments["destination"] = json!("raw-wallet-address");

        let error = match parse::<Propose>(arguments) {
            Ok(_) => panic!("raw destination was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("unknown field `destination`"));
    }

    #[tokio::test]
    async fn finance_contract_replays_identical_requests_and_rejects_changed_intents() {
        let fixture = Fixture::new();
        let first = fixture
            .service
            .call("finance_propose", proposal("same-request", 100))
            .await
            .unwrap();

        let replay = fixture
            .service
            .call("finance_propose", proposal("same-request", 100))
            .await
            .unwrap();
        let conflict = fixture
            .service
            .call("finance_propose", proposal("same-request", 101))
            .await
            .unwrap_err();

        assert_eq!(replay, first);
        assert_eq!(
            conflict.to_string(),
            "invalid_state: request_id was already used with different intent"
        );
    }

    #[test]
    fn finance_contract_enforces_rolling_limits_and_protected_reserve() {
        let fixture = Fixture::new();
        let now = Utc::now();
        fixture
            .state
            .save_transaction(&fixture.transaction(
                "existing",
                850,
                TransactionStatus::ApprovalPending,
            ))
            .unwrap();
        let next = fixture
            .transaction("next", 151, TransactionStatus::Proposed)
            .intent;

        let rolling = fixture.service.enforce_limits(&next, now).unwrap_err();
        assert_eq!(
            rolling.to_string(),
            "policy_denied: rolling, daily, or lifetime limit exceeded"
        );

        let mut reserve_fixture = Fixture::new();
        reserve_fixture
            .service
            .policy
            .assets
            .get_mut("USD")
            .unwrap()
            .rolling_limit_minor = 10_000;
        reserve_fixture
            .state
            .save_transaction(&reserve_fixture.transaction(
                "reserved",
                850,
                TransactionStatus::ApprovalPending,
            ))
            .unwrap();
        let reserve = reserve_fixture
            .service
            .enforce_limits(
                &reserve_fixture
                    .transaction("next", 51, TransactionStatus::Proposed)
                    .intent,
                now,
            )
            .unwrap_err();
        assert_eq!(
            reserve.to_string(),
            "policy_denied: protected reserve would be breached"
        );
    }

    #[test]
    fn finance_contract_approval_signature_is_bound_to_intent_and_simulation_hashes() {
        let fixture = Fixture::new();
        let mut transaction =
            fixture.transaction("approval", 10, TransactionStatus::ApprovalPending);
        let well_formed_signature =
            hex::encode(fixture.approver_key.sign(b"irrelevant").to_bytes());
        let missing_simulation = verify_approval(
            &fixture.service.policy,
            &transaction,
            "approver",
            &well_formed_signature,
        )
        .unwrap_err();
        assert_eq!(
            missing_simulation.to_string(),
            "policy_denied: approval requires accepted simulation evidence"
        );

        transaction.simulation_evidence_hash = Some("a".repeat(64));
        let original_intent_hash = transaction.intent_hash.clone();
        let message = format!(
            "singularity-finance-approval-v1:{}:{}:{}:{}:{}",
            fixture.service.policy.policy_id,
            fixture.service.policy.version,
            transaction.transaction_id,
            transaction.intent_hash,
            transaction.simulation_evidence_hash.as_deref().unwrap()
        );
        let signature = hex::encode(fixture.approver_key.sign(message.as_bytes()).to_bytes());
        verify_approval(
            &fixture.service.policy,
            &transaction,
            "approver",
            &signature,
        )
        .unwrap();

        transaction.intent_hash = "f".repeat(64);
        let intent_error = verify_approval(
            &fixture.service.policy,
            &transaction,
            "approver",
            &signature,
        )
        .unwrap_err();
        assert_eq!(
            intent_error.to_string(),
            "policy_denied: exact-intent and simulation approval signature verification failed"
        );

        transaction.intent_hash = original_intent_hash;
        transaction.simulation_evidence_hash = Some("b".repeat(64));
        let simulation_error = verify_approval(
            &fixture.service.policy,
            &transaction,
            "approver",
            &signature,
        )
        .unwrap_err();
        assert_eq!(
            simulation_error.to_string(),
            "policy_denied: exact-intent and simulation approval signature verification failed"
        );
    }

    #[tokio::test]
    async fn finance_contract_denies_disallowed_purpose_and_beneficiary_validity_windows() {
        enum Case {
            Purpose,
            NotYetValid,
            Expired,
        }

        for (name, case) in [
            ("disallowed-purpose", Case::Purpose),
            ("not-yet-valid-beneficiary", Case::NotYetValid),
            ("expired-beneficiary", Case::Expired),
        ] {
            let mut fixture = Fixture::new();
            let mut arguments = proposal(&format!("eligibility-{name}"), 1);
            let beneficiary = fixture
                .service
                .policy
                .beneficiaries
                .get_mut("beneficiary")
                .unwrap();
            match case {
                Case::Purpose => arguments["purpose"] = json!("payroll"),
                Case::NotYetValid => beneficiary.valid_from = Utc::now() + Duration::minutes(10),
                Case::Expired => beneficiary.expires_at = Utc::now() - Duration::minutes(10),
            }

            let error = fixture
                .service
                .call("finance_propose", arguments)
                .await
                .unwrap_err();

            assert_eq!(
                error.to_string(),
                "policy_denied: beneficiary is disabled, outside its validity window, or disallows the asset or purpose",
                "case: {name}"
            );
        }
    }

    #[tokio::test]
    async fn finance_contract_enforces_each_beneficiary_cap_when_asset_caps_allow() {
        enum Cap {
            PerTransaction,
            RollingReservation,
            DailySubmitted,
            LifetimeConfirmed,
        }

        for (name, cap) in [
            ("per-transaction", Cap::PerTransaction),
            ("rolling-reservation", Cap::RollingReservation),
            ("daily-submitted", Cap::DailySubmitted),
            ("lifetime-confirmed", Cap::LifetimeConfirmed),
        ] {
            let mut fixture = Fixture::new();
            make_caps_permissive(&mut fixture);
            let beneficiary = fixture
                .service
                .policy
                .beneficiaries
                .get_mut("beneficiary")
                .unwrap();
            let expected;
            match cap {
                Cap::PerTransaction => {
                    beneficiary.per_transaction_minor = 40;
                    expected = "policy_denied: per-transaction limit exceeded";
                }
                Cap::RollingReservation => {
                    beneficiary.rolling_limit_minor = 100;
                    expected =
                        "policy_denied: beneficiary rolling, daily, or lifetime limit exceeded";
                    fixture
                        .state
                        .save_transaction(&fixture.transaction(
                            "rolling-reservation",
                            60,
                            TransactionStatus::ApprovalPending,
                        ))
                        .unwrap();
                }
                Cap::DailySubmitted => {
                    beneficiary.daily_limit_minor = 100;
                    expected =
                        "policy_denied: beneficiary rolling, daily, or lifetime limit exceeded";
                    fixture
                        .state
                        .save_transaction(&fixture.transaction(
                            "daily-submitted",
                            60,
                            TransactionStatus::Submitted,
                        ))
                        .unwrap();
                }
                Cap::LifetimeConfirmed => {
                    beneficiary.lifetime_limit_minor = 100;
                    expected =
                        "policy_denied: beneficiary rolling, daily, or lifetime limit exceeded";
                    let mut confirmed =
                        fixture.transaction("lifetime-confirmed", 60, TransactionStatus::Confirmed);
                    record_execution_at(&mut confirmed, Utc::now() - Duration::days(2));
                    fixture.state.save_transaction(&confirmed).unwrap();
                }
            }

            let error = fixture
                .service
                .call(
                    "finance_propose",
                    proposal(&format!("beneficiary-cap-{name}"), 41),
                )
                .await
                .unwrap_err();

            assert_eq!(error.to_string(), expected, "case: {name}");
        }
    }

    #[test]
    fn finance_contract_timelock_must_elapse_before_ready() {
        let fixture = Fixture::new();
        let mut transaction = fixture.transaction("timelock", 10, TransactionStatus::Timelocked);
        transaction.timelock_until = Utc::now() + Duration::minutes(1);
        fixture.service.refresh_time(&mut transaction).unwrap();
        assert_eq!(transaction.status, TransactionStatus::Timelocked);

        transaction.timelock_until = Utc::now() - Duration::seconds(1);
        fixture.service.refresh_time(&mut transaction).unwrap();
        assert_eq!(transaction.status, TransactionStatus::Ready);
    }

    #[test]
    fn finance_contract_cancellation_fails_at_the_approval_deadline() {
        let fixture = Fixture::new();
        let mut transaction = fixture.transaction("cancel", 10, TransactionStatus::ApprovalPending);
        transaction.approval_deadline = Utc::now() - Duration::seconds(1);
        fixture.state.save_transaction(&transaction).unwrap();

        let error = fixture
            .service
            .cancel(Cancel {
                transaction_id: transaction.transaction_id,
                request_id: "cancel-request".into(),
            })
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid_state: cancellation deadline has passed"
        );
    }

    #[test]
    fn finance_contract_indeterminate_effect_cannot_be_resubmitted_before_reconciliation() {
        let fixture = Fixture::new();
        let mut transaction = fixture.transaction("ambiguous", 10, TransactionStatus::Signed);
        transaction.reconciliation_required = true;
        fixture.state.save_transaction(&transaction).unwrap();
        let reference = "a".repeat(64);
        let role_message = format!(
            "singularity-finance-submission-v1:{}:{}:{}:{}:{}",
            fixture.service.policy.policy_id,
            fixture.service.policy.version,
            transaction.transaction_id,
            transaction.intent_hash,
            reference
        );
        let event = OwnerEvent {
            event_id: "submit-event".into(),
            transaction_id: transaction.transaction_id,
            intent_hash: transaction.intent_hash,
            occurred_at: Utc::now(),
            action: OwnerAction::Submitted {
                executor_reference_hash: reference,
                executor_id: "executor".into(),
                executor_signature_hex: hex::encode(
                    fixture
                        .executor_key
                        .sign(role_message.as_bytes())
                        .to_bytes(),
                ),
                worm_receipt_file: fixture._directory.0.join("unused-receipt.json"),
            },
        };
        let path = fixture._directory.0.join("owner-event.json");
        write_signed(&path, &event, &fixture.document_key);

        let error = fixture.service.ingest_owner_event(&path).unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid_state: submission requires signed state and completed reconciliation"
        );
    }

    #[test]
    fn finance_contract_concurrent_request_reuse_has_one_durable_winner() {
        let fixture = Fixture::new();
        let service = Arc::new(fixture.service);
        let barrier = Arc::new(Barrier::new(3));
        let handles: Vec<_> = [100, 101]
            .into_iter()
            .map(|amount| {
                let service = Arc::clone(&service);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    service.propose(parse(proposal("racing-request", amount)).unwrap())
                })
            })
            .collect();
        barrier.wait();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        let recorded = fixture
            .state
            .load_request("racing-request")
            .unwrap()
            .unwrap();
        let winner = results
            .iter()
            .find_map(|result| result.as_ref().ok())
            .unwrap();
        assert_eq!(recorded.response, *winner);
    }
}
