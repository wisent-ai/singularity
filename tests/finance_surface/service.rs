// Finance-contract stories, attached to src/finance_surface/service.rs.

use std::collections::BTreeMap;
use std::sync::{Arc, Barrier};

use ed25519_dalek::{Signer, SigningKey};

use super::*;
use chrono::Duration;
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::finance_surface::policy::{
    ApprovalPolicy, AssetPolicy, Beneficiary, CustodyAuthorities, SignedDocument,
};
use crate::finance_surface::state::{
    CanonicalIntent, StateTransition, Transaction, TransactionStatus,
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
                parameters: Some(json!({})),
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
        .save_transaction(&fixture.transaction("existing", 850, TransactionStatus::ApprovalPending))
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
    let mut transaction = fixture.transaction("approval", 10, TransactionStatus::ApprovalPending);
    let well_formed_signature = hex::encode(fixture.approver_key.sign(b"irrelevant").to_bytes());
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
                expected = "policy_denied: beneficiary rolling, daily, or lifetime limit exceeded";
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
                expected = "policy_denied: beneficiary rolling, daily, or lifetime limit exceeded";
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
                expected = "policy_denied: beneficiary rolling, daily, or lifetime limit exceeded";
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
    let reference = "a".repeat(HASH_HEX_CHARS);
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
