// What a finance-contract story runs against: a disposable directory, the
// signing keys, the signed policy documents written to disk, and the fixture
// that assembles one service from them.

use std::collections::BTreeMap;

use ed25519_dalek::{Signer, SigningKey};

use super::super::*;
use chrono::Utc;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::finance_surface::policy::{
    ApprovalPolicy, AssetPolicy, Beneficiary, CustodyAuthorities, SignedDocument,
};
use crate::finance_surface::state::{StateTransition, Transaction};

pub(crate) struct TestDirectory(PathBuf);

impl TestDirectory {
    pub(crate) fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "wisent-finance-service-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    pub(crate) fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

pub(crate) fn key_hex(key: &SigningKey) -> String {
    hex::encode(key.verifying_key().to_bytes())
}

pub(crate) fn write_signed<T: Serialize>(path: &Path, document: &T, signer: &SigningKey) {
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

pub(crate) struct Fixture {
    _directory: TestDirectory,
    service: FinanceService,
    state: StateStore,
    document_key: SigningKey,
    approver_key: SigningKey,
    executor_key: SigningKey,
}

impl Fixture {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn transaction(&self, id: &str, amount: i64, status: TransactionStatus) -> Transaction {
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

pub(crate) fn proposal(request_id: &str, amount: i64) -> Value {
    json!({
        "request_id": request_id,
        "beneficiary_id": "beneficiary",
        "asset": "USD",
        "amount_minor": amount,
        "purpose": "invoice",
        "ttl_seconds": 600
    })
}

pub(crate) fn make_caps_permissive(fixture: &mut Fixture) {
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

pub(crate) fn record_execution_at(transaction: &mut Transaction, at: chrono::DateTime<Utc>) {
    transaction.transitions.push(StateTransition {
        status: transaction.status,
        at,
        actor: "executor".into(),
        evidence_hash: Some("evidence".into()),
    });
}

