// Append-only state store stories, attached to src/finance_surface/state.rs.

use super::*;
use std::collections::BTreeMap;

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "wisent-finance-state-test-{}",
            uuid::Uuid::new_v4()
        ));
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn transaction(id: &str) -> Transaction {
    let now = Utc::now();
    Transaction {
        transaction_id: id.into(),
        request_id: format!("request-{id}"),
        policy_id: "policy".into(),
        policy_version: 2,
        lease_id: "lease".into(),
        intent: CanonicalIntent {
            beneficiary_id: "beneficiary".into(),
            asset: "USD".into(),
            amount_minor: 10,
            purpose: "invoice".into(),
            parameters: Some(serde_json::json!({})),
            expires_at: now + chrono::Duration::minutes(10),
        },
        intent_hash: "a".repeat(64),
        created_at: now,
        status: TransactionStatus::ApprovalPending,
        transitions: vec![],
        approvals: BTreeMap::new(),
        simulation_evidence_hash: None,
        approval_deadline: now + chrono::Duration::minutes(10),
        timelock_until: now + chrono::Duration::minutes(1),
        reconciliation_required: false,
    }
}

#[test]
fn finance_contract_rejects_policy_and_lease_rollback_or_equivocation() {
    let directory = TestDirectory::new();
    let store = StateStore::open(directory.0.clone()).unwrap();
    let now = Utc::now();
    store.bind_policy("policy", 2, "hash-v2").unwrap();
    store.bind_lease("lease-2", now, "lease-hash-2").unwrap();

    let policy_rollback = store.bind_policy("policy", 1, "hash-v1").unwrap_err();
    let policy_equivocation = store.bind_policy("policy", 2, "different").unwrap_err();
    let lease_rollback = store
        .bind_lease("lease-1", now - chrono::Duration::seconds(1), "old")
        .unwrap_err();
    let lease_equivocation = store
        .bind_lease("other-lease", now, "different")
        .unwrap_err();

    assert_eq!(
        policy_rollback.to_string(),
        "policy_denied: signed policy rollback or equivocation detected"
    );
    assert_eq!(policy_equivocation.to_string(), policy_rollback.to_string());
    assert_eq!(
        lease_rollback.to_string(),
        "policy_denied: signed enable lease rollback or equivocation detected"
    );
    assert_eq!(lease_equivocation.to_string(), lease_rollback.to_string());
}

#[test]
fn finance_contract_reopen_rejects_a_corrupted_audit_record() {
    let directory = TestDirectory::new();
    let store = StateStore::open(directory.0.clone()).unwrap();
    store
        .append_audit(serde_json::json!({"type":"proposal"}))
        .unwrap();
    let audit_path = fs::read_dir(directory.0.join("audit"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    fs::write(audit_path, b"{}").unwrap();

    let error = match StateStore::open(directory.0.clone()) {
        Ok(_) => panic!("corrupted audit record was accepted"),
        Err(error) => error,
    };

    assert!(error
        .to_string()
        .starts_with("state_error: invalid audit record:"));
}

#[test]
fn finance_contract_reopen_recovers_an_unapplied_commit_journal() {
    let directory = TestDirectory::new();
    let store = StateStore::open(directory.0.clone()).unwrap();
    let transaction = transaction("crash-recovery");
    let response = serde_json::json!({"transaction_id":"crash-recovery"});
    let request = RequestRecord {
        operation: "finance_propose".into(),
        input_hash: "input-hash".into(),
        transaction_id: transaction.transaction_id.clone(),
        response: response.clone(),
    };
    let commit = CommitRecord {
        commit_id: "crash-commit".into(),
        transaction: transaction.clone(),
        request: Some(("crash-request".into(), request)),
        audit_event: serde_json::json!({
            "type":"proposal_created",
            "commit_id":"crash-commit"
        }),
    };
    atomic_json(
        &directory.0.join("commits/crash-commit.json"),
        &commit,
        true,
    )
    .unwrap();
    drop(store);

    let reopened = StateStore::open(directory.0.clone()).unwrap();

    assert_eq!(
        reopened
            .load_transaction("crash-recovery")
            .unwrap()
            .transaction_id,
        transaction.transaction_id
    );
    assert_eq!(
        reopened
            .load_request("crash-request")
            .unwrap()
            .unwrap()
            .response,
        response
    );
    assert!(directory
        .0
        .join("commit-applied/crash-commit.json")
        .exists());
}
