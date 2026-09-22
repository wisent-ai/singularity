// What happens to one transaction over its life: the timelock that must
// elapse, the cancellation that fails at the approval deadline, the effect
// nobody can resubmit before reconciliation, and the concurrent reuse of one
// request that must leave exactly one durable winner.

use std::sync::{Arc, Barrier};

use super::super::*;
use chrono::{Duration, Utc};

use crate::finance_surface::state::TransactionStatus;

use super::fixture::{proposal, Fixture};

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
