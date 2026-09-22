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


mod fixture;
mod lifecycle;

use fixture::{make_caps_permissive, proposal, record_execution_at, Fixture};
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

