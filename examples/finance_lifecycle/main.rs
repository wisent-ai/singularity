//! Drive one transaction through singularity-finance-mcp end to end: propose,
//! refuse, simulate, approve, timelock, sign, dispatch to a failing executor,
//! reconcile, submit, confirm — then flip and roll back the enable lease.
//!
//! No network, no custody system, no real money: all seven Ed25519 authority
//! keys are generated here, the executor is /usr/bin/false, and everything
//! lives in a fresh directory under target/.
//!
//! The policy's timelock is real, so the walk has two phases. The first
//! proposes and approves and prints the directory; once `timelock_until` has
//! passed, the same command given that directory signs, dispatches,
//! reconciles, submits, confirms and exercises the lease:
//!
//! ```text
//! cargo build --release --bin singularity-finance-mcp
//! cargo run --example finance_lifecycle
//! cargo run --example finance_lifecycle -- target/finance-walkthrough-<id>
//! ```
//!
//! SINGULARITY_FINANCE_MCP names another singularity-finance-mcp binary.

mod signing;
mod walk;

use std::os::unix::fs::DirBuilderExt;
use std::path::PathBuf;

use chrono::{Duration, Utc};
use serde_json::{json, Value};

use signing::{
    fresh_key, sha, ts, with, write_owner_only, Key, Outcome, Progress, POLICY_ID, VERSION,
};
use walk::Walk;

const TIMELOCK_SECONDS: u64 = 2;
const AUTHORITIES: usize = 7;
const PRIVATE_DIR: u32 = 0o700;

fn intent() -> Value {
    json!({"request_id": "req-invoice-001", "beneficiary_id": "infra-vendor", "asset": "USD",
        "amount_minor": 2500, "purpose": "invoice", "ttl_seconds": 900})
}

fn main() -> Outcome<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mcp = std::env::var_os("SINGULARITY_FINANCE_MCP")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target/release/singularity-finance-mcp"));
    match std::env::args_os().nth(1).map(PathBuf::from) {
        Some(dir) => {
            let progress: Progress =
                serde_json::from_str(&std::fs::read_to_string(dir.join("progress.json"))?)?;
            after_timelock(Walk::start(dir, mcp, progress)?)
        }
        None => {
            let dir = root
                .join("target")
                .join(format!("finance-walkthrough-{}", uuid::Uuid::new_v4()));
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(PRIVATE_DIR)
                .create(dir.join("worm"))?;
            before_timelock(dir, mcp)
        }
    }
}

/// Keys by role: document (policy, leases, owner events), approver,
/// simulator, signer, executor, reconciler, and the WORM receipt key.
fn policy(keys: &[Key], worm: &str) -> Value {
    let now = Utc::now();
    let limits = json!({"per_transaction_minor": 5000, "rolling_window_seconds": 3600, "rolling_limit_minor": 8000,
        "daily_limit_minor": 8000, "lifetime_limit_minor": 20000});
    json!({
        "policy_id": POLICY_ID, "version": VERSION,
        "valid_from": ts(now - Duration::hours(1)), "expires_at": ts(now + Duration::days(1)),
        "beneficiaries": {"infra-vendor": with(&limits, json!({"destination": "vendor-account-0001",
            "allowed_assets": ["USD"], "allowed_purposes": ["invoice"], "enabled": true,
            "valid_from": ts(now - Duration::hours(1)), "expires_at": ts(now + Duration::days(1))}))},
        "assets": {"USD": with(&limits, json!({"spendable_balance_minor": 10000, "protected_reserve_minor": 4000}))},
        "approval": {"required_approvals": 1, "approver_keys": {"treasury-owner": keys[1].public()},
            "timelock_seconds": TIMELOCK_SECONDS, "proposal_ttl_max_seconds": 3600,
            "owner_event_max_age_seconds": 86400, "owner_event_max_skew_seconds": 300},
        "custody_authorities": {"simulators": {"sim-1": keys[2].public()}, "signers": {"signer-1": keys[3].public()},
            "executors": {"exec-1": keys[4].public()}, "reconcilers": {"rec-1": keys[5].public()}},
        "worm_sink_dir": worm, "worm_sink_id": "walkthrough-worm", "worm_receipt_key_hex": keys[6].public(),
    })
}

fn before_timelock(dir: PathBuf, mcp: PathBuf) -> Outcome<()> {
    let keys: Vec<_> = (0..AUTHORITIES).map(|_| fresh_key()).collect();
    let worm = dir.join("worm").display().to_string();
    write_owner_only(
        &dir.join("policy.json"),
        &keys[0].envelope(&policy(&keys, &worm))?,
    )?;
    let progress = Progress {
        seeds: keys.iter().map(|key| key.seed()).collect(),
        first_lease: Utc::now() - Duration::seconds(30),
        tx: String::new(),
        hash: String::new(),
        events: u64::default(),
    };
    let first_lease = progress.first_lease;
    let mut walk = Walk::start(dir, mcp, progress)?;
    walk.lease("lease-1", first_lease, false)?;
    let intent = intent();
    let proposed = walk
        .call(
            "finance_propose (accepted)",
            "finance_propose",
            intent.clone(),
        )?
        .ok_or("the first proposal was refused")?;
    walk.progress.tx = proposed["transaction_id"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    walk.progress.hash = proposed["intent_hash"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let (tx, hash) = (walk.progress.tx.clone(), walk.progress.hash.clone());
    walk.call(
        "finance_propose replay (same request_id, same intent)",
        "finance_propose",
        intent.clone(),
    )?;
    for (label, changes) in [
        (
            "request_id reused with different intent",
            json!({"amount_minor": 2600}),
        ),
        (
            "unknown beneficiary",
            json!({"request_id": "req-bad-001", "beneficiary_id": "unknown-vendor"}),
        ),
        (
            "per-transaction limit",
            json!({"request_id": "req-bad-002", "amount_minor": 5001}),
        ),
        (
            "protected reserve",
            json!({"request_id": "req-bad-003", "amount_minor": 4000}),
        ),
        (
            "TTL beyond policy",
            json!({"request_id": "req-bad-004", "ttl_seconds": 4000}),
        ),
        (
            "protected intent fields in parameters",
            json!({"request_id": "req-bad-005", "parameters": {"destination": "elsewhere"}}),
        ),
    ] {
        walk.call(
            &format!("finance_propose refusal ({label})"),
            "finance_propose",
            with(&intent, changes),
        )?;
    }
    walk.call(
        "finance_execute refusal (not signed yet)",
        "finance_execute",
        json!({"transaction_id": tx}),
    )?;
    let evidence = sha("simulation-evidence");
    let simulation = walk.keys[2].role("simulation", &tx, &hash, &evidence);
    walk.owner_event(
        "simulation_accepted by sim-1",
        json!({"type": "simulation_accepted", "evidence_hash": evidence,
        "simulator_id": "sim-1", "simulator_signature_hex": simulation}),
        None,
    )?;
    let approval = walk.keys[1].sign(
        format!("singularity-finance-approval-v1:{POLICY_ID}:{VERSION}:{tx}:{hash}:{evidence}")
            .as_bytes(),
    );
    walk.owner_event(
        "approval_granted by treasury-owner",
        json!({"type": "approval_granted",
        "approver_id": "treasury-owner", "approval_signature_hex": approval}),
        None,
    )?;
    walk.call(
        "finance_status inside the timelock",
        "finance_status",
        json!({"transaction_id": tx}),
    )?;
    println!(
        "the timelock is {TIMELOCK_SECONDS}s; once timelock_until has passed, run again with:\n  cargo run --example finance_lifecycle -- {}",
        walk.dir.display()
    );
    walk.finish()
}

fn after_timelock(mut walk: Walk) -> Outcome<()> {
    let (tx, hash) = (walk.progress.tx.clone(), walk.progress.hash.clone());
    walk.call(
        "finance_status after timelock (ready)",
        "finance_status",
        json!({"transaction_id": tx}),
    )?;
    let attestation = sha("signer-attestation");
    let signed = walk.keys[3].role("signing", &tx, &hash, &attestation);
    walk.owner_event(
        "signed by signer-1",
        json!({"type": "signed", "signer_attestation_hash": attestation,
        "signer_id": "signer-1", "signer_signature_hex": signed}),
        None,
    )?;
    walk.call(
        "finance_cancel refusal (already signed)",
        "finance_cancel",
        json!({"transaction_id": tx, "request_id": "req-cancel-001"}),
    )?;
    walk.call(
        "finance_execute (executor /usr/bin/false refuses after dispatch)",
        "finance_execute",
        json!({"transaction_id": tx}),
    )?;
    walk.call(
        "finance_status after failed dispatch",
        "finance_status",
        json!({"transaction_id": tx}),
    )?;
    let reconciliation = sha("nothing-was-submitted");
    let not_submitted = walk.keys[5].role("not_submitted", &tx, &hash, &reconciliation);
    walk.owner_event("reconciled_not_submitted by rec-1", json!({"type": "reconciled_not_submitted",
        "reconciliation_hash": reconciliation, "reconciler_id": "rec-1", "reconciler_signature_hex": not_submitted}), None)?;
    let reference = sha("custody-reference");
    let at = ts(Utc::now());
    let receipt = walk.receipt("submitted", &reference, &at)?;
    let submission = walk.keys[4].role("submission", &tx, &hash, &reference);
    walk.owner_event("submitted by exec-1 + WORM receipt", json!({"type": "submitted", "executor_reference_hash": reference,
        "executor_id": "exec-1", "executor_signature_hex": submission, "worm_receipt_file": receipt}), Some(at))?;
    let confirmation = sha("chain-confirmation");
    let at = ts(Utc::now());
    let receipt = walk.receipt("confirmed", &confirmation, &at)?;
    let confirmed = walk.keys[5].role("confirmation", &tx, &hash, &confirmation);
    walk.owner_event("confirmed by rec-1 + WORM receipt", json!({"type": "confirmed", "reconciliation_hash": confirmation,
        "reconciler_id": "rec-1", "reconciler_signature_hex": confirmed, "worm_receipt_file": receipt}), Some(at))?;
    // Each lease is issued strictly after the one before it: lease-2 between
    // lease-1 and now, lease-3 now.
    let intent = intent();
    let next = with(
        &intent,
        json!({"request_id": "req-invoice-002", "amount_minor": 2400}),
    );
    let now = Utc::now();
    walk.lease("lease-2", now - Duration::seconds(10), true)?;
    walk.call(
        "finance_propose refusal (kill switch lease)",
        "finance_propose",
        next.clone(),
    )?;
    walk.lease("lease-3", now, false)?;
    walk.call(
        "finance_propose accepted again under a fresh lease (anchor advances)",
        "finance_propose",
        next,
    )?;
    walk.lease("lease-1", walk.progress.first_lease, false)?;
    walk.call(
        "finance_propose refusal (older lease restored: rollback detected)",
        "finance_propose",
        with(
            &intent,
            json!({"request_id": "req-invoice-003", "amount_minor": 2300}),
        ),
    )?;
    println!("inspect the state store: FIN={}", walk.dir.display());
    walk.finish()
}
