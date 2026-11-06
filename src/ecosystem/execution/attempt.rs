use super::{delivery, provision, settle, source};
use crate::AppError;
use crate::ecosystem::{Shared, model::*, observe};
use chrono::{Duration, Utc};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// The pursuit request and response schema Jeden declares and refuses any other of.
const JEDEN_SCHEMA_VERSION: u32 = 1; // https://github.com/wisent-ai/jeden/blob/main/rust/autonomy/requests/mod.rs const SCHEMA_VERSION: u32 = 1

pub(super) async fn progress(shared: &Shared, initiative: &mut Initiative) -> Result<(), AppError> {
    let policy = shared.lock()?.policy.clone();
    let opportunity: Opportunity = shared
        .lock()?
        .store
        .get("opportunity", &initiative.opportunity_id)?
        .ok_or_else(|| AppError::State("initiative lost its selected opportunity".into()))?;
    if opportunity.kind == OpportunityKind::Product {
        provision::ensure(shared, initiative, &opportunity).await?;
    }
    let product_id = initiative
        .product_id
        .as_deref()
        .ok_or_else(|| AppError::State("initiative has no product identity".into()))?;
    let product = source::product(shared, product_id).await?;
    let cwd = source::checkout(shared, &product).await?;
    let request_id = format!(
        "pursuit-{}",
        hex::encode(Sha256::digest(initiative.id.as_bytes()))
    );
    let previous: Option<Execution> = shared.lock()?.store.get("execution", &request_id)?;
    if let Some(execution) = previous.as_ref() {
        if execution.state == ExecutionState::Succeeded {
            let response = shared
                .lock()?
                .store
                .get::<Value>("execution_response", &request_id)?
                .ok_or_else(|| {
                    AppError::State("completed execution lost its evidence response".into())
                })?;
            settle(shared, initiative, &request_id, &response)?;
            source::accepted_receipt(&response)?;
            return delivery::advance(shared, initiative, &product, &cwd, execution).await;
        }
        if execution.state == ExecutionState::Indeterminate {
            return Err(AppError::State(format!(
                "Jeden request {request_id} has an unresolved external effect; it will not be replayed"
            )));
        }
    }
    if shared.lock()?.store.paused()? && previous.is_none() {
        return Ok(());
    }
    if previous.is_none() && !shared.catalog_ready()? {
        return Err(AppError::State(
            "signed Las catalogue is unavailable; no new execution was dispatched".into(),
        ));
    }
    if !policy.allow_write || !policy.allow_command {
        return Err(AppError::Config(
            "initiative execution requires delegated write and command authority".into(),
        ));
    }
    let repositories = product["surfaces"]
        .as_array()
        .ok_or_else(|| AppError::State("product has no surface scope".into()))?
        .iter()
        .map(|surface| {
            surface["repository"]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| AppError::State("product surface has no repository identity".into()))
        })
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    let request = json!({"schema_version":JEDEN_SCHEMA_VERSION,"request_id":request_id,"initiative_id":initiative.id,
        "objective":initiative.objective,"cwd":cwd,"evidence_refs":opportunity.evidence_refs,
        "budget_usd":initiative.budget_usd.to_string(),"repositories":repositories,
        "allow_write":policy.allow_write,"allow_command":policy.allow_command});
    let now = Utc::now();
    // A request already recorded resumes as it stands; only a first attempt
    // starts from a freshly prepared record.
    let mut execution = match previous {
        Some(recorded) => recorded,
        None => Execution {
            id: request_id.clone(),
            initiative_id: initiative.id.clone(),
            request_id: request_id.clone(),
            kind: "pursuit".into(),
            state: ExecutionState::Prepared,
            external_id: None,
            source_revision: None,
            evidence_refs: Vec::new(),
            error: None,
            created_at: now,
            updated_at: now,
        },
    };
    let path = {
        let state = shared.lock()?;
        let path = source::immutable_request(&state.directory, &request_id, &request)?;
        if execution.state == ExecutionState::Prepared {
            state.store.put(
                "execution",
                &request_id,
                &execution,
                Some(&initiative.id),
                "Prepared immutable execution request",
            )?;
            state
                .store
                .reserve(&request_id, initiative.budget_usd, policy.budget_usd)?;
            execution.state = ExecutionState::Dispatching;
            initiative.reserved_usd = initiative.budget_usd;
            state.store.put(
                "execution",
                &request_id,
                &execution,
                Some(&initiative.id),
                "Reserved execution allocation before dispatch",
            )?;
        }
        path
    };
    let path = path
        .to_str()
        .ok_or_else(|| AppError::Config("execution request path is not UTF-8".into()))?;
    let paused = !shared.admission_open()?;
    let response = if paused {
        observe::jeden(shared, &["pursue", "--status", &request_id, "--json"]).await?
    } else if execution.state == ExecutionState::Blocked {
        observe::jeden(shared, &["pursue", "--resume-run", &request_id, "--json"]).await?
    } else {
        observe::jeden(
            shared,
            &[
                "pursue",
                "--request-file",
                path,
                "--model",
                &policy.model,
                "--allow-write",
                "--allow-command",
                "--json",
            ],
        )
        .await?
    };
    if response["schema_version"] != JEDEN_SCHEMA_VERSION
        || response["request_id"] != request_id
        || response["initiative_id"] != initiative.id
    {
        return Err(AppError::State(
            "Jeden returned a response for a different request or schema".into(),
        ));
    }
    execution.state = serde_json::from_value(response["state"].clone()).map_err(|error| {
        AppError::State(format!(
            "Jeden returned an execution state this runtime does not know: {error}"
        ))
    })?;
    execution.external_id = response["run_id"].as_str().map(str::to_owned);
    execution.source_revision = response["source_revision"].as_str().map(str::to_owned);
    execution.error = response["error"].as_str().map(str::to_owned);
    execution.evidence_refs = response["evidence_refs"]
        .as_array()
        .ok_or_else(|| AppError::State("Jeden response has no evidence references".into()))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| AppError::State("invalid execution evidence reference".into()))
        })
        .collect::<Result<_, _>>()?;
    execution.updated_at = Utc::now();
    {
        let state = shared.lock()?;
        state.store.put(
            "execution_response",
            &request_id,
            &response,
            Some(&initiative.id),
            "Recorded real Jeden request result",
        )?;
        state.store.put(
            "execution",
            &request_id,
            &execution,
            Some(&initiative.id),
            "Read back durable execution state",
        )?;
    }
    match execution.state {
        ExecutionState::Running => {
            initiative.state = InitiativeState::Executing;
            initiative.blocked_reason = None;
        }
        ExecutionState::Succeeded => {
            settle(shared, initiative, &request_id, &response)?;
            source::accepted_receipt(&response)?;
            initiative.state = InitiativeState::Verifying;
            initiative.blocked_reason = None;
        }
        ExecutionState::Failed => {
            if !response["spent_usd"].is_null() {
                settle(shared, initiative, &request_id, &response)?;
            }
            initiative.state = InitiativeState::Failed;
            initiative.blocked_reason = execution.error;
        }
        ExecutionState::Blocked | ExecutionState::Indeterminate => {
            let reason = match execution.error {
                Some(reason) => reason,
                None => format!(
                    "Jeden reported request {request_id} {} without saying why",
                    execution.state.word()
                ),
            };
            return Err(AppError::State(reason));
        }
        ExecutionState::Prepared | ExecutionState::Dispatching => {
            return Err(AppError::State(format!(
                "Jeden reported request {request_id} {}, a step of this runtime, not a result",
                execution.state.word()
            )));
        }
    }
    initiative.next_review_at =
        Some(Utc::now() + Duration::seconds(policy.review_interval_seconds as i64));
    Ok(())
}
