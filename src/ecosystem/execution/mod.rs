mod attempt;
mod delivery;
mod provision;
mod source;
use super::{
    Shared,
    model::{Initiative, InitiativeState},
};
use crate::AppError;
use chrono::{Duration, Utc};
use rust_decimal::Decimal;
use serde_json::Value;

/// Billing recovery does not depend on a checkout, catalog reader or deployment service.
pub(super) fn recover(shared: &Shared) -> Result<(), AppError> {
    let responses = shared
        .lock()?
        .store
        .in_states::<Value>("execution_response", &["succeeded", "failed"])?;
    for response in responses {
        let id = response["request_id"].as_str().ok_or_else(|| {
            AppError::State("retained execution response has no request identity".into())
        })?;
        let operation = format!("accounting.{id}");
        match recover_response(shared, id, &response) {
            Ok(()) => shared.lock()?.store.clear_issue(&operation)?,
            Err(error) => shared.failure(&operation, &error)?,
        }
    }
    Ok(())
}

fn recover_response(shared: &Shared, id: &str, response: &Value) -> Result<(), AppError> {
    let initiative_id = response["initiative_id"].as_str().ok_or_else(|| {
        AppError::State("retained execution response has no initiative identity".into())
    })?;
    let mut initiative: Initiative = shared
        .lock()?
        .store
        .get("initiative", initiative_id)?
        .ok_or_else(|| {
            AppError::State(format!("execution {id} lost initiative {initiative_id}"))
        })?;
    let previous = (initiative.spent_usd, initiative.reserved_usd);
    let cost: Option<Decimal> = response["spent_usd"]
        .as_str()
        .map(str::parse)
        .transpose()
        .map_err(|error| AppError::State(format!("recorded execution cost: {error}")))?;
    if let Some(cost) = cost {
        if previous == (cost, Decimal::ZERO) && shared.lock()?.store.settled_cost(id)? == Some(cost)
        {
            return Ok(());
        }
    }
    let result = settle(shared, &mut initiative, id, response);
    if previous != (initiative.spent_usd, initiative.reserved_usd) {
        shared.lock()?.store.put(
            "initiative",
            initiative_id,
            &initiative,
            Some(initiative_id),
            "Recovered execution accounting from the retained terminal response",
        )?;
    }
    result
}

pub fn eligible(shared: &Shared) -> Result<Vec<String>, AppError> {
    let state = shared.lock()?;
    let paused = state.store.paused()?;
    Ok(state
        .store
        .list::<Initiative>("initiative")?
        .into_iter()
        .filter(|i| {
            let due = i.next_review_at.is_none_or(|at| at <= Utc::now());
            due && match i.state {
                InitiativeState::Executing
                | InitiativeState::Verifying
                | InitiativeState::Releasing
                | InitiativeState::Blocked => true,
                InitiativeState::Selected => !paused,
                _ => false,
            }
        })
        .take(state.policy.max_active)
        .map(|i| i.id)
        .collect())
}

pub async fn advance(shared: &Shared, id: &str) -> Result<(), AppError> {
    let mut initiative: Initiative = shared
        .lock()?
        .store
        .get("initiative", id)?
        .ok_or_else(|| AppError::State(format!("unknown initiative {id}")))?;
    let result = attempt::progress(shared, &mut initiative).await;
    let state = shared.lock()?;
    if let Err(error) = &result {
        initiative.state = InitiativeState::Blocked;
        initiative.blocked_reason = Some(error.to_string());
        initiative.next_review_at =
            Some(Utc::now() + Duration::seconds(state.policy.review_interval_seconds as i64));
    }
    initiative.updated_at = Utc::now();
    state.store.put(
        "initiative",
        id,
        &initiative,
        Some(id),
        if result.is_ok() {
            "Observed initiative progress"
        } else {
            "Initiative blocked; independent work remains eligible"
        },
    )?;
    if result.is_ok() {
        state.store.clear_issue(&format!("execute.{id}"))?;
    }
    result
}

fn settle(
    shared: &Shared,
    initiative: &mut Initiative,
    request_id: &str,
    response: &Value,
) -> Result<(), AppError> {
    let cost: Decimal = response["spent_usd"]
        .as_str()
        .ok_or_else(|| {
            AppError::State(
                "Jeden execution cost is unknown; its allocation remains reserved".into(),
            )
        })?
        .parse()
        .map_err(|e| AppError::State(format!("invalid execution cost: {e}")))?;
    let state = shared.lock()?;
    let settlement = match state.store.settle(request_id, cost) {
        Ok(settlement) => settlement,
        Err(error) => {
            state.store.set_meta("paused", &true)?;
            return Err(error);
        }
    };
    initiative.spent_usd = cost;
    initiative.reserved_usd = Decimal::ZERO;
    settlement.check(request_id)?;
    Ok(())
}
