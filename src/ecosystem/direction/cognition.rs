use super::super::{Shared, model::Issue, store::Store};
use crate::brama::{BramaCompletion, Quote};
use crate::{AppError, BramaClient, ChatMessage, Role};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

#[derive(Deserialize, Serialize)]
struct Input {
    id: String,
    purpose: String,
    model: String,
    max_tokens: u32,
    temperature: f64,
    instruction: String,
    evidence: Value,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Dispatching,
    Answered,
    Settled,
    Indeterminate,
}

#[derive(Deserialize, Serialize)]
struct Call {
    id: String,
    purpose: String,
    state: Phase,
    quote: Quote,
    completion: Option<BramaCompletion>,
    error: Option<String>,
}

fn pending_key(purpose: &str) -> String {
    format!("cognition.pending.{purpose}")
}

pub(crate) fn current(shared: &Shared, purpose: &str) -> Result<Option<String>, AppError> {
    shared.lock()?.store.meta(&pending_key(purpose))
}

/// Consume only a recorded answer, after its caller has saved the result or refusal.
pub(crate) fn finish(
    shared: &Shared,
    purpose: &str,
    error: Option<&AppError>,
) -> Result<(), AppError> {
    let state = shared.lock()?;
    let key = pending_key(purpose);
    let Some(id) = state.store.meta::<String>(&key)? else {
        return Ok(());
    };
    let Some(call) = state.store.get::<Call>("cognition", &id)? else {
        return Ok(());
    };
    if !matches!(call.state, Phase::Settled) {
        return Ok(());
    }
    state.store.put(
        "cognition_disposition",
        &id,
        &json!({"id":id,"purpose":purpose,"error":error.map(ToString::to_string)}),
        None,
        &error.map_or_else(
            || "Consumed recorded model answer".into(),
            |error| format!("Rejected recorded model answer: {error}"),
        ),
    )?;
    state.store.clear_meta(&key)
}

fn settle(store: &Store, call: &mut Call) -> Result<(), AppError> {
    let completion = call.completion.as_ref().ok_or_else(|| {
        AppError::State(format!(
            "model request {} has no retained provider response; its reservation remains allocated",
            call.id
        ))
    })?;
    let amount = call.quote.cost(completion)?;
    let settlement = match store.settle(&call.id, amount) {
        Ok(value) => value,
        Err(error) => {
            store.set_meta("paused", &true)?;
            return Err(error);
        }
    };
    if !matches!(call.state, Phase::Settled) {
        call.state = Phase::Settled;
        store.put(
            "cognition",
            &call.id,
            call,
            None,
            "Settled recorded provider token usage",
        )?;
    }
    settlement.check(&call.id)
}

/// Runs before new work. Known responses settle without inference; missing responses stay unknown.
pub(crate) fn recover(store: &Store) -> Result<(), AppError> {
    for mut call in
        store.in_states::<Call>("cognition", &["dispatching", "answered", "indeterminate"])?
    {
        let result = if call.completion.is_some() {
            settle(store, &mut call)
        } else {
            if !matches!(call.state, Phase::Indeterminate) {
                call.state = Phase::Indeterminate;
                store.put(
                    "cognition",
                    &call.id,
                    &call,
                    None,
                    "Interrupted model dispatch has no retained response",
                )?;
            }
            Err(AppError::State(format!(
                "model request {} ({}) has an unknown provider outcome; no inference was replayed and its reservation remains allocated; last failure: {}",
                call.id,
                call.purpose,
                call.error.as_deref().unwrap_or("no response was retained")
            )))
        };
        let operation = format!("cognition.{}", call.id);
        if let Err(error) = result {
            store.issue(&Issue {
                code: "cognition_recovery".into(),
                operation,
                message: error.to_string(),
                retryable: false,
            })?;
        } else {
            store.clear_issue(&operation)?;
        }
    }
    Ok(())
}

pub async fn ask<T: DeserializeOwned>(
    shared: &Shared,
    client: &BramaClient,
    purpose: &str,
    instruction: &str,
    evidence: Value,
) -> Result<T, AppError> {
    let (input, previous) = {
        let state = shared.lock()?;
        let key = pending_key(purpose);
        let (id, fresh) = match state.store.meta::<String>(&key)? {
            Some(id) => (id, false),
            None => (format!("cognition-{}", uuid::Uuid::new_v4()), true),
        };
        let input = match state.store.get::<Input>("cognition_input", &id)? {
            Some(input) => input,
            None if !fresh => {
                return Err(AppError::State(format!(
                    "model request {id} lost its retained input; no inference was sent"
                )));
            }
            None => {
                let input = Input {
                    id: id.clone(),
                    purpose: purpose.into(),
                    model: state.config.brama_model.clone(),
                    max_tokens: state.config.max_tokens,
                    temperature: state.config.temperature,
                    instruction: instruction.into(),
                    evidence,
                };
                state.store.put(
                    "cognition_input",
                    &id,
                    &input,
                    None,
                    "Retained exact reasoning instructions and evidence before dispatch",
                )?;
                input
            }
        };
        if fresh {
            state.store.set_meta(&key, &id)?;
        }
        if input.purpose != purpose
            || input.model != state.config.brama_model
            || input.max_tokens != state.config.max_tokens
            || input.temperature != state.config.temperature
        {
            return Err(AppError::State(format!(
                "model request {id} conflicts with its retained purpose or generation settings"
            )));
        }
        let previous = state.store.get::<Call>("cognition", &id)?;
        (input, previous)
    };
    let mut call = if let Some(call) = previous {
        if matches!(call.state, Phase::Dispatching | Phase::Indeterminate) {
            return Err(AppError::State(format!(
                "model request {} has an unresolved provider outcome and will not be replayed; last failure: {}",
                call.id,
                call.error.as_deref().unwrap_or("no response was retained")
            )));
        }
        call
    } else {
        let quote = client.quote().await?;
        let mut call = Call {
            id: input.id,
            purpose: input.purpose,
            state: Phase::Dispatching,
            quote,
            completion: None,
            error: None,
        };
        {
            let state = shared.lock()?;
            if state.store.paused()?
                || !state
                    .store
                    .meta::<bool>("las_catalog_ready")?
                    .unwrap_or(false)
            {
                return Err(AppError::State(
                    "model dispatch requires an unpaused portfolio and the signed Las catalog"
                        .into(),
                ));
            }
            if call.quote.upper_usd > state.policy.model_call_reserve_usd {
                return Err(AppError::State(format!(
                    "model call requires {} USD but its delegated reservation is {}; no inference was sent",
                    call.quote.upper_usd, state.policy.model_call_reserve_usd
                )));
            }
            state.store.reserve(
                &call.id,
                state.policy.model_call_reserve_usd,
                state.policy.budget_usd,
            )?;
            state.store.put(
                "cognition",
                &call.id,
                &call,
                None,
                "Reserved model call before dispatch",
            )?;
        }
        let messages = [
            ChatMessage::text(Role::System, input.instruction),
            ChatMessage::text(Role::User, serde_json::to_string(&input.evidence)?),
        ];
        let completion = match client.complete(&messages, &[]).await {
            Ok(completion) => completion,
            Err(error) => {
                call.state = Phase::Indeterminate;
                call.error = Some(error.to_string());
                shared.lock()?.store.put(
                    "cognition",
                    &call.id,
                    &call,
                    None,
                    &format!("Model dispatch failed: {error}; cost remains unresolved"),
                )?;
                return Err(error);
            }
        };
        call.completion = Some(completion);
        call.state = Phase::Answered;
        shared.lock()?.store.put(
            "cognition",
            &call.id,
            &call,
            None,
            "Recorded provider response before settling its cost",
        )?;
        call
    };
    settle(&shared.lock()?.store, &mut call)?;
    let completion = call
        .completion
        .ok_or_else(|| AppError::State("settled model request has no response".into()))?;
    if !completion.tool_calls.is_empty() {
        return Err(AppError::State(
            "portfolio reasoning cannot execute model tool calls".into(),
        ));
    }
    let text = completion
        .content
        .ok_or_else(|| AppError::State("portfolio reasoning returned no JSON".into()))?;
    serde_json::from_str(&text).map_err(|error| {
        AppError::State(format!(
            "{purpose}: model response is not the required JSON document: {error}"
        ))
    })
}
