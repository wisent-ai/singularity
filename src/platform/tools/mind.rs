//! The mind tools: remembering, recalling, and the agent rewriting its own prompt and rules.

use chrono::Utc;
use serde_json::{json, Map, Value};
use uuid::Uuid;

use super::*;
use crate::brama::BramaClient;
use crate::domain::{AgentState, MemoryEntry};

pub(super) fn required_text(
    arguments: &Map<String, Value>,
    key: &str,
    max_bytes: usize,
) -> Result<String, ToolOutcome> {
    let value = arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && value.len() <= max_bytes)
        .filter(|value| !value.chars().any(char::is_control))
        .map(str::to_owned);
    value.ok_or_else(|| failed("invalid_arguments", &format!("{key} is invalid")))
}

pub(super) fn remember(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let kind = match required_text(&arguments, "kind", 64) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let text = match required_text(&arguments, "text", 16 * 1024) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let entry = MemoryEntry {
        id: Uuid::new_v4(),
        kind,
        text,
        created_at: Utc::now(),
        sources: Vec::new(),
    };
    let id = entry.id;
    state.mind.memories.push(entry);
    if state.mind.memories.len() > MAX_MEMORIES {
        state.mind.memories.remove(0);
    }
    success(json!({"memory_id":id}), None, None)
}

pub(super) fn recall(state: &AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let query = match required_text(&arguments, "query", 1_024) {
        Ok(value) => value.to_ascii_lowercase(),
        Err(error) => return error,
    };
    let memories = state
        .mind
        .memories
        .iter()
        .rev()
        .filter(|entry| {
            entry.kind.to_ascii_lowercase().contains(&query)
                || entry.text.to_ascii_lowercase().contains(&query)
        })
        .take(50)
        .collect::<Vec<_>>();
    success(json!({"memories":memories}), None, None)
}

pub(super) fn set_prompt(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let prompt = match required_text(&arguments, "prompt", 64 * 1024) {
        Ok(value) => value,
        Err(error) => return error,
    };
    state.mind.system_prompt = prompt;
    success(json!({"updated":true}), None, None)
}

pub(super) fn add_rule(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let rule = match required_text(&arguments, "rule", 4 * 1024) {
        Ok(value) => value,
        Err(error) => return error,
    };
    state.mind.rules.push(rule.clone());
    success(json!({"rule":rule}), None, None)
}

pub(super) fn add_learning(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let learning = match required_text(&arguments, "learning", 8 * 1024) {
        Ok(value) => value,
        Err(error) => return error,
    };
    state.mind.learnings.push(learning.clone());
    success(json!({"learning":learning}), None, None)
}

pub(super) async fn switch_model(
    state: &mut AgentState,
    brama: &mut BramaClient,
    arguments: Map<String, Value>,
) -> ToolOutcome {
    let model = match required_text(&arguments, "model", 256) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let selector = model == "any"
        || model == "any-vision-capable"
        || model == "best"
        || model.starts_with("task:");
    match brama.models().await {
        Ok(models) if selector || models.iter().any(|available| available == &model) => {
            state.mind.current_model = model.clone();
            brama.set_model(model.clone());
            success(json!({"model":model}), None, None)
        }
        Ok(_) => failed("model_unavailable", "Brama does not advertise that model"),
        Err(error) => external_failure(error),
    }
}
