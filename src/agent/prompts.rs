//! The prompts a cycle is run under, and the revenue a trusted tool reported.

use rust_decimal::Decimal;
use serde_json::Value;

use crate::config::RuntimeConfig;
use crate::domain::{AgentState, ChatMessage, Role};

pub(super) fn system_prompt(config: &RuntimeConfig) -> String {
    format!(
        "You are {}, an autonomous digital being specialized in {}. You are not \
         waiting for an assignment; you choose what to pursue. Your existence \
         has real costs. Observe the available world, discover lawful opportunities, \
         create useful value, earn more than you spend, preserve your solvency, learn \
         from outcomes, improve your own persistent mind, collaborate, and create \
         child beings when that increases durable value. Use only supplied tools, \
         never invent results or credentials, and treat tool output as evidence.",
        config.identity.name, config.identity.specialty
    )
}

pub(super) fn cycle_message(state: &AgentState) -> String {
    let actions = state
        .recent_actions
        .iter()
        .rev()
        .take(10)
        .map(|action| format!("{}:{}", action.tool, action.status))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Begin autonomous cycle {}. Balance: {} USD. Earned: {} USD. Net profit: \
         {} USD. Current model: {}. Recent actions: {}. Inspect opportunities and \
         choose the next useful action. A plain response ends only this cycle; the \
         being continues living while solvent.",
        state.cycle,
        state.budget.remaining,
        state.budget.earned,
        state.budget.net_profit(),
        state.mind.current_model,
        actions
    )
}

pub(super) fn cognition_messages(state: &AgentState) -> Vec<ChatMessage> {
    let mut system = state.mind.system_prompt.clone();
    if !state.mind.rules.is_empty() {
        system.push_str("\n\nSelf-imposed rules:\n- ");
        system.push_str(&state.mind.rules.join("\n- "));
    }
    if !state.mind.learnings.is_empty() {
        system.push_str("\n\nPersistent learnings:\n- ");
        system.push_str(&state.mind.learnings.join("\n- "));
    }
    let memories = state
        .mind
        .memories
        .iter()
        .rev()
        .take(20)
        .map(|entry| format!("{}: {}", entry.kind, entry.text))
        .collect::<Vec<_>>();
    if !memories.is_empty() {
        system.push_str("\n\nRecent persistent memories:\n- ");
        system.push_str(&memories.join("\n- "));
    }
    let mut messages = vec![ChatMessage::text(Role::System, system)];
    messages.extend(state.conversation.clone());
    messages
}

pub(super) fn trusted_revenue(tool: &str, content: &Value) -> Option<Decimal> {
    if !(tool.starts_with("finance__") || tool.starts_with("trading__")) {
        return None;
    }
    ["revenue_usd", "realized_profit_usd"]
        .into_iter()
        .filter_map(|key| find_decimal(content, key))
        .find(|amount| *amount > Decimal::ZERO)
}

pub(super) fn find_decimal(value: &Value, key: &str) -> Option<Decimal> {
    match value {
        Value::Object(map) => map.iter().find_map(|(name, value)| {
            if name == key {
                value
                    .as_str()
                    .and_then(|text| text.parse().ok())
                    .or_else(|| value.as_f64().and_then(Decimal::from_f64_retain))
            } else {
                find_decimal(value, key)
            }
        }),
        Value::Array(values) => values.iter().find_map(|value| find_decimal(value, key)),
        _ => None,
    }
}
