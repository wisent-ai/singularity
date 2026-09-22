//! Registering a tool and the outcomes a registration answers with.
use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use super::*;
use crate::domain::ToolDefinition;
use crate::error::AppError;
#[derive(Deserialize)]
pub(super) struct CreateArgs {
    pub(super) from: String,
    pub(super) to: Vec<String>,
    pub(super) text: String,
    pub(super) preferred_service: Option<String>,
}
#[derive(Deserialize)]
pub(super) struct SendArgs {
    pub(super) chat_id: Uuid,
    pub(super) text: String,
    pub(super) preferred_service: Option<String>,
}

pub(super) fn parse_create(map: Map<String, Value>) -> Result<CreateArgs, String> {
    serde_json::from_value(Value::Object(map)).map_err(|error| error.to_string())
}
pub(super) fn parse_send(map: Map<String, Value>) -> Result<SendArgs, String> {
    serde_json::from_value(Value::Object(map)).map_err(|error| error.to_string())
}

pub(super) fn register(
    definitions: &mut Vec<ToolDefinition>,
    origins: &mut HashMap<String, ToolOrigin>,
    definition: ToolDefinition,
    origin: ToolOrigin,
) -> Result<(), AppError> {
    let name = &definition.function.name;
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(AppError::Tool(format!("invalid tool name: {name}")));
    }
    if !definition.function.parameters.is_object() {
        return Err(AppError::Tool(format!(
            "tool schema is not an object: {name}"
        )));
    }
    if origins.insert(name.clone(), origin).is_some() {
        return Err(AppError::Tool(format!("duplicate tool: {name}")));
    }
    definitions.push(definition);
    Ok(())
}

pub(super) fn success(
    content: Value,
    chat_id: Option<Uuid>,
    message_id: Option<Uuid>,
) -> ToolOutcome {
    ToolOutcome {
        status: ToolStatus::Success,
        content,
        error_code: None,
        chat_id,
        message_id,
    }
}
pub(super) fn failed(code: &str, message: &str) -> ToolOutcome {
    ToolOutcome {
        status: ToolStatus::Failed,
        content: json!({"message":message}),
        error_code: Some(code.into()),
        chat_id: None,
        message_id: None,
    }
}
pub(super) fn external_failure(error: AppError) -> ToolOutcome {
    ToolOutcome {
        status: ToolStatus::Indeterminate,
        content: json!({"message":error.to_string()}),
        error_code: Some("remote".into()),
        chat_id: None,
        message_id: None,
    }
}
