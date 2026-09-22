//! What a tool answers with, and the check every model-visible value passes.

use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::domain::{ChatMessage, ToolCall};

pub(super) const MAX_MEMORIES: usize = 1_000;
pub(super) const MAX_WORKSPACE_FILE_BYTES: u64 = 2 * 1024 * 1024;

pub(super) const MOST_HEALTH: &str = "most_health";
pub(super) const MOST_CREATE_CHAT: &str = "most_create_chat";
pub(super) const MOST_SEND_MESSAGE: &str = "most_send_message";
pub(super) const MEMORY_REMEMBER: &str = "singularity_memory_remember";
pub(super) const MEMORY_RECALL: &str = "singularity_memory_recall";
pub(super) const SELF_SET_PROMPT: &str = "singularity_self_set_prompt";
pub(super) const SELF_ADD_RULE: &str = "singularity_self_add_rule";
pub(super) const SELF_ADD_LEARNING: &str = "singularity_self_add_learning";
pub(super) const SELF_SWITCH_MODEL: &str = "singularity_self_switch_model";
pub(super) const FILE_READ: &str = "singularity_file_read";
pub(super) const FILE_WRITE: &str = "singularity_file_write";
pub(super) const SPAWN_CHILD: &str = "singularity_spawn_child";
pub(super) const MAX_MODEL_OUTPUT_BYTES: usize = 64 * 1024;
pub(super) const MAX_MODEL_OUTPUT_DEPTH: usize = 8;
pub(super) const FORBIDDEN_OUTPUT_KEYS: [&str; 14] = [
    "secret",
    "password",
    "passwd",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "authorization",
    "cookie",
    "private_key",
    "privatekey",
    "credential_path",
    "secret_path",
    "key_path",
];

#[derive(Debug, Clone)]
pub(super) enum ToolOrigin {
    Las,
    MostHealth,
    MostCreateChat,
    MostSendMessage,
    MemoryRemember,
    MemoryRecall,
    SelfSetPrompt,
    SelfAddRule,
    SelfAddLearning,
    SelfSwitchModel,
    FileRead,
    FileWrite,
    SpawnChild,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Success,
    Failed,
    Indeterminate,
}

#[derive(Debug)]
pub(super) struct ModelSafeOutput(Value);

impl ModelSafeOutput {
    fn validate(value: Value) -> Result<Self, &'static str> {
        let encoded = serde_json::to_vec(&value).map_err(|_| "serialization")?;
        if encoded.len() > MAX_MODEL_OUTPUT_BYTES {
            return Err("oversize");
        }
        validate_model_value(&value, 0)?;
        Ok(Self(value))
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct ToolOutcome {
    pub status: ToolStatus,
    pub content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<Uuid>,
}

impl ToolOutcome {
    pub fn message(&self, call: &ToolCall) -> ChatMessage {
        let content = match ModelSafeOutput::validate(self.content.clone()) {
            Ok(safe) => json!({
                "status": &self.status,
                "content": safe.0,
                "error_code": &self.error_code,
                "chat_id": &self.chat_id,
                "message_id": &self.message_id
            }),
            Err(reason) => {
                tracing::warn!(tool = %call.function.name, reason, "tool output rejected by sensitivity policy");
                json!({
                    "status":"failed",
                    "error_code":"sensitive_output_rejected"
                })
            }
        };
        ChatMessage::tool(call, content)
    }
}

pub(super) fn validate_model_value(value: &Value, depth: usize) -> Result<(), &'static str> {
    if depth > MAX_MODEL_OUTPUT_DEPTH {
        return Err("depth");
    }
    match value {
        Value::Object(map) => {
            for (key, nested) in map {
                let normalized = key.to_ascii_lowercase().replace(['-', ' '], "_");
                if FORBIDDEN_OUTPUT_KEYS.iter().any(|forbidden| {
                    normalized == *forbidden || normalized.ends_with(&format!("_{forbidden}"))
                }) {
                    return Err("forbidden_key");
                }
                validate_model_value(nested, depth.saturating_add(1))?;
            }
        }
        Value::Array(items) => {
            for nested in items {
                validate_model_value(nested, depth.saturating_add(1))?;
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            let lower = trimmed.to_ascii_lowercase();
            let contains_raw_path = trimmed
                .split_whitespace()
                .map(|part| {
                    part.trim_matches(|character: char| {
                        matches!(
                            character,
                            ',' | ';' | ':' | '(' | ')' | '[' | ']' | '\'' | '"'
                        )
                    })
                })
                .any(|part| part.starts_with('/') || part.starts_with("file://"));
            if contains_raw_path
                || trimmed.contains('\0')
                || lower.contains("-----begin private key-----")
                || lower.contains("-----begin openssh private key-----")
            {
                return Err("forbidden_value");
            }
            if matches!(trimmed.as_bytes().first(), Some(b'{') | Some(b'[')) {
                let nested: Value = serde_json::from_str(trimmed).map_err(|_| "embedded_json")?;
                validate_model_value(&nested, depth.saturating_add(1))?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
    Ok(())
}
