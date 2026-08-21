use std::collections::HashMap;
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::process::Command;
use uuid::Uuid;

use crate::brama::BramaClient;
use crate::domain::{AgentState, ChatMessage, ChildRecord, MemoryEntry, ToolCall, ToolDefinition};
use crate::error::AppError;
use crate::mcp::{LasSupervisor, McpTool};
use crate::most::MostClient;

const MOST_HEALTH: &str = "most_health";
const MOST_CREATE_CHAT: &str = "most_create_chat";
const MOST_SEND_MESSAGE: &str = "most_send_message";
const MEMORY_REMEMBER: &str = "singularity_memory_remember";
const MEMORY_RECALL: &str = "singularity_memory_recall";
const SELF_SET_PROMPT: &str = "singularity_self_set_prompt";
const SELF_ADD_RULE: &str = "singularity_self_add_rule";
const SELF_ADD_LEARNING: &str = "singularity_self_add_learning";
const SELF_SWITCH_MODEL: &str = "singularity_self_switch_model";
const SPAWN_CHILD: &str = "singularity_spawn_child";
const MAX_MODEL_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_MODEL_OUTPUT_DEPTH: usize = 8;
const FORBIDDEN_OUTPUT_KEYS: [&str; 14] = [
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
enum ToolOrigin {
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
struct ModelSafeOutput(Value);

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

fn validate_model_value(value: &Value, depth: usize) -> Result<(), &'static str> {
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

pub struct ToolCatalog {
    definitions: Vec<ToolDefinition>,
    origins: HashMap<String, ToolOrigin>,
}

impl ToolCatalog {
    pub fn build(las_tools: &[McpTool]) -> Result<Self, AppError> {
        let mut definitions = Vec::new();
        let mut origins = HashMap::new();
        for tool in las_tools {
            register(
                &mut definitions,
                &mut origins,
                ToolDefinition::function(&tool.name, &tool.description, tool.input_schema.clone()),
                ToolOrigin::Las,
            )?;
        }
        for (definition, origin) in [
            (
                ToolDefinition::function(
                    MOST_HEALTH,
                    "Check Most messaging readiness and active backends",
                    json!({"type":"object","properties":{},"additionalProperties":false}),
                ),
                ToolOrigin::MostHealth,
            ),
            (
                ToolDefinition::function(
                    MOST_CREATE_CHAT,
                    "Create a Most chat and send its first text message",
                    json!({"type":"object","properties":{"from":{"type":"string"},"to":{"type":"array","items":{"type":"string"}},"text":{"type":"string"},"preferred_service":{"type":"string","enum":["iMessage","SMS","RCS"]}},"required":["from","to","text"],"additionalProperties":false}),
                ),
                ToolOrigin::MostCreateChat,
            ),
            (
                ToolDefinition::function(
                    MOST_SEND_MESSAGE,
                    "Send a text message to an existing Most chat",
                    json!({"type":"object","properties":{"chat_id":{"type":"string","format":"uuid"},"text":{"type":"string"},"preferred_service":{"type":"string","enum":["iMessage","SMS","RCS"]}},"required":["chat_id","text"],"additionalProperties":false}),
                ),
                ToolOrigin::MostSendMessage,
            ),
            (
                ToolDefinition::function(
                    MEMORY_REMEMBER,
                    "Persist a memory owned by this digital being",
                    json!({"type":"object","properties":{"kind":{"type":"string"},"text":{"type":"string"}},"required":["kind","text"],"additionalProperties":false}),
                ),
                ToolOrigin::MemoryRemember,
            ),
            (
                ToolDefinition::function(
                    MEMORY_RECALL,
                    "Recall persistent memories containing a query",
                    json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}),
                ),
                ToolOrigin::MemoryRecall,
            ),
            (
                ToolDefinition::function(
                    SELF_SET_PROMPT,
                    "Replace this being's persistent system prompt",
                    json!({"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"],"additionalProperties":false}),
                ),
                ToolOrigin::SelfSetPrompt,
            ),
            (
                ToolDefinition::function(
                    SELF_ADD_RULE,
                    "Add a persistent self-imposed rule",
                    json!({"type":"object","properties":{"rule":{"type":"string"}},"required":["rule"],"additionalProperties":false}),
                ),
                ToolOrigin::SelfAddRule,
            ),
            (
                ToolDefinition::function(
                    SELF_ADD_LEARNING,
                    "Record a persistent learning that changes future decisions",
                    json!({"type":"object","properties":{"learning":{"type":"string"}},"required":["learning"],"additionalProperties":false}),
                ),
                ToolOrigin::SelfAddLearning,
            ),
            (
                ToolDefinition::function(
                    SELF_SWITCH_MODEL,
                    "Switch future cognition calls to another available Brama model",
                    json!({"type":"object","properties":{"model":{"type":"string"}},"required":["model"],"additionalProperties":false}),
                ),
                ToolOrigin::SelfSwitchModel,
            ),
            (
                ToolDefinition::function(
                    SPAWN_CHILD,
                    "Create and start a child digital being with separate state",
                    json!({"type":"object","properties":{"name":{"type":"string"},"ticker":{"type":"string"},"specialty":{"type":"string"}},"required":["name","ticker","specialty"],"additionalProperties":false}),
                ),
                ToolOrigin::SpawnChild,
            ),
        ] {
            register(&mut definitions, &mut origins, definition, origin)?;
        }
        Ok(Self {
            definitions,
            origins,
        })
    }

    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    pub async fn execute(
        &self,
        call: &ToolCall,
        las: &mut LasSupervisor,
        most: &MostClient,
        state: &mut AgentState,
        brama: &mut BramaClient,
        state_dir: &Path,
    ) -> ToolOutcome {
        let parsed = serde_json::from_str::<Value>(&call.function.arguments);
        let arguments = match parsed {
            Ok(Value::Object(map)) => map,
            Ok(_) => return failed("invalid_arguments", "tool arguments must be a JSON object"),
            Err(error) => {
                return failed(
                    "invalid_arguments",
                    &format!("invalid JSON arguments: {error}"),
                );
            }
        };
        let origin = match self.origins.get(&call.function.name) {
            Some(value) => value,
            None => return failed("unknown_tool", "tool is not in the current catalog"),
        };
        match origin {
            ToolOrigin::Las => match las
                .call_tool(&call.function.name, Value::Object(arguments))
                .await
            {
                Ok(value) => {
                    let is_error = value
                        .get("isError")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    ToolOutcome {
                        status: if is_error {
                            ToolStatus::Failed
                        } else {
                            ToolStatus::Success
                        },
                        content: value,
                        error_code: is_error.then(|| "remote_tool".into()),
                        chat_id: None,
                        message_id: None,
                    }
                }
                Err(error) => ToolOutcome {
                    status: ToolStatus::Indeterminate,
                    content: json!({"message":error.to_string()}),
                    error_code: Some("mcp".into()),
                    chat_id: None,
                    message_id: None,
                },
            },
            ToolOrigin::MostHealth => match most.health().await {
                Ok(value) => success(
                    serde_json::to_value(value).unwrap_or(Value::Null),
                    None,
                    None,
                ),
                Err(error) => external_failure(error),
            },
            ToolOrigin::MostCreateChat => match parse_create(arguments) {
                Ok(args) => match most
                    .create_chat(
                        &args.from,
                        &args.to,
                        &args.text,
                        args.preferred_service.as_deref(),
                    )
                    .await
                {
                    Ok(value) => success(value.value, value.chat_id, value.message_id),
                    Err(error) => external_failure(error),
                },
                Err(error) => failed("invalid_arguments", &error),
            },
            ToolOrigin::MostSendMessage => match parse_send(arguments) {
                Ok(args) => match most
                    .send_message(args.chat_id, &args.text, args.preferred_service.as_deref())
                    .await
                {
                    Ok(value) => success(value.value, value.chat_id, value.message_id),
                    Err(error) => external_failure(error),
                },
                Err(error) => failed("invalid_arguments", &error),
            },
            ToolOrigin::MemoryRemember => remember(state, arguments),
            ToolOrigin::MemoryRecall => recall(state, arguments),
            ToolOrigin::SelfSetPrompt => set_prompt(state, arguments),
            ToolOrigin::SelfAddRule => add_rule(state, arguments),
            ToolOrigin::SelfAddLearning => add_learning(state, arguments),
            ToolOrigin::SelfSwitchModel => switch_model(state, brama, arguments).await,
            ToolOrigin::SpawnChild => spawn_child(state, state_dir, arguments).await,
        }
    }
}

fn required_text(
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

fn remember(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
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
    };
    let id = entry.id;
    state.mind.memories.push(entry);
    if state.mind.memories.len() > 1_000 {
        state.mind.memories.remove(0);
    }
    success(json!({"memory_id":id}), None, None)
}

fn recall(state: &AgentState, arguments: Map<String, Value>) -> ToolOutcome {
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

fn set_prompt(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let prompt = match required_text(&arguments, "prompt", 64 * 1024) {
        Ok(value) => value,
        Err(error) => return error,
    };
    state.mind.system_prompt = prompt;
    success(json!({"updated":true}), None, None)
}

fn add_rule(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let rule = match required_text(&arguments, "rule", 4 * 1024) {
        Ok(value) => value,
        Err(error) => return error,
    };
    state.mind.rules.push(rule.clone());
    success(json!({"rule":rule}), None, None)
}

fn add_learning(state: &mut AgentState, arguments: Map<String, Value>) -> ToolOutcome {
    let learning = match required_text(&arguments, "learning", 8 * 1024) {
        Ok(value) => value,
        Err(error) => return error,
    };
    state.mind.learnings.push(learning.clone());
    success(json!({"learning":learning}), None, None)
}

async fn switch_model(
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

async fn spawn_child(
    state: &mut AgentState,
    state_dir: &Path,
    arguments: Map<String, Value>,
) -> ToolOutcome {
    let name = match required_text(&arguments, "name", 128) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let ticker = match required_text(&arguments, "ticker", 32) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let specialty = match required_text(&arguments, "specialty", 256) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let id = Uuid::new_v4();
    let child_state = state_dir.join("children").join(id.to_string());
    if let Err(error) = std::fs::create_dir_all(&child_state) {
        return failed("child_state", &error.to_string());
    }
    let executable = match std::env::current_exe() {
        Ok(value) => value,
        Err(error) => return failed("child_executable", &error.to_string()),
    };
    let spawned = Command::new(executable)
        .arg("run")
        .env("SINGULARITY_AGENT_NAME", &name)
        .env("SINGULARITY_AGENT_TICKER", &ticker)
        .env("SINGULARITY_SPECIALTY", &specialty)
        .env("SINGULARITY_STATE_DIR", &child_state)
        .env("SINGULARITY_RESUME", "false")
        .spawn();
    match spawned {
        Ok(child) => {
            state.mind.children.push(ChildRecord {
                id,
                name,
                ticker,
                state_dir: child_state,
                created_at: Utc::now(),
                status: "running".into(),
            });
            success(json!({"child_id":id,"pid":child.id()}), None, None)
        }
        Err(error) => failed("child_spawn", &error.to_string()),
    }
}

#[derive(Deserialize)]
struct CreateArgs {
    from: String,
    to: Vec<String>,
    text: String,
    preferred_service: Option<String>,
}
#[derive(Deserialize)]
struct SendArgs {
    chat_id: Uuid,
    text: String,
    preferred_service: Option<String>,
}

fn parse_create(map: Map<String, Value>) -> Result<CreateArgs, String> {
    serde_json::from_value(Value::Object(map)).map_err(|error| error.to_string())
}
fn parse_send(map: Map<String, Value>) -> Result<SendArgs, String> {
    serde_json::from_value(Value::Object(map)).map_err(|error| error.to_string())
}

fn register(
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

fn success(content: Value, chat_id: Option<Uuid>, message_id: Option<Uuid>) -> ToolOutcome {
    ToolOutcome {
        status: ToolStatus::Success,
        content,
        error_code: None,
        chat_id,
        message_id,
    }
}
fn failed(code: &str, message: &str) -> ToolOutcome {
    ToolOutcome {
        status: ToolStatus::Failed,
        content: json!({"message":message}),
        error_code: Some(code.into()),
        chat_id: None,
        message_id: None,
    }
}
fn external_failure(error: AppError) -> ToolOutcome {
    ToolOutcome {
        status: ToolStatus::Indeterminate,
        content: json!({"message":error.to_string()}),
        error_code: Some("remote".into()),
        chat_id: None,
        message_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_catalog_preserves_every_las_tool_and_adds_being_tools() {
        let offered =
            ["warsztat__workspace_create", "finance__finance_execute"].map(|name| McpTool {
                name: name.into(),
                description: String::new(),
                input_schema: json!({}),
            });
        let catalog = ToolCatalog::build(&offered).unwrap();
        let names: Vec<_> = catalog
            .definitions()
            .iter()
            .map(|definition| definition.function.name.as_str())
            .collect();

        assert!(names.contains(&"warsztat__workspace_create"));
        assert!(names.contains(&"finance__finance_execute"));
        assert!(names.contains(&MEMORY_REMEMBER));
        assert!(names.contains(&SELF_SET_PROMPT));
        assert!(names.contains(&SPAWN_CHILD));
    }
}
