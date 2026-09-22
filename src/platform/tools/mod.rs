use std::collections::HashMap;
use std::path::Path;

use serde_json::{Value, json};

use crate::brama::BramaClient;
use crate::domain::{AgentState, ToolCall, ToolDefinition};
use crate::error::AppError;
use crate::mcp::{LasSupervisor, McpTool};
use crate::most::MostClient;

/// The mind keeps at most a thousand memories; a workspace file read or written
/// through a tool is at most 2 MiB.
pub struct ToolCatalog {
    definitions: Vec<ToolDefinition>,
    origins: HashMap<String, ToolOrigin>,
}

impl ToolCatalog {
    pub fn build(las_tools: &[McpTool], most_enabled: bool) -> Result<Self, AppError> {
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
            (
                ToolDefinition::function(
                    FILE_READ,
                    "Read one UTF-8 file inside the configured workspace",
                    json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}),
                ),
                ToolOrigin::FileRead,
            ),
            (
                ToolDefinition::function(
                    FILE_WRITE,
                    "Atomically create or replace one UTF-8 file inside the configured workspace",
                    json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false}),
                ),
                ToolOrigin::FileWrite,
            ),
        ] {
            if !most_enabled
                && matches!(
                    origin,
                    ToolOrigin::MostHealth
                        | ToolOrigin::MostCreateChat
                        | ToolOrigin::MostSendMessage
                )
            {
                continue;
            }
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
        most: Option<&MostClient>,
        state: &mut AgentState,
        brama: &mut BramaClient,
        workspace: &Path,
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
                    // The MCP contract writes `isError` only on a failed call; a flag that is
                    // present but not a boolean is a malformed answer, not a successful one.
                    let is_error = match value.get("isError") {
                        Some(flag) => match flag.as_bool() {
                            Some(flag) => flag,
                            None => {
                                return failed(
                                    "remote_tool_malformed",
                                    &format!("isError is not a boolean: {flag}"),
                                );
                            }
                        },
                        None => false,
                    };
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
            ToolOrigin::MostHealth => match most {
                Some(most) => match most.health().await {
                    Ok(value) => match serde_json::to_value(value) {
                        Ok(value) => success(value, None, None),
                        Err(error) => failed(
                            "most_malformed",
                            &format!("Most health is not serializable: {error}"),
                        ),
                    },
                    Err(error) => external_failure(error),
                },
                None => failed("most_unavailable", "Most credential is not configured"),
            },
            ToolOrigin::MostCreateChat => match (most, parse_create(arguments)) {
                (Some(most), Ok(args)) => match most
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
                (None, _) => failed("most_unavailable", "Most credential is not configured"),
                (_, Err(error)) => failed("invalid_arguments", &error),
            },
            ToolOrigin::MostSendMessage => match (most, parse_send(arguments)) {
                (Some(most), Ok(args)) => match most
                    .send_message(args.chat_id, &args.text, args.preferred_service.as_deref())
                    .await
                {
                    Ok(value) => success(value.value, value.chat_id, value.message_id),
                    Err(error) => external_failure(error),
                },
                (None, _) => failed("most_unavailable", "Most credential is not configured"),
                (_, Err(error)) => failed("invalid_arguments", &error),
            },
            ToolOrigin::MemoryRemember => remember(state, arguments),
            ToolOrigin::MemoryRecall => recall(state, arguments),
            ToolOrigin::SelfSetPrompt => set_prompt(state, arguments),
            ToolOrigin::SelfAddRule => add_rule(state, arguments),
            ToolOrigin::SelfAddLearning => add_learning(state, arguments),
            ToolOrigin::SelfSwitchModel => switch_model(state, brama, arguments).await,
            ToolOrigin::SpawnChild => spawn_child(state, state_dir, arguments).await,
            ToolOrigin::FileRead => file_read(workspace, arguments),
            ToolOrigin::FileWrite => file_write(workspace, arguments),
        }
    }
}

mod mind;
mod outcome;
mod registry;
mod workspace;

use mind::*;
pub use outcome::*;
use registry::*;
use workspace::*;

#[cfg(test)]
#[path = "../../../tests/platform/tools.rs"]
mod tests;
