use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::domain::{ChatMessage, ToolCall, ToolDefinition};
use crate::error::AppError;
use crate::mcp::{LasSupervisor, McpTool};
use crate::most::MostClient;

const MOST_HEALTH: &str = "most_health";
const MOST_CREATE_CHAT: &str = "most_create_chat";
const MOST_SEND_MESSAGE: &str = "most_send_message";

#[derive(Debug, Clone)]
enum ToolOrigin {
    Las,
    MostHealth,
    MostCreateChat,
    MostSendMessage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Success,
    Failed,
    Indeterminate,
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
        ChatMessage::tool(
            call,
            serde_json::to_value(self)
                .unwrap_or_else(|_| json!({"status":"failed","error_code":"serialization"})),
        )
    }
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
        register(
            &mut definitions,
            &mut origins,
            ToolDefinition::function(
                MOST_HEALTH,
                "Check Most messaging readiness and active backends",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            ),
            ToolOrigin::MostHealth,
        )?;
        register(
            &mut definitions,
            &mut origins,
            ToolDefinition::function(
                MOST_CREATE_CHAT,
                "Create a Most chat and send its first text message",
                json!({"type":"object","properties":{"from":{"type":"string"},"to":{"type":"array","items":{"type":"string"}},"text":{"type":"string"},"preferred_service":{"type":"string","enum":["iMessage","SMS","RCS"]}},"required":["from","to","text"],"additionalProperties":false}),
            ),
            ToolOrigin::MostCreateChat,
        )?;
        register(
            &mut definitions,
            &mut origins,
            ToolDefinition::function(
                MOST_SEND_MESSAGE,
                "Send a text message to an existing Most chat",
                json!({"type":"object","properties":{"chat_id":{"type":"string","format":"uuid"},"text":{"type":"string"},"preferred_service":{"type":"string","enum":["iMessage","SMS","RCS"]}},"required":["chat_id","text"],"additionalProperties":false}),
            ),
            ToolOrigin::MostSendMessage,
        )?;
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
        }
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
