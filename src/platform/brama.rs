use std::time::Duration;

use hmac::{Hmac, Mac};
use reqwest::{Client, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use url::Url;

use crate::domain::{ChatMessage, TokenUsage, ToolCall, ToolDefinition};
use crate::error::{AppError, ErrorClass};

type HmacSha = Hmac<Sha256>;

/// A non-JSON error body is quoted up to 800 characters.
const MAX_ERROR_EXCERPT_CHARS: usize = 800;

#[derive(Debug, Serialize)]
struct CompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    max_tokens: u32,
    temperature: f64,
    tools: &'a [ToolDefinition],
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    #[allow(dead_code)]
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: RemoteError,
}

#[derive(Debug, Deserialize)]
struct RemoteError {
    message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BramaCompletion {
    pub model: String,
    /// The assistant text, when the response carried any: a response that ends in tool calls
    /// carries none, and the absence stays visible instead of reading as an empty answer.
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
}

pub struct BramaClient {
    http: Client,
    base_url: Url,
    model: String,
    agent_id: String,
    secret: SecretString,
    max_tokens: u32,
    temperature: f64,
}

impl BramaClient {
    pub fn new(
        base_url: Url,
        model: String,
        agent_id: String,
        secret: SecretString,
        max_tokens: u32,
        temperature: f64,
        timeout: Duration,
    ) -> Result<Self, AppError> {
        let http = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(map_network)?;
        Ok(Self {
            http,
            base_url,
            model,
            agent_id,
            secret,
            max_tokens,
            temperature,
        })
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub async fn health(&self) -> Result<(), AppError> {
        let value: Value = self
            .http
            .get(self.endpoint("health")?)
            .send()
            .await
            .map_err(map_network)?
            .error_for_status()
            .map_err(map_network)?
            .json()
            .await
            .map_err(map_network)?;
        if value.get("status").and_then(Value::as_str) != Some("ok") {
            return Err(brama(ErrorClass::Permanent, "health response is not ok"));
        }
        Ok(())
    }

    pub async fn models(&self) -> Result<Vec<String>, AppError> {
        let response: ModelsResponse = self
            .http
            .get(self.endpoint("v1/models")?)
            .send()
            .await
            .map_err(map_network)?
            .error_for_status()
            .map_err(map_network)?
            .json()
            .await
            .map_err(map_network)?;
        Ok(response.data.into_iter().map(|entry| entry.id).collect())
    }

    pub async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<BramaCompletion, AppError> {
        let request = CompletionRequest {
            model: &self.model,
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            tools,
        };
        let body = serde_json::to_vec(&request)?;
        let timestamp = chrono::Utc::now().timestamp().to_string();
        let body_hash = hex::encode(Sha256::digest(&body));
        let signed = format!("{}:{}:{}", self.agent_id, timestamp, body_hash);
        let mut mac =
            HmacSha::new_from_slice(self.secret.expose_secret().as_bytes()).map_err(|error| {
                brama(
                    ErrorClass::Permanent,
                    format!("cannot initialize signer: {error}"),
                )
            })?;
        mac.update(signed.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        let response = self
            .http
            .post(self.endpoint("v1/chat/completions")?)
            .header("content-type", "application/json")
            .header("x-agent-id", &self.agent_id)
            .header("x-agent-timestamp", timestamp)
            .header("x-agent-body-sha256", body_hash)
            .header("x-agent-signature", signature)
            .body(body)
            .send()
            .await
            .map_err(map_indeterminate)?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(map_indeterminate)?;
        if !status.is_success() {
            let message = match serde_json::from_slice::<ErrorEnvelope>(&bytes) {
                Ok(value) => value.error.message,
                Err(error) => {
                    let excerpt: String = String::from_utf8_lossy(&bytes)
                        .chars()
                        .take(MAX_ERROR_EXCERPT_CHARS)
                        .collect();
                    format!("error body is not a Brama error envelope ({error}): {excerpt}")
                }
            };
            let class = if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                ErrorClass::Transient
            } else {
                ErrorClass::Permanent
            };
            return Err(brama(class, format!("HTTP {}: {message}", status.as_str())));
        }
        let parsed: CompletionResponse = serde_json::from_slice(&bytes)?;
        if parsed.choices.len() != usize::from(true) {
            return Err(brama(
                ErrorClass::Permanent,
                "response must contain exactly one choice",
            ));
        }
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .expect("choice length checked");
        let calls = choice.message.tool_calls.filter(|calls| !calls.is_empty());
        let tool_calls = match (choice.finish_reason.as_str(), calls) {
            ("tool_calls", None) => {
                return Err(brama(
                    ErrorClass::Permanent,
                    "tool_calls finish reason without calls",
                ));
            }
            ("stop", Some(_)) => {
                return Err(brama(
                    ErrorClass::Permanent,
                    "stop finish reason with tool calls",
                ));
            }
            ("tool_calls", Some(calls)) => calls,
            ("stop", None) => Vec::new(),
            (other, _) => {
                return Err(brama(
                    ErrorClass::Permanent,
                    format!("unsupported finish reason: {other}"),
                ));
            }
        };
        let computed_total = parsed
            .usage
            .prompt_tokens
            .saturating_add(parsed.usage.completion_tokens);
        if parsed.usage.total_tokens != u64::default()
            && parsed.usage.total_tokens != computed_total
        {
            return Err(brama(
                ErrorClass::Permanent,
                "usage total does not match prompt plus completion",
            ));
        }
        Ok(BramaCompletion {
            model: parsed.model,
            content: choice.message.content,
            tool_calls,
            usage: TokenUsage {
                prompt_tokens: parsed.usage.prompt_tokens,
                completion_tokens: parsed.usage.completion_tokens,
                total_tokens: computed_total,
            },
        })
    }

    pub fn configured_model(&self) -> &str {
        &self.model
    }

    fn endpoint(&self, path: &str) -> Result<Url, AppError> {
        self.base_url
            .join(path)
            .map_err(|error| brama(ErrorClass::Permanent, format!("invalid endpoint: {error}")))
    }
}

fn brama(class: ErrorClass, message: impl Into<String>) -> AppError {
    AppError::Brama {
        class,
        message: message.into(),
    }
}
fn map_network(error: reqwest::Error) -> AppError {
    brama(ErrorClass::Transient, error.to_string())
}
fn map_indeterminate(error: reqwest::Error) -> AppError {
    brama(ErrorClass::Indeterminate, error.to_string())
}
