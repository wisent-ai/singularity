use std::time::Duration;

use reqwest::{Client, Response, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;
use uuid::Uuid;

use crate::error::{AppError, ErrorClass};

/// A failed Most answer is quoted up to 800 characters.
const MAX_ERROR_EXCERPT_CHARS: usize = 800;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MostHealth {
    pub status: String,
    #[serde(default)]
    pub backends: String,
    #[serde(default)]
    pub composition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MostResult {
    pub value: Value,
    pub chat_id: Option<Uuid>,
    pub message_id: Option<Uuid>,
}

pub struct MostClient {
    http: Client,
    base_url: Url,
    token: SecretString,
}

impl MostClient {
    pub fn new(base_url: Url, token: SecretString, deadline: Duration) -> Result<Self, AppError> {
        let http = Client::builder()
            .timeout(deadline)
            .build()
            .map_err(map_network)?;
        Ok(Self {
            http,
            base_url,
            token,
        })
    }

    pub async fn health(&self) -> Result<MostHealth, AppError> {
        let response = self
            .http
            .get(self.endpoint("healthz")?)
            .send()
            .await
            .map_err(map_network)?;
        parse_response(response).await
    }

    pub async fn create_chat(
        &self,
        from: &str,
        to: &[String],
        text: &str,
        preferred_service: Option<&str>,
    ) -> Result<MostResult, AppError> {
        if from.trim().is_empty() || to.is_empty() || text.trim().is_empty() {
            return Err(most(
                ErrorClass::Permanent,
                "from, recipients, and text are required",
            ));
        }
        let mut message = json!({"parts":[{"type":"text","value":text}]});
        if let Some(service) = preferred_service {
            message["preferred_service"] = Value::String(service.into());
        }
        let body = json!({"from":from,"to":to,"message":message});
        let value: Value = parse_response(
            self.authorized(self.http.post(self.endpoint("v3/chats")?))
                .json(&body)
                .send()
                .await
                .map_err(map_indeterminate)?,
        )
        .await?;
        let chat_id = parse_uuid(&value, "/id")?;
        let message_id = parse_uuid(&value, "/message/id")?;
        Ok(MostResult {
            value,
            chat_id: Some(chat_id),
            message_id: Some(message_id),
        })
    }

    pub async fn send_message(
        &self,
        chat_id: Uuid,
        text: &str,
        preferred_service: Option<&str>,
    ) -> Result<MostResult, AppError> {
        if text.trim().is_empty() {
            return Err(most(ErrorClass::Permanent, "text is required"));
        }
        let mut message = json!({"parts":[{"type":"text","value":text}]});
        if let Some(service) = preferred_service {
            message["preferred_service"] = Value::String(service.into());
        }
        let body = json!({"message":message});
        let path = format!("v3/chats/{chat_id}/messages");
        let value: Value = parse_response(
            self.authorized(self.http.post(self.endpoint(&path)?))
                .json(&body)
                .send()
                .await
                .map_err(map_indeterminate)?,
        )
        .await?;
        let message_id = parse_uuid(&value, "/id")?;
        Ok(MostResult {
            value,
            chat_id: Some(chat_id),
            message_id: Some(message_id),
        })
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request.bearer_auth(self.token.expose_secret())
    }

    fn endpoint(&self, path: &str) -> Result<Url, AppError> {
        self.base_url
            .join(path)
            .map_err(|error| most(ErrorClass::Permanent, format!("invalid endpoint: {error}")))
    }
}

async fn parse_response<T: for<'de> Deserialize<'de>>(response: Response) -> Result<T, AppError> {
    let status = response.status();
    let bytes = response.bytes().await.map_err(map_indeterminate)?;
    if !status.is_success() {
        let message = String::from_utf8_lossy(&bytes)
            .chars()
            .take(MAX_ERROR_EXCERPT_CHARS)
            .collect::<String>();
        let class = if status == StatusCode::SERVICE_UNAVAILABLE {
            ErrorClass::Indeterminate
        } else if status.is_server_error() {
            ErrorClass::Transient
        } else {
            ErrorClass::Permanent
        };
        let meaning = match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "authentication",
            StatusCode::UNPROCESSABLE_ENTITY => "invalid arguments",
            StatusCode::NOT_IMPLEMENTED => "unsupported capability",
            StatusCode::SERVICE_UNAVAILABLE => "worker unavailable",
            _ => "HTTP error",
        };
        return Err(most(
            class,
            format!("{meaning} ({}): {message}", status.as_str()),
        ));
    }
    serde_json::from_slice(&bytes).map_err(|error| {
        most(
            ErrorClass::Permanent,
            format!("invalid response JSON: {error}"),
        )
    })
}

fn parse_uuid(value: &Value, pointer: &str) -> Result<Uuid, AppError> {
    let raw = value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| most(ErrorClass::Permanent, format!("response missing {pointer}")))?;
    Uuid::parse_str(raw).map_err(|error| {
        most(
            ErrorClass::Permanent,
            format!("invalid UUID at {pointer}: {error}"),
        )
    })
}

fn most(class: ErrorClass, message: impl Into<String>) -> AppError {
    AppError::Most {
        class,
        message: message.into(),
    }
}
fn map_network(error: reqwest::Error) -> AppError {
    most(ErrorClass::Transient, error.to_string())
}
fn map_indeterminate(error: reqwest::Error) -> AppError {
    most(ErrorClass::Indeterminate, error.to_string())
}


#[cfg(test)]
#[path = "../../tests/platform/most.rs"]
mod tests;
