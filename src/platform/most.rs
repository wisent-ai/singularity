use std::time::Duration;

use reqwest::{Client, Response, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::Url;
use uuid::Uuid;

use crate::error::{AppError, ErrorClass};

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
            .take("800".parse().expect("static limit"))
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
mod tests {
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use uuid::Uuid;

    use super::{parse_response, parse_uuid};
    use crate::error::{AppError, ErrorClass};

    #[test]
    fn parses_uuid_at_nested_response_pointer() {
        let expected = Uuid::parse_str("d9428888-122b-11e1-b85c-61cd3cbb3210").unwrap();
        let response = json!({"message": {"id": expected.to_string()}});

        assert_eq!(parse_uuid(&response, "/message/id").unwrap(), expected);
    }

    #[test]
    fn missing_or_invalid_response_uuid_is_a_permanent_protocol_error() {
        for (response, expected_message) in [
            (json!({"message": {}}), "response missing /message/id"),
            (
                json!({"message": {"id": "not-a-uuid"}}),
                "invalid UUID at /message/id",
            ),
        ] {
            let error = parse_uuid(&response, "/message/id").unwrap_err();
            match error {
                AppError::Most { class, message } => {
                    assert_eq!(class, ErrorClass::Permanent);
                    assert!(message.starts_with(expected_message), "{message}");
                }
                other => panic!("expected Most protocol error, got {other:?}"),
            }
        }
    }
    async fn response(status: &str, body: &str) -> reqwest::Response {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let reply = format!(
            "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let server = async move {
            let (mut socket, _) = listener.accept().await?;
            let mut request = vec![u8::default(); "1024".parse().unwrap()];
            let bytes_read = socket.read(&mut request).await?;
            assert_ne!(
                bytes_read,
                usize::default(),
                "client closed without a request"
            );
            socket.write_all(reply.as_bytes()).await
        };
        let (response, served) = tokio::join!(reqwest::get(url), server);
        served.unwrap();
        response.unwrap()
    }

    #[tokio::test]
    async fn response_status_maps_to_retry_semantics_and_operator_meaning() {
        for (status, expected_class, expected_meaning) in [
            (
                "422 Unprocessable Entity",
                ErrorClass::Permanent,
                "invalid arguments",
            ),
            (
                "500 Internal Server Error",
                ErrorClass::Transient,
                "HTTP error",
            ),
            (
                "503 Service Unavailable",
                ErrorClass::Indeterminate,
                "worker unavailable",
            ),
        ] {
            let error = parse_response::<Value>(response(status, "remote detail").await)
                .await
                .unwrap_err();
            match error {
                AppError::Most { class, message } => {
                    assert_eq!(class, expected_class, "{status}");
                    assert!(message.contains(expected_meaning), "{message}");
                    assert!(message.contains("remote detail"), "{message}");
                }
                other => panic!("expected Most HTTP error, got {other:?}"),
            }
        }
    }
}
