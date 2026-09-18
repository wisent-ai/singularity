// Most client stories, attached to src/platform/most.rs.

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
        let mut request = vec![u8::default(); 1024];
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
