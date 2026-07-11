use std::{collections::HashMap, time::Duration};

use hmac::{Hmac, Mac};
use secrecy::SecretString;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use url::Url;

use crate::{
    BramaClient,
    domain::{ChatMessage, Role, ToolDefinition},
};

struct CapturedRequest {
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

async fn capture_one_request(listener: TcpListener, response_body: String) -> CapturedRequest {
    let (mut socket, _) = listener.accept().await.unwrap();
    let delimiter = b"\r\n\r\n";
    let mut request = Vec::new();
    let mut chunk = vec![u8::default(); "1024".parse().unwrap()];
    let header_end = loop {
        if let Some(offset) = request
            .windows(delimiter.len())
            .position(|window| window == delimiter)
        {
            break offset + delimiter.len();
        }
        let read = socket.read(&mut chunk).await.unwrap();
        assert_ne!(
            read,
            usize::default(),
            "connection closed before HTTP headers completed"
        );
        request.extend_from_slice(&chunk[..read]);
    };

    let header_text = std::str::from_utf8(&request[..header_end - delimiter.len()]).unwrap();
    let mut lines = header_text.split("\r\n");
    let _ = lines.next();
    let headers: HashMap<_, _> = lines
        .map(|line| {
            let (name, value) = line.split_once(':').unwrap();
            (name.to_ascii_lowercase(), value.trim().to_owned())
        })
        .collect();
    let content_length: usize = headers["content-length"].parse().unwrap();
    let request_end = header_end + content_length;
    while request.len() < request_end {
        let read = socket.read(&mut chunk).await.unwrap();
        assert_ne!(
            read,
            usize::default(),
            "connection closed before HTTP body completed"
        );
        request.extend_from_slice(&chunk[..read]);
    }
    let body = request[header_end..request_end].to_vec();

    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        response_body.len(),
        response_body
    );
    socket.write_all(response.as_bytes()).await.unwrap();

    CapturedRequest { headers, body }
}

#[tokio::test]
async fn complete_signs_the_exact_body_and_parses_native_tool_call_with_usage() {
    let prompt_tokens = "17".parse::<u64>().unwrap();
    let completion_tokens = "5".parse::<u64>().unwrap();
    let total_tokens = "22".parse::<u64>().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
    let response_body = json!({
        "id": "completion-exact-body",
        "model": "brama-test-model",
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call-publish-1",
                    "type": "function",
                    "function": {
                        "name": "publish_article",
                        "arguments": "{\"slug\":\"żubr\",\"featured\":true}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    })
    .to_string();
    let server = tokio::spawn(capture_one_request(listener, response_body));

    let agent_id = "agent-exact-body";
    let key_material = "fixture-key-material";
    let client = BramaClient::new(
        base_url,
        "brama-test-model".to_owned(),
        agent_id.to_owned(),
        SecretString::from(key_material.to_owned()),
        "256".parse().unwrap(),
        "0.25".parse().unwrap(),
        Duration::from_secs("2".parse().unwrap()),
    )
    .unwrap();
    let messages = [ChatMessage::text(Role::User, "Napisz o żubrze")];
    let tools = [ToolDefinition::function(
        "publish_article",
        "Publikuje artykuł",
        json!({
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "featured": {"type": "boolean"}
            },
            "required": ["slug", "featured"]
        }),
    )];

    let completion = client.complete(&messages, &tools).await.unwrap();
    let captured = server.await.unwrap();

    assert_eq!(captured.headers["x-agent-id"], agent_id);
    let exact_body_hash = hex::encode(Sha256::digest(&captured.body));
    assert_eq!(
        captured.headers["x-agent-body-sha256"], exact_body_hash,
        "body hash must cover the bytes actually sent on the wire"
    );
    let signed = format!(
        "{}:{}:{}",
        agent_id, captured.headers["x-agent-timestamp"], exact_body_hash
    );
    let mut mac = Hmac::<Sha256>::new_from_slice(key_material.as_bytes()).unwrap();
    mac.update(signed.as_bytes());
    let signature = hex::decode(&captured.headers["x-agent-signature"]).unwrap();
    mac.verify_slice(&signature).unwrap();

    assert_eq!(completion.model, "brama-test-model");
    assert_eq!(completion.content, "");
    let [call] = completion.tool_calls.as_slice() else {
        panic!("expected exactly one native tool call");
    };
    assert_eq!(call.id, "call-publish-1");
    assert_eq!(call.kind, "function");
    assert_eq!(call.function.name, "publish_article");
    assert_eq!(
        serde_json::from_str::<Value>(&call.function.arguments).unwrap(),
        json!({"slug": "żubr", "featured": true})
    );
    assert_eq!(completion.usage.prompt_tokens, prompt_tokens);
    assert_eq!(completion.usage.completion_tokens, completion_tokens);
    assert_eq!(completion.usage.total_tokens, total_tokens);
}
