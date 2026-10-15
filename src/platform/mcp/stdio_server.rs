//! The MCP server loop both owner-only surfaces run over stdio: one JSON-RPC
//! request per line in, one response per line out
//! (https://modelcontextprotocol.io/specification/2024-11-05/basic/transports).
//! A surface supplies its tools and how a call is answered; everything about
//! the protocol lives here once.

use serde_json::{Value, json};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

/// The MCP revision these servers speak.
const PROTOCOL_VERSION: &str = "2024-11-05"; // https://modelcontextprotocol.io/specification/2024-11-05

const PARSE_ERROR: i64 = -32700; // https://www.jsonrpc.org/specification#error_object "-32700 Parse error"
const INVALID_REQUEST: i64 = -32600; // https://www.jsonrpc.org/specification#error_object "-32600 Invalid Request"
const METHOD_NOT_FOUND: i64 = -32601; // https://www.jsonrpc.org/specification#error_object "-32601 Method not found"
const INVALID_PARAMS: i64 = -32602; // https://www.jsonrpc.org/specification#error_object "-32602 Invalid params"

/// One surface's tools and its answer to a call, as an MCP tool result.
pub(crate) trait ToolHost {
    /// The server name `initialize` reports.
    fn server_name(&self) -> &'static str;
    /// The `tools/list` result's tools.
    fn tools(&self) -> Value;
    /// The `tools/call` result for `name` with `arguments`; a refusal is a
    /// tool result with `isError`, not a protocol error.
    async fn call(&self, name: &str, arguments: Value) -> Value;
}

/// Answers every request on stdin until stdin ends. A line that is not JSON
/// is answered with the parser's own reason; a notification (no `id`) gets no
/// answer, as JSON-RPC requires.
pub(crate) async fn serve<H: ToolHost>(host: &H) -> io::Result<()> {
    let mut lines = BufReader::new(io::stdin()).lines();
    let mut stdout = io::stdout();
    while let Some(line) = lines.next_line().await? {
        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                let response =
                    rpc_error(Value::Null, PARSE_ERROR, "parse error", &error.to_string());
                write_response(&mut stdout, &response).await?;
                continue;
            }
        };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let response = answer(host, &request, id).await;
        write_response(&mut stdout, &response).await?;
    }
    Ok(())
}

async fn answer<H: ToolHost>(host: &H, request: &Value, id: Value) -> Value {
    match request.get("method").and_then(Value::as_str) {
        Some("initialize") => result(
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": host.server_name(), "version": env!("CARGO_PKG_VERSION")},
            }),
        ),
        Some("ping") => result(id, json!({})),
        Some("tools/list") => result(id, json!({"tools": host.tools()})),
        Some("tools/call") => {
            let Some(params) = request.get("params") else {
                return rpc_error(
                    id,
                    INVALID_PARAMS,
                    "invalid params",
                    "tools/call requires params",
                );
            };
            let Some(name) = params.get("name").and_then(Value::as_str) else {
                return rpc_error(
                    id,
                    INVALID_PARAMS,
                    "invalid params",
                    "tools/call requires a string name",
                );
            };
            let Some(arguments) = params.get("arguments").cloned() else {
                return rpc_error(
                    id,
                    INVALID_PARAMS,
                    "invalid params",
                    "tools/call requires an arguments object",
                );
            };
            result(id, host.call(name, arguments).await)
        }
        Some(method) => rpc_error(
            id,
            METHOD_NOT_FOUND,
            "method not found",
            &format!("{} does not answer {method}", host.server_name()),
        ),
        None => rpc_error(
            id,
            INVALID_REQUEST,
            "invalid request",
            "the request names no method",
        ),
    }
}

/// A successful tool answer: the value as text and as structured content.
pub(crate) fn tool_answer(value: &Value) -> Result<Value, serde_json::Error> {
    let text = serde_json::to_string(value)?;
    Ok(
        json!({"content": [{"type": "text", "text": text}], "structuredContent": value, "isError": false}),
    )
}

fn result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn rpc_error(id: Value, code: i64, message: &str, data: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": data}})
}

async fn write_response(stdout: &mut io::Stdout, response: &Value) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(response).map_err(io::Error::other)?;
    bytes.push(b'\n');
    stdout.write_all(&bytes).await?;
    stdout.flush().await
}
