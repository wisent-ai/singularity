use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::sleep;

use crate::error::{AppError, ErrorClass};

const PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default = "empty_schema")]
    pub input_schema: Value,
}

fn empty_schema() -> Value {
    json!({"type":"object","properties":{}})
}

#[derive(Debug, Deserialize)]
struct ToolList {
    tools: Vec<McpTool>,
}

pub struct LasSupervisor {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    tools: Vec<McpTool>,
    request_deadline: Duration,
}

impl LasSupervisor {
    #[allow(clippy::too_many_arguments)]
    pub async fn spawn(
        command: &str,
        entrypoint: &Path,
        only: &str,
        skip: Option<&str>,
        required_surfaces: &[String],
        request_deadline: Duration,
    ) -> Result<Self, AppError> {
        let mut process = Command::new(command);
        process
            .arg(entrypoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("LAS_ONLY", only);
        if let Some(value) = skip {
            process.env("LAS_SKIP", value);
        }
        let mut child = process
            .spawn()
            .map_err(|error| mcp(ErrorClass::Permanent, format!("cannot start Las: {error}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| mcp(ErrorClass::Permanent, "Las stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| mcp(ErrorClass::Permanent, "Las stdout unavailable"))?;
        let mut supervisor = Self {
            child,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            next_id: u64::default(),
            tools: vec![],
            request_deadline,
        };
        let initialized = supervisor.request("initialize", json!({"protocolVersion":PROTOCOL_VERSION,"capabilities":{},"clientInfo":{"name":"singularity","version":env!("CARGO_PKG_VERSION")}})).await?;
        if initialized.get("protocolVersion").and_then(Value::as_str) != Some(PROTOCOL_VERSION) {
            return Err(mcp(
                ErrorClass::Permanent,
                "Las negotiated an unsupported MCP version",
            ));
        }
        let list: ToolList =
            serde_json::from_value(supervisor.request("tools/list", json!({})).await?)?;
        for surface in required_surfaces {
            let prefix = format!("{surface}__");
            if !list.tools.iter().any(|tool| tool.name.starts_with(&prefix)) {
                return Err(mcp(
                    ErrorClass::Permanent,
                    format!("required Las surface unavailable: {surface}"),
                ));
            }
        }
        supervisor.tools = list.tools;
        Ok(supervisor)
    }

    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value, AppError> {
        if !self.tools.iter().any(|tool| tool.name == name) {
            return Err(mcp(
                ErrorClass::Permanent,
                format!("unknown Las tool: {name}"),
            ));
        }
        self.request("tools/call", json!({"name":name,"arguments":arguments}))
            .await
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, AppError> {
        self.next_id = self.next_id.saturating_add(u64::from(true));
        let id = self.next_id;
        let payload =
            serde_json::to_vec(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| mcp(ErrorClass::Permanent, "Las is shutting down"))?;
        stdin
            .write_all(&payload)
            .await
            .map_err(|error| mcp(ErrorClass::Indeterminate, error.to_string()))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|error| mcp(ErrorClass::Indeterminate, error.to_string()))?;
        stdin
            .flush()
            .await
            .map_err(|error| mcp(ErrorClass::Indeterminate, error.to_string()))?;
        let wait = async {
            loop {
                let mut line = String::new();
                let bytes = self
                    .stdout
                    .read_line(&mut line)
                    .await
                    .map_err(|error| mcp(ErrorClass::Indeterminate, error.to_string()))?;
                if bytes == usize::default() {
                    return Err(mcp(ErrorClass::Indeterminate, "Las closed stdout"));
                }
                if line.trim().is_empty() {
                    continue;
                }
                let value: Value = serde_json::from_str(line.trim()).map_err(|error| {
                    mcp(
                        ErrorClass::Permanent,
                        format!("non-JSON Las stdout: {error}"),
                    )
                })?;
                if value.get("id").and_then(Value::as_u64) != Some(id) {
                    if value.get("id").is_none() {
                        continue;
                    }
                    return Err(mcp(
                        ErrorClass::Permanent,
                        "Las returned an unexpected response id",
                    ));
                }
                if let Some(error) = value.get("error") {
                    let code = error.get("code").cloned().unwrap_or(Value::Null);
                    let message = error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("remote MCP error");
                    return Err(mcp(
                        ErrorClass::Indeterminate,
                        format!("JSON-RPC {code}: {message}"),
                    ));
                }
                return value
                    .get("result")
                    .cloned()
                    .ok_or_else(|| mcp(ErrorClass::Permanent, "Las response has no result"));
            }
        };
        tokio::select! {
            result = wait => result,
            _ = sleep(self.request_deadline) => Err(mcp(ErrorClass::Indeterminate, format!("Las request deadline exceeded: {method}"))),
        }
    }

    pub async fn shutdown(&mut self, grace: Duration) -> Result<(), AppError> {
        self.stdin.take();
        tokio::select! {
            result = self.child.wait() => { result.map_err(|error| mcp(ErrorClass::Transient, error.to_string()))?; }
            _ = sleep(grace) => {
                self.child.kill().await.map_err(|error| mcp(ErrorClass::Transient, error.to_string()))?;
                self.child.wait().await.map_err(|error| mcp(ErrorClass::Transient, error.to_string()))?;
            }
        }
        Ok(())
    }
}

fn mcp(class: ErrorClass, message: impl Into<String>) -> AppError {
    AppError::Mcp {
        class,
        message: message.into(),
    }
}
