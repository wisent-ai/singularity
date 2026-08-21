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
const SKARBIEC_PATH_ENV: [&str; 9] = [
    "SKARBIEC_CAP_POLICY",
    "SKARBIEC_CAP_POLICY_SIG",
    "SKARBIEC_CAP_TRUST_ROOT",
    "SKARBIEC_WORKLOAD_REGISTRY",
    "SKARBIEC_WORKLOAD_REGISTRY_SIG",
    "SKARBIEC_CAP_STATE",
    "SKARBIEC_CAP_SOCKET",
    "SKARBIEC_WORM_RECEIPT_DIR",
    "SKARBIEC_WORM_CHECKPOINT",
];
const SKARBIEC_COMMAND_ENV: &str = "SKARBIEC_WORM_RECEIPT_COMMAND";

fn selected(csv: &str, name: &str) -> bool {
    csv.split(',').map(str::trim).any(|item| item == name)
}

fn valid_agent_id(agent_id: &str) -> bool {
    !agent_id.is_empty()
        && agent_id.len() <= 128
        && agent_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

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
        agent_id: Option<&str>,
        release_manifest: &Path,
        release_manifest_signature: &Path,
        release_trust_store: &Path,
        release_watermark: &Path,
        required_surfaces: &[String],
        request_deadline: Duration,
    ) -> Result<Self, AppError> {
        if let Some(value) = agent_id
            && !valid_agent_id(value)
        {
            return Err(mcp(
                ErrorClass::Permanent,
                "invalid immutable Las agent identity",
            ));
        }
        let skarbiec_active = (only.trim().is_empty() || selected(only, "skarbiec"))
            && !skip.is_some_and(|surfaces| selected(surfaces, "skarbiec"));
        if skarbiec_active && agent_id.is_none() {
            return Err(mcp(
                ErrorClass::Permanent,
                "Skarbiec requires an explicit immutable Las agent identity",
            ));
        }

        let mut process = Command::new(command);
        process
            .arg(entrypoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env_clear()
            .env(
                "PATH",
                "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
            )
            .env("LANG", "C.UTF-8")
            .env("LC_ALL", "C.UTF-8")
            .env("LAS_ONLY", only)
            .env("LAS_RELEASE_MANIFEST_FILE", release_manifest)
            .env(
                "LAS_RELEASE_MANIFEST_SIGNATURE_FILE",
                release_manifest_signature,
            )
            .env("LAS_RELEASE_TRUST_STORE_FILE", release_trust_store)
            .env("LAS_RELEASE_WATERMARK_FILE", release_watermark);
        if let Some(value) = skip {
            process.env("LAS_SKIP", value);
        }
        if skarbiec_active {
            for name in SKARBIEC_PATH_ENV.into_iter().chain([SKARBIEC_COMMAND_ENV]) {
                if let Some(value) = std::env::var_os(name) {
                    process.env(name, value);
                }
            }
            process.env(
                "SKARBIEC_MCP_AGENT_ID",
                agent_id.expect("Skarbiec identity checked above"),
            );
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
        let mut initialize_params = json!({"protocolVersion":PROTOCOL_VERSION,"capabilities":{},"clientInfo":{"name":"singularity","version":env!("CARGO_PKG_VERSION")}});
        if let Some(value) = agent_id {
            initialize_params["agentId"] = Value::String(value.to_owned());
        }
        let initialized = supervisor.request("initialize", initialize_params).await?;
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
