// lib.rs loads this file with #[path = "platform/mcp.rs"], so its child names
// its own file, as the Brama client's children do.
#[path = "mcp/rpc.rs"]
mod rpc;

use std::path::Path;
use std::process::Stdio;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::BufReader;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::error::{AppError, ErrorClass};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SKARBIEC_PATH_ENV: &[&str] = &[
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
        && agent_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct ToolList {
    tools: Vec<McpTool>,
}

pub struct LasSupervisor {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    tools: Vec<McpTool>,
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
        trading: Option<&crate::trading::TradingSurface>,
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
        let skarbiec_agent = match (skarbiec_active, agent_id) {
            (false, _) => None,
            (true, Some(agent)) => Some(agent),
            (true, None) => {
                return Err(mcp(
                    ErrorClass::Permanent,
                    "Skarbiec requires an explicit immutable Las agent identity",
                ));
            }
        };

        let (home, path) = crate::config::environment::runtime_paths()?;
        let mut process = Command::new(command);
        process
            .arg(entrypoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .env_clear()
            .env("HOME", home)
            .env("PATH", path)
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
        if let Some(agent) = skarbiec_agent {
            for name in SKARBIEC_PATH_ENV.iter().copied().chain([SKARBIEC_COMMAND_ENV]) {
                if let Some(value) = std::env::var_os(name) {
                    process.env(name, value);
                }
            }
            process.env("SKARBIEC_MCP_AGENT_ID", agent);
        }
        if (only.trim().is_empty() || selected(only, "warsztat"))
            && !skip.is_some_and(|surfaces| selected(surfaces, "warsztat"))
        {
            for name in ["JEDEN_REPO_POLICY_FILE", "JEDEN_REPO_STATE_DIR"] {
                if let Some(value) = std::env::var_os(name) {
                    process.env(name, value);
                }
            }
        }
        if (only.trim().is_empty() || selected(only, "trading"))
            && !skip.is_some_and(|surfaces| selected(surfaces, "trading"))
        {
            if let Some(surface) = trading {
                process.envs(surface.environment());
            }
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
            tools: vec![],
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
        supervisor.refresh(required_surfaces).await?;
        Ok(supervisor)
    }

    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    pub async fn refresh(&mut self, required_surfaces: &[String]) -> Result<(), AppError> {
        let list: ToolList = serde_json::from_value(self.request("tools/list", json!({})).await?)?;
        for surface in required_surfaces {
            let prefix = format!("{surface}__");
            if !list.tools.iter().any(|tool| tool.name.starts_with(&prefix)) {
                return Err(mcp(
                    ErrorClass::Permanent,
                    format!("required Las surface unavailable: {surface}"),
                ));
            }
        }
        self.tools = list.tools;
        Ok(())
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
}

fn mcp(class: ErrorClass, message: impl Into<String>) -> AppError {
    AppError::Mcp {
        class,
        message: message.into(),
    }
}
