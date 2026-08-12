use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

use crate::config::RuntimeConfig;
use crate::error::AppError;

const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResult {
    pub request_id: String,
    pub text: String,
    pub session_path: PathBuf,
}

#[derive(Debug)]
pub struct SessionHandle {
    pub id: String,
    pub path: PathBuf,
}

pub struct JedenRpc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    deadline: Duration,
}

impl JedenRpc {
    pub async fn spawn(command: &Path, cwd: &Path, deadline: Duration) -> Result<Self, AppError> {
        let mut child = Command::new(command)
            .arg("rpc")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                AppError::Jeden(format!("failed to start {}: {error}", command.display()))
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AppError::Jeden("RPC stdin is unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AppError::Jeden("RPC stdout is unavailable".into()))?;
        let mut rpc = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            deadline,
        };
        let ready = rpc.read_message().await?;
        if ready.get("type").and_then(Value::as_str) != Some("ready")
            || ready.get("protocol").and_then(Value::as_str) != Some("jeden-rpc")
            || ready.get("version").and_then(Value::as_u64) != Some(1)
        {
            return Err(AppError::Jeden(
                "unexpected Jeden RPC readiness envelope".into(),
            ));
        }
        let initialized = rpc.request("initialize", json!({})).await?;
        if initialized.get("protocol").and_then(Value::as_str) != Some("jeden-rpc") {
            return Err(AppError::Jeden("Jeden RPC initialization failed".into()));
        }
        Ok(rpc)
    }

    pub async fn create_session(
        &mut self,
        config: &RuntimeConfig,
    ) -> Result<SessionHandle, AppError> {
        let options = json!({
            "cwd": config.workspace,
            "model": config.model,
            "maxSteps": config.max_steps,
            "allowWrite": config.allow_write,
            "allowCommand": config.allow_command,
            "autoApprove": config.auto_approve,
        });
        let result = if let Some(path) = &config.resume_session {
            self.request("session/open", json!({"session": path, "options": options}))
                .await?
        } else {
            self.request("session/new", json!({"options": options}))
                .await?
        };
        let id = result
            .get("sessionId")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| AppError::Jeden("session response omitted sessionId".into()))?;
        let path = result
            .get("sessionPath")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| AppError::Jeden("session response omitted sessionPath".into()))?;
        Ok(SessionHandle { id, path })
    }

    pub async fn prompt(
        &mut self,
        session_id: &str,
        request_id: &str,
        prompt: &str,
        goal: &str,
    ) -> Result<PromptResult, AppError> {
        let result = self
            .request(
                "session/prompt",
                json!({
                    "sessionId": session_id,
                    "requestId": request_id,
                    "prompt": prompt,
                    "goal": goal,
                }),
            )
            .await?;
        serde_json::from_value(result)
            .map_err(|error| AppError::Jeden(format!("invalid prompt result: {error}")))
    }

    pub async fn shutdown(mut self) -> Result<(), AppError> {
        let request = self.request("shutdown", json!({})).await;
        self.stdin.shutdown().await.ok();
        let waited = timeout(self.deadline, self.child.wait())
            .await
            .map_err(|_| AppError::Jeden("Jeden RPC did not stop before its deadline".into()))?
            .map_err(AppError::Io)?;
        request?;
        if !waited.success() {
            return Err(AppError::Jeden(format!("Jeden RPC exited with {waited}")));
        }
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, AppError> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.write(json!({"id": id, "method": method, "params": params}))
            .await?;
        loop {
            let message = self.read_message().await?;
            if message.get("id").and_then(Value::as_u64) == Some(id) {
                if let Some(error) = message.get("error") {
                    let code = error
                        .get("code")
                        .and_then(Value::as_str)
                        .unwrap_or("rpc_error");
                    let detail = error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("Jeden RPC request failed");
                    return Err(AppError::Jeden(format!("{code}: {detail}")));
                }
                return message
                    .get("result")
                    .cloned()
                    .ok_or_else(|| AppError::Jeden("RPC response omitted result".into()));
            }
            self.handle_notification(&message).await?;
        }
    }

    async fn handle_notification(&mut self, message: &Value) -> Result<(), AppError> {
        match message.get("method").and_then(Value::as_str) {
            Some("session/event") => {
                if let Some(params) = message.get("params") {
                    tracing::debug!(event = %params, "Jeden session event");
                }
            }
            Some("session/request_permission") => {
                let token = message
                    .pointer("/params/token")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AppError::Jeden("permission request omitted token".into()))?;
                self.write_auxiliary(
                    "approval/resolve",
                    json!({"token": token, "approved": false}),
                )
                .await?;
            }
            Some("session/request_input") => {
                let token = message
                    .pointer("/params/token")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AppError::Jeden("input request omitted token".into()))?;
                self.write_auxiliary("elicitation/resolve", json!({"token": token}))
                    .await?;
            }
            _ => {}
        }
        Ok(())
    }

    async fn write_auxiliary(&mut self, method: &str, params: Value) -> Result<(), AppError> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.write(json!({"id": id, "method": method, "params": params}))
            .await
    }

    async fn write(&mut self, value: Value) -> Result<(), AppError> {
        let mut frame = serde_json::to_vec(&value)?;
        if frame.len() > MAX_FRAME_BYTES {
            return Err(AppError::Jeden("outbound RPC frame exceeds 1 MiB".into()));
        }
        frame.push(b'\n');
        self.stdin.write_all(&frame).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_message(&mut self) -> Result<Value, AppError> {
        let mut frame = Vec::new();
        let count = timeout(self.deadline, self.stdout.read_until(b'\n', &mut frame))
            .await
            .map_err(|_| AppError::Jeden("Jeden RPC response deadline exceeded".into()))??;
        if count == 0 {
            return Err(AppError::Jeden("Jeden RPC closed stdout".into()));
        }
        if frame.len() > MAX_FRAME_BYTES {
            return Err(AppError::Jeden("inbound RPC frame exceeds 1 MiB".into()));
        }
        serde_json::from_slice(&frame)
            .map_err(|error| AppError::Jeden(format!("invalid RPC JSON: {error}")))
    }
}
