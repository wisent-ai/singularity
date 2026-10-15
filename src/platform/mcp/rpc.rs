//! One JSON-RPC exchange with Las over its stdio, and Las's shutdown.

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

use super::{LasSupervisor, mcp};
use crate::error::{AppError, ErrorClass};

impl LasSupervisor {
    pub(super) async fn request(&mut self, method: &str, params: Value) -> Result<Value, AppError> {
        // JSON-RPC lets an id be text; a random one pairs this answer with
        // this question without a counter.
        let id = uuid::Uuid::new_v4().to_string();
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
        let stdout = &mut self.stdout;
        let answer = async {
            loop {
                let mut line = String::new();
                stdout
                    .read_line(&mut line)
                    .await
                    .map_err(|error| mcp(ErrorClass::Indeterminate, error.to_string()))?;
                // read_line leaves the line empty only at the end of the
                // stream; a blank line still carries its newline.
                if line.is_empty() {
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
                if value.get("id").and_then(Value::as_str) != Some(id.as_str()) {
                    if value.get("id").is_none() {
                        continue;
                    }
                    return Err(mcp(
                        ErrorClass::Permanent,
                        "Las returned an unexpected response id",
                    ));
                }
                if let Some(error) = value.get("error") {
                    let described = match (error.get("code"), error.get("message")) {
                        (Some(code), Some(message)) => format!("JSON-RPC {code}: {message}"),
                        _ => format!("JSON-RPC error object without a code and a message: {error}"),
                    };
                    return Err(mcp(ErrorClass::Indeterminate, described));
                }
                return value
                    .get("result")
                    .cloned()
                    .ok_or_else(|| mcp(ErrorClass::Permanent, "Las response has no result"));
            }
        };
        stado_wait::until(
            stado_wait::Kind::Process,
            format!("Las's answer to {method}"),
            "Las stdio",
            answer,
        )
        .await
    }

    /// Stop Las the way MCP's stdio transport describes: close its input, then
    /// ask it to exit with SIGTERM, and wait for it to do so. No grace period
    /// is guessed before a SIGKILL: SIGTERM is the request to stop, and a
    /// server that ignores it is a defect of that server, which this wait
    /// then makes visible instead of cutting short.
    pub async fn shutdown(&mut self) -> Result<(), AppError> {
        self.stdin.take();
        let place = match self.child.id() {
            Some(pid) => {
                let signalled = libc::pid_t::try_from(pid).map_err(|error| {
                    mcp(
                        ErrorClass::Permanent,
                        format!("Las pid {pid} is not a valid process id: {error}"),
                    )
                })?;
                // SAFETY: kill sends a signal to our own child by its pid.
                if unsafe { libc::kill(signalled, libc::SIGTERM) }.is_negative() {
                    let error = std::io::Error::last_os_error();
                    // The child has already exited and been reaped; wait() reports its status.
                    if error.raw_os_error() != Some(libc::ESRCH) {
                        return Err(mcp(
                            ErrorClass::Transient,
                            format!("cannot ask Las (pid {pid}) to stop: {error}"),
                        ));
                    }
                }
                format!("pid {pid}")
            }
            None => "a Las process that has already ended".to_string(),
        };
        stado_wait::until(
            stado_wait::Kind::Process,
            "Las to exit after SIGTERM",
            place,
            self.child.wait(),
        )
        .await
        .map_err(|error| mcp(ErrorClass::Transient, error.to_string()))?;
        Ok(())
    }
}
