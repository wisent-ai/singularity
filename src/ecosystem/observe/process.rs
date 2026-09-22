use super::constants::{CAPTURE_BYTES, IO_BUFFER_BYTES};
use crate::AppError;
use std::process::Stdio;
use tokio::{io::{AsyncRead, AsyncReadExt}, process::Command};

async fn capture(mut stream: impl AsyncRead + Unpin) -> std::io::Result<(Vec<u8>, bool)> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; IO_BUFFER_BYTES];
    let mut truncated = false;
    loop {
        let count = stream.read(&mut buffer).await?;
        if count == 0 { return Ok((bytes, truncated)); }
        let keep = count.min(CAPTURE_BYTES.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&buffer[..keep]);
        truncated |= keep != count;
    }
}

pub(super) async fn text(mut command: Command, operation: &str) -> Result<String, AppError> {
    let mut child = command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .kill_on_drop(true).spawn().map_err(|error| AppError::Runtime(format!("{operation}: could not start: {error}")))?;
    let stdout = child.stdout.take().ok_or_else(|| AppError::Runtime(format!("{operation}: stdout pipe unavailable")))?;
    let stderr = child.stderr.take().ok_or_else(|| AppError::Runtime(format!("{operation}: stderr pipe unavailable")))?;
    let (stdout, stderr, status) = tokio::try_join!(capture(stdout), capture(stderr), child.wait())
        .map_err(|error| AppError::Runtime(format!("{operation}: could not observe completion: {error}")))?;
    if stdout.1 || stderr.1 {
        return Err(AppError::Runtime(format!("{operation}: exit {:?}; output exceeded {CAPTURE_BYTES} bytes; the operation may have completed and must be reconciled", status.code())));
    }
    if !status.success() {
        return Err(AppError::Runtime(format!("{operation}: exit {:?}; stderr: {}; stdout: {}", status.code(),
            String::from_utf8_lossy(&stderr.0), String::from_utf8_lossy(&stdout.0))));
    }
    String::from_utf8(stdout.0).map_err(|error| AppError::Runtime(format!("{operation}: output is not UTF-8: {error}")))
}
