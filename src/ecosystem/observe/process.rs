use crate::AppError;
use std::process::Stdio;
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
};

async fn capture(mut stream: impl AsyncRead + Unpin) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).await?;
    Ok(bytes)
}

pub(super) async fn text(mut command: Command, operation: &str) -> Result<String, AppError> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| AppError::Runtime(format!("{operation}: could not start: {error}")))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::Runtime(format!("{operation}: stdout pipe unavailable")))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| AppError::Runtime(format!("{operation}: stderr pipe unavailable")))?;
    let (stdout, stderr, status) = tokio::try_join!(capture(stdout), capture(stderr), child.wait())
        .map_err(|error| {
            AppError::Runtime(format!(
                "{operation}: could not observe completion: {error}"
            ))
        })?;
    if !status.success() {
        return Err(AppError::Runtime(format!(
            "{operation}: exit {:?}; stderr: {}; stdout: {}",
            status.code(),
            String::from_utf8_lossy(&stderr),
            String::from_utf8_lossy(&stdout)
        )));
    }
    String::from_utf8(stdout)
        .map_err(|error| AppError::Runtime(format!("{operation}: output is not UTF-8: {error}")))
}
