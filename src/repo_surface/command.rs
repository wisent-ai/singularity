use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::timeout;

use super::{SurfaceError, SurfaceResult};

pub const OUTPUT_CAP: usize = 256 * 1024;

#[derive(Debug)]
pub struct CommandOutput {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub truncated: bool,
}

#[derive(Clone, Copy)]
enum EnvironmentProfile {
    Local,
    GitNetwork,
    Github,
}

async fn drain_capped<R: AsyncRead + Unpin>(mut reader: R) -> std::io::Result<(Vec<u8>, bool)> {
    let mut kept = Vec::with_capacity(8192);
    let mut buf = [0_u8; 8192];
    let mut truncated = false;
    loop {
        let count = reader.read(&mut buf).await?;
        if count == 0 {
            break;
        }
        let remaining = OUTPUT_CAP.saturating_sub(kept.len());
        kept.extend_from_slice(&buf[..count.min(remaining)]);
        truncated |= count > remaining;
    }
    Ok((kept, truncated))
}

type DrainTask = tokio::task::JoinHandle<std::io::Result<(Vec<u8>, bool)>>;

async fn terminate_child(
    child: &mut tokio::process::Child,
    process_id: Option<u32>,
    stdout_task: &mut DrainTask,
    stderr_task: &mut DrainTask,
) {
    #[cfg(unix)]
    if let Some(process_id) = process_id {
        unsafe {
            kill(-(process_id as i32), SIGKILL);
        }
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
    let drain_grace = Duration::from_secs(5);
    if timeout(drain_grace, &mut *stdout_task).await.is_err() {
        stdout_task.abort();
    }
    if timeout(drain_grace, &mut *stderr_task).await.is_err() {
        stderr_task.abort();
    }
}

async fn run_fixed(
    program: &Path,
    args: &[String],
    cwd: &Path,
    stdin: Option<&[u8]>,
    timeout_secs: u64,
    environment: EnvironmentProfile,
) -> SurfaceResult<CommandOutput> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .env_clear()
        .env(
            "PATH",
            "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin",
        )
        .env("HOME", "/var/empty")
        .env("XDG_CONFIG_HOME", "/var/empty")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GH_PROMPT_DISABLED", "1")
        .env("LC_ALL", "C");
    let inherited = match environment {
        EnvironmentProfile::Local => &[][..],
        EnvironmentProfile::GitNetwork => &["HOME", "SSH_AUTH_SOCK"][..],
        EnvironmentProfile::Github => &["HOME", "XDG_CONFIG_HOME", "GH_TOKEN", "GH_HOST"][..],
    };
    for name in inherited {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|e| SurfaceError::command(format!("cannot start approved command: {e}")))?;
    let process_id = child.id();
    let mut stdin_pipe = child.stdin.take();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| SurfaceError::internal("missing command stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| SurfaceError::internal("missing command stderr"))?;
    let mut stdout_task = tokio::spawn(drain_capped(stdout));
    let mut stderr_task = tokio::spawn(drain_capped(stderr));
    let execution = async {
        if let Some(input) = stdin {
            let mut pipe = stdin_pipe
                .take()
                .ok_or_else(|| SurfaceError::internal("missing command stdin"))?;
            pipe.write_all(input).await.map_err(|error| {
                SurfaceError::command(format!("cannot write command stdin: {error}"))
            })?;
        }
        drop(stdin_pipe);
        child
            .wait()
            .await
            .map_err(|error| SurfaceError::command(format!("cannot wait for command: {error}")))
    };
    let status = match timeout(Duration::from_secs(timeout_secs), execution).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            terminate_child(&mut child, process_id, &mut stdout_task, &mut stderr_task).await;
            return Err(error);
        }
        Err(_) => {
            terminate_child(&mut child, process_id, &mut stdout_task, &mut stderr_task).await;
            return Err(SurfaceError::command(format!(
                "command timed out after {timeout_secs}s"
            )));
        }
    };
    let (stdout, stdout_truncated) = stdout_task
        .await
        .map_err(|e| SurfaceError::internal(format!("stdout reader failed: {e}")))?
        .map_err(|e| SurfaceError::command(format!("cannot read stdout: {e}")))?;
    let (stderr, stderr_truncated) = stderr_task
        .await
        .map_err(|e| SurfaceError::internal(format!("stderr reader failed: {e}")))?
        .map_err(|e| SurfaceError::command(format!("cannot read stderr: {e}")))?;
    Ok(CommandOutput {
        success: status.success(),
        code: status.code(),
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        truncated: stdout_truncated || stderr_truncated,
    })
}

pub async fn git(
    cwd: &Path,
    args_input: &[&str],
    stdin: Option<&[u8]>,
    timeout_secs: u64,
) -> SurfaceResult<CommandOutput> {
    let mut args = vec![
        "-c".to_owned(),
        "core.hooksPath=/dev/null".to_owned(),
        "-c".to_owned(),
        "core.fsmonitor=false".to_owned(),
    ];
    args.extend(args_input.iter().map(|value| (*value).to_owned()));
    run_fixed(
        Path::new("/usr/bin/git"),
        &args,
        cwd,
        stdin,
        timeout_secs,
        EnvironmentProfile::Local,
    )
    .await
}

pub async fn git_network(
    cwd: &Path,
    args_input: &[&str],
    timeout_secs: u64,
) -> SurfaceResult<CommandOutput> {
    let mut args = vec![
        "-c".to_owned(),
        "core.hooksPath=/dev/null".to_owned(),
        "-c".to_owned(),
        "core.fsmonitor=false".to_owned(),
    ];
    args.extend(args_input.iter().map(|value| (*value).to_owned()));
    run_fixed(
        Path::new("/usr/bin/git"),
        &args,
        cwd,
        None,
        timeout_secs,
        EnvironmentProfile::GitNetwork,
    )
    .await
}

pub async fn gh(cwd: &Path, args: &[String], timeout_secs: u64) -> SurfaceResult<CommandOutput> {
    let candidates = [
        Path::new("/opt/homebrew/bin/gh"),
        Path::new("/usr/local/bin/gh"),
        Path::new("/usr/bin/gh"),
    ];
    let program = candidates
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| SurfaceError::command("gh executable not found in approved locations"))?;
    run_fixed(
        program,
        args,
        cwd,
        None,
        timeout_secs,
        EnvironmentProfile::Github,
    )
    .await
}

#[cfg(unix)]
const SIGKILL: i32 = 9;

#[cfg(unix)]
unsafe extern "C" {
    fn kill(process_group: i32, signal: i32) -> i32;
}
