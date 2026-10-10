use std::num::NonZeroU64;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::timeout;
use stado_wait as wait;

use super::{SurfaceError, SurfaceResult};

/// What a command answered, whole: nothing is cut from either stream.
#[derive(Debug)]
pub struct CommandOutput {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Copy)]
enum EnvironmentProfile {
    Local,
    GitNetwork,
}

async fn drain<R: AsyncRead + Unpin>(mut reader: R) -> std::io::Result<Vec<u8>> {
    let mut kept = Vec::new();
    reader.read_to_end(&mut kept).await?;
    Ok(kept)
}

type DrainTask = tokio::task::JoinHandle<std::io::Result<Vec<u8>>>;


/// Stop readers before terminating the group: descendants may retain inherited
/// pipe descriptors after the direct child exits.
async fn terminate_child(
    child: &mut tokio::process::Child,
    process_id: Option<u32>,
    stdout_task: &mut DrainTask,
    stderr_task: &mut DrainTask,
) {
    stdout_task.abort();
    stderr_task.abort();
    #[cfg(unix)]
    if let Some(process_id) = process_id {
        if unsafe { kill(-(process_id as i32), SIGKILL) } != 0 {
            eprintln!("command cleanup: killing process group {process_id} failed: {}", std::io::Error::last_os_error());
        }
    }
    if let Err(error) = wait::until(wait::Kind::Process, "terminate approved command", format!("process {process_id:?}"), child.kill()).await {
        eprintln!("command cleanup failed: {error}");
    }
    if let Err(error) = wait::until(wait::Kind::Process, "reap approved command", format!("process {process_id:?}"), child.wait()).await {
        eprintln!("command cleanup failed: {error}");
    }
}

async fn run_fixed(
    program: &Path,
    args: &[String],
    cwd: &Path,
    stdin: Option<&[u8]>,
    deadline: Option<NonZeroU64>,
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
    let mut stdout_task = tokio::spawn(drain(stdout));
    let mut stderr_task = tokio::spawn(drain(stderr));
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
        let status = wait::until(
            wait::Kind::Process,
            format!("approved command {}", program.display()),
            cwd.display(),
            child.wait(),
        ).await.map_err(|error| SurfaceError::command(format!("cannot wait for command: {error}")))?;
        let stdout = (&mut stdout_task).await
            .map_err(|e| SurfaceError::internal(format!("stdout reader failed: {e}")))?
            .map_err(|e| SurfaceError::command(format!("cannot read stdout: {e}")))?;
        let stderr = (&mut stderr_task).await
            .map_err(|e| SurfaceError::internal(format!("stderr reader failed: {e}")))?
            .map_err(|e| SurfaceError::command(format!("cannot read stderr: {e}")))?;
        Ok((status, stdout, stderr))
    };
    // The policy duration includes stdin, child exit and both output EOFs.
    let execution = wait::until(wait::Kind::Process, "command input, exit and output EOF", cwd.display(), execution);
    let finished = match deadline {
        None => Ok(execution.await),
        Some(seconds) => timeout(Duration::from_secs(seconds.get()), execution)
            .await
            .map_err(|_| seconds),
    };
    let (status, stdout, stderr) = match finished {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            terminate_child(&mut child, process_id, &mut stdout_task, &mut stderr_task).await;
            return Err(error);
        }
        Err(seconds) => {
            terminate_child(&mut child, process_id, &mut stdout_task, &mut stderr_task).await;
            return Err(SurfaceError::command(format!(
                "command exceeded the {seconds} s its policy gives input, execution and output; termination requested"
            )));
        }
    };
    Ok(CommandOutput {
        success: status.success(),
        code: status.code(),
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    })
}

fn git_arguments(args_input: &[&str]) -> Vec<String> {
    let mut args = vec![
        "-c".to_owned(),
        "core.hooksPath=/dev/null".to_owned(),
        "-c".to_owned(),
        "core.fsmonitor=false".to_owned(),
    ];
    args.extend(args_input.iter().map(|value| (*value).to_owned()));
    args
}

/// A local git command, answered when git answers.
pub async fn git(
    cwd: &Path,
    args_input: &[&str],
    stdin: Option<&[u8]>,
) -> SurfaceResult<CommandOutput> {
    run_fixed(
        Path::new("/usr/bin/git"),
        &git_arguments(args_input),
        cwd,
        stdin,
        None,
        EnvironmentProfile::Local,
    )
    .await
}

/// A local git command a policy check runs, ended after the check's own
/// `timeout_secs`.
pub async fn git_within(
    cwd: &Path,
    args_input: &[&str],
    seconds: NonZeroU64,
) -> SurfaceResult<CommandOutput> {
    run_fixed(
        Path::new("/usr/bin/git"),
        &git_arguments(args_input),
        cwd,
        None,
        Some(seconds),
        EnvironmentProfile::Local,
    )
    .await
}

/// A git command that reaches the remote, answered when git answers.
pub async fn git_network(cwd: &Path, args_input: &[&str]) -> SurfaceResult<CommandOutput> {
    run_fixed(
        Path::new("/usr/bin/git"),
        &git_arguments(args_input),
        cwd,
        None,
        None,
        EnvironmentProfile::GitNetwork,
    )
    .await
}

#[cfg(unix)]
const SIGKILL: i32 = 9;

#[cfg(unix)]
unsafe extern "C" {
    fn kill(process_group: i32, signal: i32) -> i32;
}
