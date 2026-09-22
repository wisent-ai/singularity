use super::evidence::{BINARY, Evidence, Result};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    os::unix::{fs::OpenOptionsExt, process::CommandExt},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    thread::JoinHandle,
};

pub struct Owner<'a> {
    pub evidence: &'a mut Evidence,
    pub state: PathBuf,
    policy: PathBuf,
    child: Option<Child>,
    reader: Option<JoinHandle<Result<()>>>,
    attempt: usize,
    arguments: Vec<String>,
    stdout: PathBuf,
    stderr: PathBuf,
}

impl<'a> Owner<'a> {
    pub fn new(evidence: &'a mut Evidence, policy: PathBuf) -> Self {
        Self {
            state: evidence.root.join("state"),
            evidence,
            policy,
            child: None,
            reader: None,
            attempt: 0,
            arguments: Vec::new(),
            stdout: PathBuf::new(),
            stderr: PathBuf::new(),
        }
    }

    pub fn start(&mut self, start_paused: bool) -> Result<()> {
        if self.child.is_some() {
            return Err("test owner is already running".into());
        }
        self.attempt += 1;
        self.stdout = self
            .evidence
            .root
            .join(format!("owner-{}.stdout", self.attempt));
        self.stderr = self
            .evidence
            .root
            .join(format!("owner-{}.stderr", self.attempt));
        self.arguments = vec![
            "ecosystem".into(),
            "run".into(),
            "--policy".into(),
            self.policy.display().to_string(),
            "--state-dir".into(),
            "state".into(),
            "--agent-name".into(),
            "Kontrola — Żółw".into(),
            "--ready-json".into(),
        ];
        if start_paused {
            self.arguments.push("--start-paused".into());
        }
        let log = log_file(&self.stdout)?;
        self.child = Some(
            Command::new(BINARY)
                .args(&self.arguments)
                .process_group(0)
                .current_dir(&self.evidence.root)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(log_file(&self.stderr)?)
                .spawn()
                .map_err(|e| format!("start real owner: {e}"))?,
        );
        let stream = self
            .child
            .as_mut()
            .ok_or("owner missing")?
            .stdout
            .take()
            .ok_or("owner stdout missing")?;
        let (sender, receiver) = mpsc::channel();
        self.reader = Some(std::thread::spawn(move || read_output(stream, log, sender)));
        match receiver.recv() {
            Ok(value) => self
                .evidence
                .snapshot(&format!("control_readiness_{}", self.attempt), value),
            Err(error) => {
                self.stop()?;
                let stderr = fs::read_to_string(&self.stderr).map_err(|e| e.to_string())?;
                Err(format!(
                    "owner did not emit control readiness: {error}; stderr={stderr}"
                ))
            }
        }
    }

    pub fn cli(&mut self, arguments: &[&str], success: bool) -> Result<Value> {
        let mut argv = vec!["ecosystem".to_owned()];
        argv.extend(arguments.iter().map(|value| (*value).to_owned()));
        argv.extend(["--state-dir".into(), "state".into(), "--json".into()]);
        let output = self.evidence.command(&argv)?;
        let value: Value = serde_json::from_slice(&output.stdout).map_err(|e| {
            format!(
                "decode CLI response: {e}; stderr={}",
                String::from_utf8_lossy(&output.stderr)
            )
        })?;
        if output.status.success() != success
            || value["ok"].as_bool() != Some(success)
            || value["schema_version"] != 2
        {
            return Err(format!(
                "unexpected CLI result for {arguments:?}: status={}, envelope={value}",
                output.status
            ));
        }
        if !success
            && (!value["error"]["operation"].is_string() || !value["error"]["message"].is_string())
        {
            return Err(format!(
                "refusal omitted the failed operation or cause: {value}"
            ));
        }
        Ok(value)
    }

    pub fn stop(&mut self) -> Result<()> {
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        let status = match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => status,
            None => {
                let pid = i32::try_from(child.id()).map_err(|e| e.to_string())?;
                // This unreaped child was created by the driver and owns its process group.
                if unsafe { libc::kill(-pid, libc::SIGKILL) } != 0 {
                    let error = std::io::Error::last_os_error();
                    if error.raw_os_error() != Some(libc::ESRCH) {
                        return Err(format!("interrupt test-owned group {pid}: {error}"));
                    }
                }
                child.wait().map_err(|e| e.to_string())?
            }
        };
        self.child = None;
        let reader = self
            .reader
            .take()
            .map(|reader| {
                reader
                    .join()
                    .unwrap_or_else(|_| Err("owner output reader panicked".into()))
            })
            .unwrap_or(Ok(()));
        self.evidence
            .owner_exit(&self.arguments, &self.stdout, &self.stderr, status)?;
        reader
    }
}

impl Drop for Owner<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("test owner cleanup failed: {error}");
        }
    }
}

fn log_file(path: &Path) -> Result<fs::File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))
}

fn read_output(
    stream: std::process::ChildStdout,
    mut log: fs::File,
    sender: mpsc::Sender<Value>,
) -> Result<()> {
    let mut sender = Some(sender);
    for line in BufReader::new(stream).split(b'\n') {
        let line = line.map_err(|e| e.to_string())?;
        log.write_all(&line)
            .and_then(|_| log.write_all(b"\n"))
            .map_err(|e| e.to_string())?;
        if sender.is_some() {
            if let Ok(value) = serde_json::from_slice::<Value>(&line) {
                if value["event"] == json!("ecosystem_control_ready") {
                    sender
                        .take()
                        .ok_or("readiness sender missing")?
                        .send(value)
                        .map_err(|e| e.to_string())?;
                }
            }
        }
    }
    log.sync_all().map_err(|e| e.to_string())
}
