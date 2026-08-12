use std::path::PathBuf;
use std::process::Stdio;

use chrono::Utc;
use serde::Serialize;
use tokio::process::Command as ProcessCommand;
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use crate::config::{Command, CommonArgs, JedenArgs, RuntimeConfig, validate_jeden_args};
use crate::domain::{ActivityEvent, ActivityStore, AgentState, AgentStatus};
use crate::error::AppError;
use crate::jeden::{JedenRpc, SessionHandle};

const COMPLETE_MARKER: &str = "SINGULARITY_STATUS: COMPLETE";
const CONTINUE_MARKER: &str = "SINGULARITY_STATUS: CONTINUE";

#[derive(Debug, Serialize)]
pub struct CycleReport {
    pub goal: String,
    pub cycle: u64,
    pub status: AgentStatus,
    pub final_content: Option<String>,
    pub jeden_session_path: Option<PathBuf>,
}

pub struct Agent {
    config: RuntimeConfig,
    state: AgentState,
    store: ActivityStore,
    rpc: JedenRpc,
    session: SessionHandle,
}

impl Agent {
    pub async fn bootstrap(args: &CommonArgs) -> Result<Self, AppError> {
        let store = ActivityStore::open(&args.state_dir)?;
        let loaded = store.load()?;
        let mut config = RuntimeConfig::from_args(args, None)?;
        let state = match (args.resume, loaded) {
            (true, Some(state)) => {
                if state.identity != config.identity {
                    return Err(AppError::State(
                        "resume identity does not exactly match configuration".into(),
                    ));
                }
                if state.mission != config.mission {
                    return Err(AppError::State(
                        "resume mission does not exactly match configuration".into(),
                    ));
                }
                if state.max_cycles != config.max_cycles {
                    return Err(AppError::State(
                        "resume cycle budget does not exactly match configuration".into(),
                    ));
                }
                let path = state.jeden_session_path.clone().ok_or_else(|| {
                    AppError::State("resume state has no Jeden session path".into())
                })?;
                config.resume_session = Some(path);
                state
            }
            (true, None) => {
                return Err(AppError::State(
                    "resume requested but no Singularity state exists".into(),
                ));
            }
            (false, Some(_)) => {
                return Err(AppError::State(format!(
                    "state already exists at {}; use --resume or a new directory",
                    store.state_path().display()
                )));
            }
            (false, None) => AgentState::new(
                config.identity.clone(),
                config.mission.clone(),
                config.max_cycles,
            ),
        };
        let mut rpc =
            JedenRpc::spawn(&config.jeden_command, &config.workspace, config.rpc_timeout).await?;
        let session = rpc.create_session(&config).await?;
        let mut agent = Self {
            config,
            state,
            store,
            rpc,
            session,
        };
        agent.state.jeden_session_path = Some(agent.session.path.clone());
        agent.state.status = AgentStatus::Running;
        agent.state.updated_at = Utc::now();
        agent.store.append(&ActivityEvent::Started {
            at: Utc::now(),
            session_path: Some(agent.session.path.clone()),
        })?;
        agent.store.save(&agent.state)?;
        Ok(agent)
    }

    pub async fn run_once(&mut self) -> Result<CycleReport, AppError> {
        if self.state.status == AgentStatus::Completed {
            return Ok(self.report());
        }
        if self.state.cycle >= self.state.max_cycles {
            self.state.status = AgentStatus::CycleLimit;
            self.state.updated_at = Utc::now();
            self.store.save(&self.state)?;
            return Ok(self.report());
        }
        self.state.cycle = self.state.cycle.saturating_add(1);
        self.state.status = AgentStatus::Running;
        self.state.updated_at = Utc::now();
        self.store.append(&ActivityEvent::CycleStarted {
            at: Utc::now(),
            cycle: self.state.cycle,
        })?;
        self.store.save(&self.state)?;

        let request_id = format!("singularity-cycle-{}", self.state.cycle);
        let prompt = cycle_prompt(self.state.cycle, self.state.max_cycles);
        let result = match self
            .rpc
            .prompt(
                &self.session.id,
                &request_id,
                &prompt,
                &self.state.mission.goal,
            )
            .await
        {
            Ok(result) => result,
            Err(error) => {
                self.state.status = AgentStatus::Failed;
                self.state.updated_at = Utc::now();
                self.store.append(&ActivityEvent::Warning {
                    at: Utc::now(),
                    cycle: self.state.cycle,
                    message: error.to_string(),
                })?;
                self.store.save(&self.state)?;
                return Err(error);
            }
        };
        if result.request_id != request_id {
            return Err(AppError::Jeden(format!(
                "prompt result request id mismatch: expected {request_id}, got {}",
                result.request_id
            )));
        }
        let (status, content) = parse_result(&result.text)?;
        self.session.path = result.session_path.clone();
        self.state.jeden_session_path = Some(result.session_path.clone());
        self.state.last_result = Some(content);
        self.state.status = status;
        self.state.updated_at = Utc::now();
        self.store.append(&ActivityEvent::JedenCompleted {
            at: Utc::now(),
            cycle: self.state.cycle,
            request_id,
            status: self.state.status.clone(),
            session_path: result.session_path,
        })?;
        self.store.save(&self.state)?;
        Ok(self.report())
    }

    pub async fn run(&mut self, cancellation: CancellationToken) -> Result<(), AppError> {
        while self.state.status != AgentStatus::Completed
            && self.state.cycle < self.state.max_cycles
            && !cancellation.is_cancelled()
        {
            self.run_once().await?;
            if self.state.status == AgentStatus::Completed {
                break;
            }
            tokio::select! {
                _ = cancellation.cancelled() => break,
                _ = sleep(self.config.cycle_interval) => {}
            }
        }
        if self.state.status != AgentStatus::Completed && self.state.cycle >= self.state.max_cycles
        {
            self.state.status = AgentStatus::CycleLimit;
            self.state.updated_at = Utc::now();
            self.store.save(&self.state)?;
        }
        Ok(())
    }

    pub async fn shutdown(mut self) -> Result<(), AppError> {
        if self.state.status == AgentStatus::Running {
            self.state.status = AgentStatus::Stopping;
            self.state.updated_at = Utc::now();
            self.store.save(&self.state)?;
        }
        let rpc_result = self.rpc.shutdown().await;
        if self.state.status == AgentStatus::Stopping {
            self.state.status = AgentStatus::Stopped;
        }
        self.state.updated_at = Utc::now();
        self.store.append(&ActivityEvent::Stopped {
            at: Utc::now(),
            cycle: self.state.cycle,
            status: self.state.status.clone(),
        })?;
        self.store.save(&self.state)?;
        rpc_result
    }

    fn report(&self) -> CycleReport {
        CycleReport {
            goal: self.state.mission.goal.clone(),
            cycle: self.state.cycle,
            status: self.state.status.clone(),
            final_content: self.state.last_result.clone(),
            jeden_session_path: self.state.jeden_session_path.clone(),
        }
    }
}

pub async fn execute(command: Command, cancellation: CancellationToken) -> Result<(), AppError> {
    match command {
        Command::Run(args) => {
            let mut agent = Agent::bootstrap(&args).await?;
            let result = agent.run(cancellation).await;
            let shutdown = agent.shutdown().await;
            result.and(shutdown)
        }
        Command::Once(args) => {
            let mut agent = Agent::bootstrap(&args).await?;
            let result = agent.run_once().await;
            let shutdown = agent.shutdown().await;
            let report = result?;
            shutdown?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::Doctor(args) => delegate_command(&args, "doctor").await,
        Command::Tools(args) => delegate_command(&args, "tools").await,
    }
}

async fn delegate_command(args: &JedenArgs, subcommand: &str) -> Result<(), AppError> {
    let workspace = validate_jeden_args(args)?;
    let output = timeout(
        std::time::Duration::from_secs(args.rpc_timeout_secs),
        ProcessCommand::new(&args.jeden_command)
            .arg(subcommand)
            .arg("--json")
            .arg("--cwd")
            .arg(&workspace)
            .current_dir(&workspace)
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .output(),
    )
    .await
    .map_err(|_| AppError::Jeden(format!("jeden {subcommand} deadline exceeded")))?
    .map_err(|error| AppError::Jeden(format!("failed to start Jeden: {error}")))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(AppError::Jeden(format!(
            "jeden {subcommand} exited with {}: {detail}",
            output.status
        )));
    }
    print!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

fn cycle_prompt(cycle: u64, max_cycles: u64) -> String {
    format!(
        "Continue the immutable autonomous mission using Jeden's complete approved tool harness. This is Singularity cycle {cycle} of {max_cycles}. Inspect the existing Jeden session before acting, perform all reachable work, and preserve product ownership: Brama for models, Las for federated Wisent tools, Skarbiec for credentials, Stado for placement, Tama for policy, Probierz for evidence, Most for communication, and each child surface for its own behavior. Do not invent tool results or broaden authority. End the final response with exactly one standalone line: {COMPLETE_MARKER} when the mission is fully complete, otherwise {CONTINUE_MARKER}."
    )
}

fn parse_result(text: &str) -> Result<(AgentStatus, String), AppError> {
    let mut lines = text.lines().collect::<Vec<_>>();
    let marker = lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .ok_or_else(|| AppError::Jeden("Jeden returned an empty final response".into()))?;
    let status = match lines[marker].trim() {
        COMPLETE_MARKER => AgentStatus::Completed,
        CONTINUE_MARKER => AgentStatus::Running,
        _ => {
            return Err(AppError::Jeden(format!(
                "Jeden final response omitted the required status marker ({COMPLETE_MARKER} or {CONTINUE_MARKER})"
            )));
        }
    };
    lines.remove(marker);
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
    Ok((status, lines.join("\n")))
}
