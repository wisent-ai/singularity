use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AppError;

pub const STATE_SCHEMA_VERSION: &str = "jeden-v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentIdentity {
    pub name: String,
    pub ticker: String,
    pub specialty: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Mission {
    pub goal: String,
    pub workspace: PathBuf,
    pub model: Option<String>,
    pub allow_write: bool,
    pub allow_command: bool,
    pub auto_approve: bool,
    pub max_steps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Starting,
    Running,
    Completed,
    CycleLimit,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub schema_version: String,
    pub identity: AgentIdentity,
    pub mission: Mission,
    pub status: AgentStatus,
    pub cycle: u64,
    pub max_cycles: u64,
    pub jeden_session_path: Option<PathBuf>,
    pub last_result: Option<String>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AgentState {
    pub fn new(identity: AgentIdentity, mission: Mission, max_cycles: u64) -> Self {
        let now = Utc::now();
        Self {
            schema_version: STATE_SCHEMA_VERSION.into(),
            identity,
            mission,
            status: AgentStatus::Starting,
            cycle: 0,
            max_cycles,
            jeden_session_path: None,
            last_result: None,
            started_at: now,
            updated_at: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActivityEvent {
    Started {
        at: DateTime<Utc>,
        session_path: Option<PathBuf>,
    },
    CycleStarted {
        at: DateTime<Utc>,
        cycle: u64,
    },
    JedenCompleted {
        at: DateTime<Utc>,
        cycle: u64,
        request_id: String,
        status: AgentStatus,
        session_path: PathBuf,
    },
    Warning {
        at: DateTime<Utc>,
        cycle: u64,
        message: String,
    },
    Stopped {
        at: DateTime<Utc>,
        cycle: u64,
        status: AgentStatus,
    },
}

pub struct ActivityStore {
    dir: PathBuf,
    state_path: PathBuf,
    journal_path: PathBuf,
}

impl ActivityStore {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, AppError> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        set_mode(&dir, 0o700)?;
        Ok(Self {
            state_path: dir.join("state.json"),
            journal_path: dir.join("activity.jsonl"),
            dir,
        })
    }

    pub fn load(&self) -> Result<Option<AgentState>, AppError> {
        if !self.state_path.exists() {
            return Ok(None);
        }
        let state: AgentState = serde_json::from_slice(&fs::read(&self.state_path)?)?;
        if state.schema_version != STATE_SCHEMA_VERSION {
            return Err(AppError::State(format!(
                "unsupported state schema {}; start with a new state directory",
                state.schema_version
            )));
        }
        Ok(Some(state))
    }

    pub fn append(&self, event: &ActivityEvent) -> Result<(), AppError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.journal_path)?;
        serde_json::to_writer(&mut file, event)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        set_mode(&self.journal_path, 0o600)
    }

    pub fn save(&self, state: &AgentState) -> Result<(), AppError> {
        let tmp = self.dir.join(format!(".state-{}.tmp", Uuid::new_v4()));
        {
            let mut file = OpenOptions::new().create_new(true).write(true).open(&tmp)?;
            file.write_all(&serde_json::to_vec_pretty(state)?)?;
            file.sync_all()?;
        }
        set_mode(&tmp, 0o600)?;
        fs::rename(&tmp, &self.state_path)?;
        set_mode(&self.state_path, 0o600)
    }

    pub fn state_path(&self) -> &Path {
        &self.state_path
    }
}

fn set_mode(path: &Path, mode: u32) -> Result<(), AppError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    }
    Ok(())
}
