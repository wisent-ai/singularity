//! The activity log on disk and the events written to it.
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::*;
use crate::error::AppError;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActivityEvent {
    Started {
        at: DateTime<Utc>,
    },
    CycleStarted {
        at: DateTime<Utc>,
        cycle: u64,
    },
    ModelCompleted {
        at: DateTime<Utc>,
        cycle: u64,
        usage: TokenUsage,
    },
    ToolFinished {
        at: DateTime<Utc>,
        cycle: u64,
        tool: String,
        status: String,
    },
    CostDebited {
        at: DateTime<Utc>,
        cycle: u64,
        amount: Decimal,
    },
    RevenueCredited {
        at: DateTime<Utc>,
        cycle: u64,
        amount: Decimal,
        source: String,
    },
    MindImported {
        at: DateTime<Utc>,
        source_kind: String,
        source_id: String,
        imported: usize,
        attributed: usize,
        unchanged: usize,
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
        set_mode(&dir, "448")?;
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
                "unsupported state schema {}",
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
        set_mode(&self.journal_path, "384")
    }

    pub fn save(&self, state: &AgentState) -> Result<(), AppError> {
        let tmp = self.dir.join(format!(".state-{}.tmp", Uuid::new_v4()));
        {
            let mut file = OpenOptions::new().create_new(true).write(true).open(&tmp)?;
            file.write_all(&serde_json::to_vec_pretty(state)?)?;
            file.sync_all()?;
        }
        set_mode(&tmp, "384")?;
        fs::rename(&tmp, &self.state_path)?;
        set_mode(&self.state_path, "384")
    }

    pub fn state_path(&self) -> &Path {
        &self.state_path
    }
}

fn set_mode(path: &Path, decimal_mode: &str) -> Result<(), AppError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = decimal_mode
            .parse()
            .map_err(|error| AppError::State(format!("invalid file mode: {error}")))?;
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    }
    Ok(())
}
