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
            journal_path: journal_path(&dir),
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

/// The journal file of the being whose state lives in `state_dir`.
pub fn journal_path(state_dir: &Path) -> PathBuf {
    state_dir.join("activity.jsonl")
}

/// Every event of a being's activity journal, oldest first, read without
/// creating or changing anything. A journal that does not exist yet holds no
/// event; any other read failure, and any line that is not a JSON object with
/// a `type`, is refused naming the journal and the physical line, so a
/// damaged history is never shown as a shorter complete one.
pub fn read_journal(state_dir: &Path) -> Result<Vec<serde_json::Value>, AppError> {
    let path = journal_path(state_dir);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(AppError::State(format!(
                "activity journal {} cannot be read: {error}",
                path.display()
            )));
        }
    };
    let text = String::from_utf8(bytes).map_err(|error| {
        AppError::State(format!(
            "activity journal {} is not UTF-8: {error}",
            path.display()
        ))
    })?;
    // An empty file is a journal nothing was written to. Otherwise there is
    // one record per physical line: the terminator after the last record ends
    // it and is no record itself, but an empty line inside the journal is one.
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let records = text.strip_suffix('\n').unwrap_or(&text);
    records
        .split('\n')
        .enumerate()
        .map(|(index, line)| {
            let number = index + 1;
            let damaged = |reason: String| {
                AppError::State(format!(
                    "activity journal {} line {number} {reason}; the journal is damaged and no line of it was shown",
                    path.display()
                ))
            };
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|error| damaged(format!("is not JSON: {error}")))?;
            if !value.is_object() {
                return Err(damaged("is not a JSON object".into()));
            }
            if !value.get("type").is_some_and(serde_json::Value::is_string) {
                return Err(damaged("has no type".into()));
            }
            Ok(value)
        })
        .collect()
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
