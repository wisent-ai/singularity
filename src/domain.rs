use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::error::AppError;

pub const STATE_SCHEMA_VERSION: &str = "rust-v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(Value::String(text.into())),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }
    }

    pub fn tool(call: &ToolCall, content: Value) -> Self {
        Self {
            role: Role::Tool,
            content: Some(Value::String(content.to_string())),
            tool_call_id: Some(call.id.clone()),
            name: Some(call.function.name.clone()),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunctionDefinition,
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            kind: "function".into(),
            function: ToolFunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pricing {
    pub input_per_million: Decimal,
    pub output_per_million: Decimal,
    pub instance_per_hour: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub starting: Decimal,
    pub remaining: Decimal,
    pub api_spent: Decimal,
    pub instance_spent: Decimal,
    pub total_tokens: u64,
}

impl Budget {
    pub fn new(starting: Decimal) -> Result<Self, AppError> {
        if starting.is_sign_negative() {
            return Err(AppError::Config(
                "starting balance cannot be negative".into(),
            ));
        }
        Ok(Self {
            starting,
            remaining: starting,
            api_spent: Decimal::ZERO,
            instance_spent: Decimal::ZERO,
            total_tokens: u64::default(),
        })
    }

    pub fn can_call(&self) -> bool {
        self.remaining > Decimal::ZERO
    }

    pub fn debit(
        &mut self,
        usage: TokenUsage,
        elapsed: std::time::Duration,
        pricing: &Pricing,
    ) -> Decimal {
        let million = Decimal::from_str("1000000").expect("static decimal is valid");
        let nanos_per_hour = Decimal::from_str("3600000000000").expect("static decimal is valid");
        let api = Decimal::from(usage.prompt_tokens) * pricing.input_per_million / million
            + Decimal::from(usage.completion_tokens) * pricing.output_per_million / million;
        let instance = Decimal::from_str(&elapsed.as_nanos().to_string())
            .expect("duration is decimal")
            * pricing.instance_per_hour
            / nanos_per_hour;
        let total = api + instance;
        self.api_spent += api;
        self.instance_spent += instance;
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
        self.remaining -= total;
        total
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Exhausted,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    pub name: String,
    pub ticker: String,
    pub agent_type: String,
    pub specialty: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub cycle: u64,
    pub tool: String,
    pub status: String,
    pub at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreatedResources {
    pub chat_ids: Vec<Uuid>,
    pub message_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub schema_version: String,
    pub identity: AgentIdentity,
    pub status: AgentStatus,
    pub cycle: u64,
    pub budget: Budget,
    pub conversation: Vec<ChatMessage>,
    pub recent_actions: Vec<ActionRecord>,
    pub created_resources: CreatedResources,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AgentState {
    pub fn new(identity: AgentIdentity, budget: Budget, system_prompt: String) -> Self {
        let now = Utc::now();
        Self {
            schema_version: STATE_SCHEMA_VERSION.into(),
            identity,
            status: AgentStatus::Starting,
            cycle: u64::default(),
            budget,
            conversation: vec![ChatMessage::text(Role::System, system_prompt)],
            recent_actions: vec![],
            created_resources: CreatedResources::default(),
            started_at: now,
            updated_at: now,
        }
    }

    pub fn record_action(&mut self, tool: impl Into<String>, status: impl Into<String>) {
        self.recent_actions.push(ActionRecord {
            cycle: self.cycle,
            tool: tool.into(),
            status: status.into(),
            at: Utc::now(),
        });
        let limit: usize = "100".parse().expect("static limit is valid");
        if self.recent_actions.len() > limit {
            self.recent_actions
                .drain(..self.recent_actions.len() - limit);
        }
        self.updated_at = Utc::now();
    }
}

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
