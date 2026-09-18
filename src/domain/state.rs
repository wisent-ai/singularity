//! The state a being carries: its mind, its identity and the actions it recorded.
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::*;
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct MemorySource {
    pub kind: String,
    pub source_id: String,
    pub item_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub kind: String,
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub sources: Vec<MemorySource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChildRecord {
    pub id: Uuid,
    pub name: String,
    pub ticker: String,
    pub state_dir: PathBuf,
    pub created_at: DateTime<Utc>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BeingMind {
    pub system_prompt: String,
    pub rules: Vec<String>,
    pub learnings: Vec<String>,
    pub memories: Vec<MemoryEntry>,
    pub children: Vec<ChildRecord>,
    pub current_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AgentIdentity {
    pub agent_id: String,
    pub name: String,
    pub ticker: String,
    pub agent_type: String,
    pub specialty: String,
    pub role: String,
    pub environment: String,
    pub host: String,
    pub workload_id: String,
    pub workload_public_key: String,
    pub executable_digest: String,
    pub code_digest: String,
    pub policy_digest: String,
    pub policy_sequence: u64,
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
#[serde(deny_unknown_fields)]
pub struct AgentState {
    pub schema_version: String,
    pub identity: AgentIdentity,
    pub mind: BeingMind,
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
    pub fn new(identity: AgentIdentity, mind: BeingMind, budget: Budget) -> Self {
        let now = Utc::now();
        Self {
            schema_version: STATE_SCHEMA_VERSION.into(),
            identity,
            mind,
            status: AgentStatus::Starting,
            cycle: u64::default(),
            budget,
            conversation: Vec::new(),
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
        let limit = RECENT_ACTIONS_KEPT;
        if self.recent_actions.len() > limit {
            self.recent_actions
                .drain(..self.recent_actions.len() - limit);
        }
        self.updated_at = Utc::now();
    }
}
