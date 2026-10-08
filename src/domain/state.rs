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

impl AgentIdentity {
    /// Take a new launch's identity for a resumed being. Who the being is
    /// (agent, persona, role, environment, host, workload) must be unchanged;
    /// what every launch issues anew (its workload key, the executable and
    /// code it runs, the policy digest) comes from the launch. A policy
    /// sequence older than the one the state last ran under is refused.
    pub fn resume_as(&mut self, launched: &AgentIdentity) -> Result<(), String> {
        let same_being = [
            ("agent id", &self.agent_id, &launched.agent_id),
            ("name", &self.name, &launched.name),
            ("ticker", &self.ticker, &launched.ticker),
            ("agent type", &self.agent_type, &launched.agent_type),
            ("specialty", &self.specialty, &launched.specialty),
            ("role", &self.role, &launched.role),
            ("environment", &self.environment, &launched.environment),
            ("host", &self.host, &launched.host),
            ("workload id", &self.workload_id, &launched.workload_id),
        ];
        let differing: Vec<String> = same_being
            .iter()
            .filter(|(_, stored, now)| stored != now)
            .map(|(field, stored, now)| format!("{field} {stored:?} is now {now:?}"))
            .collect();
        if !differing.is_empty() {
            return Err(format!(
                "resume identity does not match configuration: {}",
                differing.join("; ")
            ));
        }
        if launched.policy_sequence < self.policy_sequence {
            return Err(format!(
                "resume identity does not match configuration: policy sequence {} is older than {}, the one this state last ran under",
                launched.policy_sequence, self.policy_sequence
            ));
        }
        *self = launched.clone();
        Ok(())
    }
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

    /// Every action is kept: the state is the agent's record of what it did, and none
    /// of it is dropped by a count chosen here.
    pub fn record_action(&mut self, tool: impl Into<String>, status: impl Into<String>) {
        self.recent_actions.push(ActionRecord {
            cycle: self.cycle,
            tool: tool.into(),
            status: status.into(),
            at: Utc::now(),
        });
        self.updated_at = Utc::now();
    }
}
