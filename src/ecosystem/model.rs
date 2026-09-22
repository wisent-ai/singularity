//! Durable ecosystem records. Observations are facts; opportunities are hypotheses.
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub schema_version: u32,
    pub id: String,
    pub workspace_root: PathBuf,
    pub product_ids: Vec<String>,
    pub model: String,
    pub allow_product_creation: bool,
    pub allow_write: bool,
    pub allow_command: bool,
    pub allow_release: bool,
    pub budget_usd: Decimal,
    pub initiative_limit_usd: Decimal,
    pub exploration_budget_usd: Decimal,
    pub model_call_reserve_usd: Decimal,
    pub max_active: usize,
    pub review_interval_seconds: u64,
    pub observation_interval_seconds: u64,
    pub sources: Vec<Source>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    ProductCatalog,
    ProductAnalytics,
    MarketResearch,
    OperatorDecisions,
    FleetServices,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: String,
    pub source: Source,
    pub observed_at: DateTime<Utc>,
    pub source_revision: Option<String>,
    pub content_sha256: String,
    pub content: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpportunityKind {
    Feature,
    Product,
    Research,
    Growth,
    Maintenance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Opportunity {
    pub id: String,
    pub title: String,
    pub description: String,
    pub product_id: Option<String>,
    pub kind: OpportunityKind,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
    pub expected_outcome: String,
    pub rejection_condition: String,
    pub alternatives: Vec<String>,
    pub estimated_cost_usd: Decimal,
    pub uncertainty: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InitiativeState {
    Selected,
    Executing,
    Verifying,
    Releasing,
    Observing,
    Blocked,
    Failed,
    Completed,
    Stopped,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Initiative {
    pub id: String,
    pub opportunity_id: String,
    pub title: String,
    pub objective: String,
    pub product_id: Option<String>,
    pub state: InitiativeState,
    pub budget_usd: Decimal,
    pub spent_usd: Decimal,
    pub reserved_usd: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub next_review_at: Option<DateTime<Utc>>,
    pub blocked_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    pub id: String,
    pub initiative_id: String,
    pub request_id: String,
    pub kind: String,
    pub state: String,
    pub external_id: Option<String>,
    pub source_revision: Option<String>,
    pub evidence_refs: Vec<String>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeDecision {
    Continue,
    Change,
    Maintain,
    Stop,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub id: String,
    pub initiative_id: String,
    pub decision: OutcomeDecision,
    pub summary: String,
    pub evidence_refs: Vec<String>,
    pub measured_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issue {
    pub code: String,
    pub operation: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub accept: bool,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
    pub rejection_reasons: Vec<String>,
}
