#[path = "agent/mod.rs"]
pub mod agent;
pub mod brama;
pub mod config;
pub mod domain;
pub mod error;
#[path = "platform/mcp.rs"]
pub mod mcp;
#[path = "platform/most.rs"]
pub mod most;
#[path = "platform/tools.rs"]
pub mod tools;

pub use agent::{Agent, CycleReport};
pub use brama::BramaClient;
pub use config::{Cli, Command, RuntimeConfig};
pub use domain::{AgentState, AgentStatus, Budget, ChatMessage, ToolCall, ToolDefinition};
pub use error::{AppError, ErrorClass};
pub use mcp::LasSupervisor;
pub use most::MostClient;
pub use tools::{ToolCatalog, ToolOutcome};

#[cfg(test)]
#[path = "agent/budget_tests.rs"]
mod budget_tests;

#[cfg(test)]
#[path = "agent/brama_tests.rs"]
mod brama_tests;
