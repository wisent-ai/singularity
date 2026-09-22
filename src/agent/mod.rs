use std::time::Duration;

use chrono::Utc;
use rust_decimal::Decimal;
use serde::Serialize;

use crate::brama::BramaClient;
use crate::config::RuntimeConfig;
use crate::domain::{ActivityEvent, ActivityStore, AgentState, AgentStatus, BeingMind, Budget};
use crate::error::AppError;
use crate::mcp::LasSupervisor;
use crate::most::MostClient;
use crate::tools::ToolCatalog;

/// Tool listing's child shutdown grace; MCP requests themselves wait for completion.
const LAS_LISTING_DEADLINE: Duration = Duration::from_secs(120);

#[derive(Debug, Serialize)]
pub struct CycleReport {
    pub cycle: u64,
    pub status: String,
    pub final_content: Option<String>,
    pub balance_usd: Decimal,
    pub earned_usd: Decimal,
    pub net_profit_usd: Decimal,
    pub total_tokens: u64,
    pub actions: Vec<String>,
}

pub struct Agent {
    config: RuntimeConfig,
    state: AgentState,
    store: ActivityStore,
    brama: BramaClient,
    las: LasSupervisor,
    most: Option<MostClient>,
    catalog: ToolCatalog,
}

impl Agent {
    pub async fn bootstrap(config: RuntimeConfig) -> Result<Self, AppError> {
        Self::bootstrap_with_import(config, None)
            .await
            .map(|(agent, _)| agent)
    }

    async fn bootstrap_with_import(
        config: RuntimeConfig,
        startup_import: Option<&crate::import::MindImport>,
    ) -> Result<(Self, Option<crate::import::ImportReport>), AppError> {
        let store = ActivityStore::open(&config.state_dir)?;
        let loaded = store.load()?;
        let system = system_prompt(&config);
        let mut state = match (config.resume, loaded) {
            (true, Some(state)) => {
                if state.identity != config.identity {
                    return Err(AppError::State(
                        "resume identity does not match configuration".into(),
                    ));
                }
                state
            }
            (true, None) => {
                return Err(AppError::State(
                    "resume requested but no state exists".into(),
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
                BeingMind {
                    system_prompt: system,
                    rules: Vec::new(),
                    learnings: Vec::new(),
                    memories: Vec::new(),
                    children: Vec::new(),
                    current_model: config.brama_model.clone(),
                },
                Budget::new(config.starting_balance)?,
            ),
        };
        let startup_report = if let Some(input) = startup_import {
            let (imported, report) = crate::import::apply_import(state, input)?;
            if !report.accepted {
                return Err(AppError::State(format!(
                    "startup import refused: {} conflicting and {} rejected item(s); no state was created",
                    report.conflicting, report.rejected
                )));
            }
            state = imported;
            Some(report)
        } else {
            None
        };
        let brama = BramaClient::new(
            config.brama_url.clone(),
            config.brama_model.clone(),
            config.identity.agent_id.clone(),
            config.brama_secret.clone(),
            config.brama_bearer.clone(),
            config.max_tokens,
            config.temperature,
            config.http_timeout,
        )?;
        let mut las = LasSupervisor::spawn(
            &config.las_command,
            &config.las_entrypoint,
            &config.las_only,
            config.las_skip.as_deref(),
            Some(config.identity.agent_id.as_str()),
            &config.las_release_manifest,
            &config.las_release_manifest_signature,
            &config.las_release_trust_store,
            &config.las_release_watermark,
            &config.required_surfaces,
        )
        .await?;
        let catalog = match ToolCatalog::build(las.tools(), config.most_token.is_some()) {
            Ok(value) => value,
            Err(error) => {
                let _ = las.shutdown(config.shutdown_grace).await;
                return Err(error);
            }
        };
        let most = config
            .most_token
            .clone()
            .map(|token| MostClient::new(config.most_url.clone(), token, config.http_timeout))
            .transpose()?;
        let mut agent = Self {
            config,
            state,
            store,
            brama,
            las,
            most,
            catalog,
        };
        agent
            .brama
            .set_model(agent.state.mind.current_model.clone());
        agent.state.status = AgentStatus::Running;
        agent
            .store
            .append(&ActivityEvent::Started { at: Utc::now() })?;
        if let Some(report) = startup_report.as_ref() {
            agent.store.append(&ActivityEvent::MindImported {
                at: Utc::now(),
                source_kind: report.source_kind.clone(),
                source_id: report.source_id.clone(),
                imported: report.imported,
                attributed: report.attributed,
                unchanged: report.unchanged,
            })?;
        }
        agent.store.save(&agent.state)?;
        Ok((agent, startup_report))
    }

    pub(super) fn import_mind(
        &mut self,
        input: &crate::import::MindImport,
    ) -> Result<crate::import::ImportReport, AppError> {
        let (state, report) = crate::import::apply_import(self.state.clone(), input)?;
        if report.accepted {
            self.store.save(&state)?;
            self.store.append(&ActivityEvent::MindImported {
                at: Utc::now(),
                source_kind: report.source_kind.clone(),
                source_id: report.source_id.clone(),
                imported: report.imported,
                attributed: report.attributed,
                unchanged: report.unchanged,
            })?;
            self.state = state;
        }
        Ok(report)
    }
}

mod command;
mod cycle;
mod prompts;
mod run;

pub use command::execute;
use prompts::{cognition_messages, cycle_message, system_prompt, trusted_revenue};
