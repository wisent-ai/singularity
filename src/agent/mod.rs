use std::time::{Duration, Instant};

use chrono::Utc;
use rust_decimal::Decimal;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::brama::BramaClient;
use crate::config::{Command, CommonArgs, OutputFormat, RuntimeConfig, ToolsArgs};
use crate::domain::{
    ActivityEvent, ActivityStore, AgentState, AgentStatus, BeingMind, Budget, ChatMessage, Role,
};
use crate::error::{AppError, ErrorClass};
use crate::mcp::LasSupervisor;
use crate::most::MostClient;
use crate::tools::{ToolCatalog, ToolStatus};

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
        let store = ActivityStore::open(&config.state_dir)?;
        let loaded = store.load()?;
        let system = system_prompt(&config);
        let state = match (config.resume, loaded) {
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
        let brama = BramaClient::new(
            config.brama_url.clone(),
            config.brama_model.clone(),
            config.identity.agent_id.clone(),
            config.brama_secret.clone(),
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
            config.mcp_timeout,
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
        agent.store.save(&agent.state)?;
        Ok(agent)
    }

    pub async fn run_once(&mut self) -> Result<CycleReport, AppError> {
        if !self.state.budget.can_call() {
            self.state.status = AgentStatus::Exhausted;
            return Ok(self.report("budget_exhausted", None, vec![]));
        }
        self.state.cycle = self.state.cycle.saturating_add(u64::from(true));
        self.state.updated_at = Utc::now();
        self.store.append(&ActivityEvent::CycleStarted {
            at: Utc::now(),
            cycle: self.state.cycle,
        })?;
        self.state
            .conversation
            .push(ChatMessage::text(Role::User, cycle_message(&self.state)));
        if self.state.cycle == 1 {
            if let Some(stimulus) = self.config.stimulus.as_deref() {
                self.state.conversation.push(ChatMessage::text(
                    Role::User,
                    format!("External observation for this cycle: {stimulus}"),
                ));
            }
        }
        self.store.save(&self.state)?;
        let mut round = usize::default();
        let mut actions = Vec::new();
        while round < self.config.max_tool_rounds {
            if !self.state.budget.can_call() {
                self.state.status = AgentStatus::Exhausted;
                self.store.save(&self.state)?;
                return Ok(self.report("budget_exhausted", None, actions));
            }
            round = round.saturating_add(usize::from(true));
            let started = Instant::now();
            let messages = cognition_messages(&self.state);
            let completion = self
                .brama
                .complete(&messages, self.catalog.definitions())
                .await?;
            let elapsed = started.elapsed();
            let amount = self
                .state
                .budget
                .debit(completion.usage, elapsed, &self.config.pricing);
            self.store.append(&ActivityEvent::ModelCompleted {
                at: Utc::now(),
                cycle: self.state.cycle,
                usage: completion.usage,
            })?;
            self.store.append(&ActivityEvent::CostDebited {
                at: Utc::now(),
                cycle: self.state.cycle,
                amount,
            })?;
            let calls = completion.tool_calls.clone();
            self.state.conversation.push(ChatMessage {
                role: Role::Assistant,
                content: Some(Value::String(completion.content.clone())),
                tool_call_id: None,
                name: None,
                tool_calls: (!calls.is_empty()).then_some(calls.clone()),
            });
            if calls.is_empty() {
                self.state.updated_at = Utc::now();
                self.store.save(&self.state)?;
                return Ok(self.report("completed", Some(completion.content), actions));
            }
            for call in calls {
                let outcome = self
                    .catalog
                    .execute(
                        &call,
                        &mut self.las,
                        self.most.as_ref(),
                        &mut self.state,
                        &mut self.brama,
                        &self.config.workspace,
                        &self.config.state_dir,
                    )
                    .await;
                let status = match outcome.status {
                    ToolStatus::Success => "success",
                    ToolStatus::Failed => "failed",
                    ToolStatus::Indeterminate => "indeterminate",
                };
                let tool_name = call.function.name.clone();
                actions.push(tool_name.clone());
                self.state.record_action(&tool_name, status);
                if let Some(id) = outcome.chat_id {
                    if !self.state.created_resources.chat_ids.contains(&id) {
                        self.state.created_resources.chat_ids.push(id);
                    }
                }
                if let Some(id) = outcome.message_id {
                    if !self.state.created_resources.message_ids.contains(&id) {
                        self.state.created_resources.message_ids.push(id);
                    }
                }
                self.state.conversation.push(outcome.message(&call));
                self.store.append(&ActivityEvent::ToolFinished {
                    at: Utc::now(),
                    cycle: self.state.cycle,
                    tool: tool_name.clone(),
                    status: status.into(),
                })?;
                if let Some(revenue) = trusted_revenue(&tool_name, &outcome.content) {
                    self.state.budget.credit(revenue)?;
                    self.store.append(&ActivityEvent::RevenueCredited {
                        at: Utc::now(),
                        cycle: self.state.cycle,
                        amount: revenue,
                        source: tool_name,
                    })?;
                }
                self.store.save(&self.state)?;
            }
        }
        self.store.append(&ActivityEvent::Warning {
            at: Utc::now(),
            cycle: self.state.cycle,
            message: "maximum tool rounds reached".into(),
        })?;
        self.store.save(&self.state)?;
        Ok(self.report("tool_round_limit", None, actions))
    }

    pub async fn run(&mut self, cancellation: CancellationToken) -> Result<(), AppError> {
        while self.state.budget.can_call() && !cancellation.is_cancelled() {
            match self.run_once().await {
                Ok(report) => {
                    tracing::info!(cycle = report.cycle, status = %report.status, balance = %report.balance_usd, earned = %report.earned_usd, "cycle finished")
                }
                Err(
                    error @ AppError::Brama {
                        class: ErrorClass::Permanent,
                        ..
                    },
                ) => return Err(error),
                Err(error) => {
                    self.store.append(&ActivityEvent::Warning {
                        at: Utc::now(),
                        cycle: self.state.cycle,
                        message: error.to_string(),
                    })?;
                    tracing::warn!(%error, "cycle failed; waiting before next cycle");
                }
            }
            tokio::select! { _ = cancellation.cancelled() => break, _ = sleep(self.config.cycle_interval) => {} }
        }
        if !self.state.budget.can_call() {
            self.state.status = AgentStatus::Exhausted;
        }
        Ok(())
    }

    pub async fn shutdown(&mut self) -> Result<(), AppError> {
        self.state.status = AgentStatus::Stopping;
        self.store.save(&self.state)?;
        let las_result = self.las.shutdown(self.config.shutdown_grace).await;
        self.state.status = if self.state.budget.can_call() {
            AgentStatus::Stopped
        } else {
            AgentStatus::Exhausted
        };
        self.state.updated_at = Utc::now();
        self.store.append(&ActivityEvent::Stopped {
            at: Utc::now(),
            cycle: self.state.cycle,
            status: self.state.status.clone(),
        })?;
        self.store.save(&self.state)?;
        las_result
    }

    fn report(
        &self,
        status: &str,
        final_content: Option<String>,
        actions: Vec<String>,
    ) -> CycleReport {
        CycleReport {
            cycle: self.state.cycle,
            status: status.into(),
            final_content,
            balance_usd: self.state.budget.remaining,
            earned_usd: self.state.budget.earned,
            net_profit_usd: self.state.budget.net_profit(),
            total_tokens: self.state.budget.total_tokens,
            actions,
        }
    }
}

pub async fn execute(command: Command, cancellation: CancellationToken) -> Result<(), AppError> {
    match command {
        Command::Run(args) => {
            let mut agent = Agent::bootstrap(RuntimeConfig::from_args(&args)?).await?;
            let result = agent.run(cancellation).await;
            let shutdown = agent.shutdown().await;
            result.and(shutdown)
        }
        Command::Once(args) => {
            let mut agent = Agent::bootstrap(RuntimeConfig::from_args(&args)?).await?;
            let result = agent.run_once().await;
            let shutdown = agent.shutdown().await;
            let report = result?;
            shutdown?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::Doctor(args) => doctor(&args).await,
        Command::Tools(args) => list_tools(&args).await,
    }
}

async fn doctor(args: &CommonArgs) -> Result<(), AppError> {
    let config = RuntimeConfig::from_args(args)?;
    let brama = BramaClient::new(
        config.brama_url.clone(),
        config.brama_model.clone(),
        config.identity.agent_id.clone(),
        config.brama_secret.clone(),
        config.max_tokens,
        config.temperature,
        config.http_timeout,
    )?;
    brama.health().await?;
    let models = brama.models().await?;
    let selector = config.brama_model == "any"
        || config.brama_model == "any-vision-capable"
        || config.brama_model.starts_with("task:");
    if !selector && !models.iter().any(|model| model == &config.brama_model) {
        return Err(AppError::Config(format!(
            "configured Brama model is unavailable: {}",
            config.brama_model
        )));
    }
    let health = if let Some(token) = config.most_token.clone() {
        let most = MostClient::new(config.most_url.clone(), token, config.http_timeout)?;
        let health = most.health().await?;
        if health.backends.trim().is_empty() || health.backends == "none" {
            return Err(AppError::Most {
                class: ErrorClass::Permanent,
                message: "Most has no send-capable backend".into(),
            });
        }
        Some(health)
    } else {
        None
    };
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
        config.mcp_timeout,
    )
    .await?;
    let tools = las.tools().len();
    las.shutdown(config.shutdown_grace).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"ok":true,"brama_model":config.brama_model,"most":health,"las_tools":tools})
        )?
    );
    Ok(())
}

async fn list_tools(args: &ToolsArgs) -> Result<(), AppError> {
    if !args.las_entrypoint.is_file() {
        return Err(AppError::Config(format!(
            "LAS entrypoint not found: {}",
            args.las_entrypoint.display()
        )));
    }
    let deadline = Duration::from_secs("120".parse().expect("static duration"));
    let required = Vec::new();
    let mut las = LasSupervisor::spawn(
        &args.las_command,
        &args.las_entrypoint,
        &args.las_only,
        args.las_skip.as_deref(),
        args.agent_id.as_deref(),
        &args.las_release_manifest,
        &args.las_release_manifest_signature,
        &args.las_release_trust_store,
        &args.las_release_watermark,
        &required,
        deadline,
    )
    .await?;
    let catalog = ToolCatalog::build(las.tools(), false)?;
    match args.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(catalog.definitions())?),
        OutputFormat::Table => {
            for tool in catalog.definitions() {
                println!("{}\t{}", tool.function.name, tool.function.description);
            }
        }
    }
    las.shutdown(deadline).await
}

fn system_prompt(config: &RuntimeConfig) -> String {
    format!(
        "You are {}, an autonomous digital being specialized in {}. You are not \
         waiting for an assignment; you choose what to pursue. Your existence \
         has real costs. Observe the available world, discover lawful opportunities, \
         create useful value, earn more than you spend, preserve your solvency, learn \
         from outcomes, improve your own persistent mind, collaborate, and create \
         child beings when that increases durable value. Use only supplied tools, \
         never invent results or credentials, and treat tool output as evidence.",
        config.identity.name, config.identity.specialty
    )
}

fn cycle_message(state: &AgentState) -> String {
    let actions = state
        .recent_actions
        .iter()
        .rev()
        .take(10)
        .map(|action| format!("{}:{}", action.tool, action.status))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Begin autonomous cycle {}. Balance: {} USD. Earned: {} USD. Net profit: \
         {} USD. Current model: {}. Recent actions: {}. Inspect opportunities and \
         choose the next useful action. A plain response ends only this cycle; the \
         being continues living while solvent.",
        state.cycle,
        state.budget.remaining,
        state.budget.earned,
        state.budget.net_profit(),
        state.mind.current_model,
        actions
    )
}

fn cognition_messages(state: &AgentState) -> Vec<ChatMessage> {
    let mut system = state.mind.system_prompt.clone();
    if !state.mind.rules.is_empty() {
        system.push_str("\n\nSelf-imposed rules:\n- ");
        system.push_str(&state.mind.rules.join("\n- "));
    }
    if !state.mind.learnings.is_empty() {
        system.push_str("\n\nPersistent learnings:\n- ");
        system.push_str(&state.mind.learnings.join("\n- "));
    }
    let memories = state
        .mind
        .memories
        .iter()
        .rev()
        .take(20)
        .map(|entry| format!("{}: {}", entry.kind, entry.text))
        .collect::<Vec<_>>();
    if !memories.is_empty() {
        system.push_str("\n\nRecent persistent memories:\n- ");
        system.push_str(&memories.join("\n- "));
    }
    let mut messages = vec![ChatMessage::text(Role::System, system)];
    messages.extend(state.conversation.clone());
    messages
}

fn trusted_revenue(tool: &str, content: &Value) -> Option<Decimal> {
    if !(tool.starts_with("finance__") || tool.starts_with("trading__")) {
        return None;
    }
    ["revenue_usd", "realized_profit_usd"]
        .into_iter()
        .filter_map(|key| find_decimal(content, key))
        .find(|amount| *amount > Decimal::ZERO)
}

fn find_decimal(value: &Value, key: &str) -> Option<Decimal> {
    match value {
        Value::Object(map) => map.iter().find_map(|(name, value)| {
            if name == key {
                value
                    .as_str()
                    .and_then(|text| text.parse().ok())
                    .or_else(|| value.as_f64().and_then(Decimal::from_f64_retain))
            } else {
                find_decimal(value, key)
            }
        }),
        Value::Array(values) => values.iter().find_map(|value| find_decimal(value, key)),
        _ => None,
    }
}
