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
    ActivityEvent, ActivityStore, AgentState, AgentStatus, Budget, ChatMessage, Role,
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
    pub remaining_usd: Decimal,
    pub total_tokens: u64,
    pub actions: Vec<String>,
}

pub struct Agent {
    config: RuntimeConfig,
    state: AgentState,
    store: ActivityStore,
    brama: BramaClient,
    las: LasSupervisor,
    most: MostClient,
    catalog: ToolCatalog,
}

impl Agent {
    pub async fn bootstrap(config: RuntimeConfig) -> Result<Self, AppError> {
        let store = ActivityStore::open(&config.state_dir)?;
        let loaded = store.load()?;
        let system = system_prompt(&config);
        let state = match (config.resume, loaded) {
            (true, Some(state)) => {
                if state.identity.ticker != config.identity.ticker {
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
                Budget::new(config.starting_balance)?,
                system,
            ),
        };
        let brama = BramaClient::new(
            config.brama_url.clone(),
            config.brama_model.clone(),
            config.brama_agent_id.clone(),
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
            &config.required_surfaces,
            config.mcp_timeout,
        )
        .await?;
        let catalog = match ToolCatalog::build(las.tools()) {
            Ok(value) => value,
            Err(error) => {
                let _ = las.shutdown(config.shutdown_grace).await;
                return Err(error);
            }
        };
        let most = MostClient::new(
            config.most_url.clone(),
            config.most_token.clone(),
            config.http_timeout,
        )?;
        let mut agent = Self {
            config,
            state,
            store,
            brama,
            las,
            most,
            catalog,
        };
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
            let completion = self
                .brama
                .complete(&self.state.conversation, self.catalog.definitions())
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
                let outcome = self.catalog.execute(&call, &mut self.las, &self.most).await;
                let status = match outcome.status {
                    ToolStatus::Success => "success",
                    ToolStatus::Failed => "failed",
                    ToolStatus::Indeterminate => "indeterminate",
                };
                actions.push(call.function.name.clone());
                self.state.record_action(&call.function.name, status);
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
                    tool: call.function.name,
                    status: status.into(),
                })?;
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
                    tracing::info!(cycle = report.cycle, status = %report.status, remaining = %report.remaining_usd, "cycle finished")
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
            remaining_usd: self.state.budget.remaining,
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
        config.brama_agent_id.clone(),
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
    let most = MostClient::new(
        config.most_url.clone(),
        config.most_token.clone(),
        config.http_timeout,
    )?;
    let health = most.health().await?;
    if health.backends.trim().is_empty() || health.backends == "none" {
        return Err(AppError::Most {
            class: ErrorClass::Permanent,
            message: "Most has no send-capable backend".into(),
        });
    }
    let mut las = LasSupervisor::spawn(
        &config.las_command,
        &config.las_entrypoint,
        &config.las_only,
        config.las_skip.as_deref(),
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
        &required,
        deadline,
    )
    .await?;
    let catalog = ToolCatalog::build(las.tools())?;
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
        "You are {}, an autonomous Wisent agent specialized in {}. Use only the native tools supplied with each request. Execute one purposeful step at a time, inspect tool results, and stop when the current objective is complete. Your remaining budget is finite; never invent tool results or credentials.",
        config.identity.name, config.identity.specialty
    )
}

fn cycle_message(state: &AgentState) -> String {
    let actions = state
        .recent_actions
        .iter()
        .rev()
        .take("10".parse().expect("static limit"))
        .map(|action| format!("{}:{}", action.tool, action.status))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Begin cycle {}. Remaining budget: {} USD. Recent actions: {}. Choose the next action using a supplied tool, or respond with a final status when no tool is needed.",
        state.cycle, state.budget.remaining, actions
    )
}
