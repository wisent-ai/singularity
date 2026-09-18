//! One cycle: the model call, the tool rounds, and what the report says about them.
use std::time::Instant;

use chrono::Utc;
use serde_json::Value;

use super::*;
use crate::domain::{ActivityEvent, AgentStatus, ChatMessage, Role};
use crate::error::AppError;
use crate::tools::ToolStatus;
impl Agent {
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
}
