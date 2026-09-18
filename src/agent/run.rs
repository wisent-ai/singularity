//! Running until the budget or the operator stops the agent, and shutting it down.

use chrono::Utc;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::domain::{ActivityEvent, AgentStatus};
use crate::error::{AppError, ErrorClass};
use crate::state_service::StateImportService;
impl Agent {
    pub async fn run(&mut self, cancellation: CancellationToken) -> Result<(), AppError> {
        let (_state_import_service, mut import_requests) =
            StateImportService::start(&self.config.state_dir).await?;
        'cycles: while self.state.budget.can_call() && !cancellation.is_cancelled() {
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
            let wait = sleep(self.config.cycle_interval);
            tokio::pin!(wait);
            loop {
                tokio::select! {
                    _ = cancellation.cancelled() => break 'cycles,
                    _ = &mut wait => break,
                    request = import_requests.recv() => {
                        let Some(request) = request else {
                            return Err(AppError::State("local state import service stopped".into()));
                        };
                        let result = self.import_mind(&request.input);
                        request.respond(result);
                    }
                }
            }
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

    pub(super) fn report(
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
