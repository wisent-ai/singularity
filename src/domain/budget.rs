//! The budget a being spends from, and what a debit or credit does to it.
use std::str::FromStr;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::*;
use crate::error::AppError;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub starting: Decimal,
    pub remaining: Decimal,
    pub api_spent: Decimal,
    pub instance_spent: Decimal,
    pub total_tokens: u64,
    pub earned: Decimal,
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
            earned: Decimal::ZERO,
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

    pub fn credit(&mut self, amount: Decimal) -> Result<(), AppError> {
        if amount.is_sign_negative() {
            return Err(AppError::State("revenue cannot be negative".into()));
        }
        self.earned += amount;
        self.remaining += amount;
        Ok(())
    }

    pub fn net_profit(&self) -> Decimal {
        self.earned - self.api_spent - self.instance_spent
    }
}
