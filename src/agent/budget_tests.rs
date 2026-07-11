use std::{str::FromStr, time::Duration};

use rust_decimal::Decimal;

use crate::{
    Budget,
    domain::{Pricing, TokenUsage},
};

fn decimal(value: &str) -> Decimal {
    Decimal::from_str(value).unwrap()
}

fn tokens(value: &str) -> u64 {
    value.parse().unwrap()
}

#[test]
fn debit_accounts_for_tokens_and_instance_time_exactly() {
    let mut budget = Budget::new(decimal("10")).unwrap();
    let pricing = Pricing {
        input_per_million: decimal("2"),
        output_per_million: decimal("6"),
        instance_per_hour: decimal("4"),
    };
    let usage = TokenUsage {
        prompt_tokens: tokens("250000"),
        completion_tokens: tokens("125000"),
        total_tokens: tokens("375000"),
    };

    let debited = budget.debit(usage, Duration::from_secs(tokens("5400")), &pricing);

    assert_eq!(debited, decimal("7.25"));
    assert_eq!(budget.api_spent, decimal("1.25"));
    assert_eq!(budget.instance_spent, decimal("6"));
    assert_eq!(budget.total_tokens, tokens("375000"));
    assert_eq!(budget.remaining, decimal("2.75"));
}

#[test]
fn a_started_call_may_overdraw_budget_but_blocks_the_next_call() {
    let mut budget = Budget::new(decimal("0.50")).unwrap();
    let pricing = Pricing {
        input_per_million: decimal("2"),
        output_per_million: decimal("6"),
        instance_per_hour: decimal("4"),
    };
    let usage = TokenUsage {
        prompt_tokens: tokens("500000"),
        completion_tokens: u64::default(),
        total_tokens: tokens("500000"),
    };

    assert!(budget.can_call());
    let debited = budget.debit(usage, Duration::from_secs(tokens("1800")), &pricing);

    assert_eq!(debited, decimal("3"));
    assert_eq!(budget.remaining, decimal("-2.50"));
    assert!(!budget.can_call());
}
