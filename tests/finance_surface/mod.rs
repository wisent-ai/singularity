// Surface error shapes, attached to src/finance_surface/mod.rs.

use super::*;

#[test]
fn finance_contract_exposes_policy_lifecycle_and_execution() {
    let definitions = tools();
    let names: Vec<_> = definitions
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();

    assert_eq!(
        names,
        [
            "finance_propose",
            "finance_status",
            "finance_cancel",
            "finance_execute"
        ]
    );
}
