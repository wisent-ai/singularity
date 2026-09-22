mod cognition;
mod discovery;
mod review;
use super::{Shared, model::*};
use crate::{AppError, BramaClient};
use chrono::Utc;
pub(super) use cognition::{ask, current, finish, recover};
use rust_decimal::Decimal;

pub fn evidence_valid(shared: &Shared, refs: &[String]) -> Result<(), AppError> {
    if refs.is_empty() {
        return Err(AppError::State(
            "decision has no observation references".into(),
        ));
    }
    let state = shared.lock()?;
    for id in refs {
        let observation: Observation = state
            .store
            .get("observation", id)?
            .ok_or_else(|| AppError::State(format!("unknown observation {id}")))?;
        if (Utc::now() - observation.observed_at).num_seconds()
            > state.policy.observation_interval_seconds as i64
        {
            return Err(AppError::State(format!(
                "observation {id} is stale; collect new evidence before selecting work"
            )));
        }
    }
    Ok(())
}

pub async fn select(shared: &Shared, client: &BramaClient) -> Result<bool, AppError> {
    // A lost review response blocks that proposal, not reviews for unrelated products.
    let pending = shared.lock()?.store.list::<Opportunity>("opportunity")?;
    for opportunity in pending
        .into_iter()
        .filter(|value| value.status == "proposed")
    {
        if !review::due(shared, &opportunity.id)? {
            continue;
        }
        let operation = format!("review.{}", opportunity.id);
        match review::review(shared, client, opportunity).await {
            Ok(()) => shared.lock()?.store.clear_issue(&operation)?,
            Err(error) => shared.failure(&operation, &error)?,
        }
    }
    let policy = {
        let state = shared.lock()?;
        let last = state
            .store
            .meta::<chrono::DateTime<Utc>>("last_selection_at")?;
        if state.store.active_count()? as u64 >= state.policy.max_active as u64
            || last.is_some_and(|at| {
                (Utc::now() - at).num_seconds() < state.policy.review_interval_seconds as i64
            })
        {
            return Ok(false);
        }
        state.store.set_meta("last_selection_at", &Utc::now())?;
        state.policy.clone()
    };
    let result = discovery::discover(shared, client, &policy).await;
    cognition::finish(shared, discovery::PURPOSE, result.as_ref().err())?;
    if let Some(opportunity) = result? {
        review::review(shared, client, opportunity).await?;
    }
    Ok(true)
}

/// Reconstruct the selected initiative deterministically after any interrupted write.
pub fn materialize(shared: &Shared) -> Result<(), AppError> {
    let state = shared.lock()?;
    for mut opportunity in state.store.list::<Opportunity>("opportunity")? {
        if opportunity.status == "rejected" {
            continue;
        }
        let Some(review) = state.store.get::<Review>("review", &opportunity.id)? else {
            continue;
        };
        if !review.accept || !review.rejection_reasons.is_empty() {
            continue;
        }
        let initiative_id = format!("initiative-{}", opportunity.id);
        if state
            .store
            .get::<Initiative>("initiative", &initiative_id)?
            .is_some()
        {
            continue;
        }
        let existing = state.store.list::<Initiative>("initiative")?;
        if existing
            .iter()
            .filter(|initiative| {
                !matches!(
                    initiative.state,
                    InitiativeState::Completed | InitiativeState::Stopped | InitiativeState::Failed
                )
            })
            .count()
            >= state.policy.max_active
        {
            break;
        }
        let admission = format!("admission.{}", opportunity.id);
        if let Some(owner) = existing.iter().find(|initiative| {
            initiative.product_id == opportunity.product_id
                && !matches!(
                    initiative.state,
                    InitiativeState::Completed | InitiativeState::Stopped | InitiativeState::Failed
                )
        }) {
            let message = format!(
                "Product scope is retained by nonterminal initiative {}; this accepted opportunity is waiting",
                owner.id
            );
            if state
                .store
                .get::<Issue>("issue", &admission)?
                .is_none_or(|previous| previous.message != message)
            {
                state.store.issue(&Issue {
                    code: "product_scope_busy".into(),
                    operation: admission,
                    message,
                    retryable: true,
                })?;
            }
            continue;
        }
        state.store.clear_issue(&admission)?;
        let mut exploration_spent = Decimal::ZERO;
        for initiative in &existing {
            let original: Opportunity = state
                .store
                .get("opportunity", &initiative.opportunity_id)?
                .ok_or_else(|| AppError::State("initiative lost its opportunity".into()))?;
            if matches!(
                original.kind,
                OpportunityKind::Product | OpportunityKind::Research
            ) {
                exploration_spent += initiative.budget_usd;
            }
        }
        if matches!(
            opportunity.kind,
            OpportunityKind::Product | OpportunityKind::Research
        ) && exploration_spent + opportunity.estimated_cost_usd
            > state.policy.exploration_budget_usd
        {
            opportunity.status = "rejected".into();
            state.store.put(
                "opportunity",
                &opportunity.id,
                &opportunity,
                None,
                "Exploration allocation exhausted",
            )?;
            continue;
        }
        let now = Utc::now();
        let initiative = Initiative {
            id: initiative_id.clone(),
            opportunity_id: opportunity.id,
            title: opportunity.title,
            objective: format!(
                "{}\nExpected outcome: {}\nReject when: {}\nEvidence: {}\nRead canonical product documentation first. Deliver reusable functionality through all applicable CLI and graphical surfaces with canonical documentation and real tests. Work only in each repository's canonical main checkout; no worktrees or copied checkouts. Commit and push the reviewed implementation. Do not promote releases in this implementation stage. New products must use Wisent Products creation, not an ad-hoc repository. A scaffold, mock, build or unverifiable claim is not completion.",
                opportunity.description,
                opportunity.expected_outcome,
                opportunity.rejection_condition,
                opportunity.evidence_refs.join(", ")
            ),
            product_id: opportunity.product_id,
            state: InitiativeState::Selected,
            budget_usd: opportunity.estimated_cost_usd,
            spent_usd: Decimal::ZERO,
            reserved_usd: Decimal::ZERO,
            created_at: now,
            updated_at: now,
            next_review_at: None,
            blocked_reason: None,
        };
        state.store.put(
            "initiative",
            &initiative_id,
            &initiative,
            Some(&initiative_id),
            "Independent review selected an outcome contract",
        )?;
    }
    Ok(())
}
