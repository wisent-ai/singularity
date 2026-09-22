use super::{RunArgs, Shared, State, control, direction, execution, observe, outcomes};
use super::{model::Policy, store::Store};
use crate::{AppError, BramaClient, RuntimeConfig};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    },
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;
fn policy(args: &RunArgs, config: &RuntimeConfig) -> Result<Policy, AppError> {
    let bytes = fs::read(&args.policy)?;
    if hex::encode(Sha256::digest(&bytes)) != config.identity.policy_digest {
        return Err(AppError::Config(
            "ecosystem policy does not match the delegated identity's policy digest".into(),
        ));
    }
    let policy: Policy = serde_json::from_slice(&bytes)?;
    if policy.schema_version != Policy::SCHEMA_VERSION
        || policy.id.trim().is_empty()
        || policy.model.trim().is_empty()
        || policy.model != config.brama_model
        || policy.budget_usd <= Decimal::ZERO
        || policy.initiative_limit_usd <= Decimal::ZERO
        || policy.initiative_limit_usd > policy.budget_usd
        || policy.exploration_budget_usd < Decimal::ZERO
        || policy.exploration_budget_usd > policy.budget_usd
        || policy.model_call_reserve_usd <= Decimal::ZERO
        || policy.model_call_reserve_usd > policy.budget_usd
        || policy.max_active == 0
        || policy.review_interval_seconds == 0
        || policy.observation_interval_seconds == 0
        || policy.review_interval_seconds > i64::MAX as u64
        || policy.observation_interval_seconds > i64::MAX as u64
        || policy.sources.is_empty()
        || config.cycle_interval.is_zero()
    {
        return Err(AppError::Config(
            "ecosystem policy has an invalid version, scope, budget or observation schedule".into(),
        ));
    }
    if !policy.workspace_root.is_absolute()
        || fs::canonicalize(&policy.workspace_root)? != policy.workspace_root
    {
        return Err(AppError::Config(
            "ecosystem workspace_root must be the absolute canonical checkout root".into(),
        ));
    }
    Ok(policy)
}

pub(super) async fn run(args: RunArgs, cancellation: CancellationToken) -> Result<(), AppError> {
    let config = Arc::new(RuntimeConfig::from_args(&args.common)?);
    let policy = policy(&args, &config)?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&config.state_dir)?;
    let directory = fs::canonicalize(&config.state_dir)?;
    let metadata = fs::metadata(&directory)?;
    if metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
        || directory.starts_with(&policy.workspace_root)
    {
        return Err(AppError::Config(
            "ecosystem state must be owner-only and outside the executor's checkout root".into(),
        ));
    }
    let owner = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(directory.join("ecosystem.lock"))?;
    if unsafe { libc::flock(owner.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(AppError::State(format!(
            "another ecosystem runtime owns {}: {}",
            directory.display(),
            std::io::Error::last_os_error()
        )));
    }
    let shared = Shared(Arc::new(Mutex::new(State {
        store: Store::open(&directory, &policy, &config.identity, args.start_paused)?,
        policy,
        config: config.clone(),
        directory: directory.clone(),
    })));
    shared.lock()?.store.set_meta("las_catalog_ready", &false)?;
    direction::recover(&shared.lock()?.store)?;
    execution::recover(&shared)?;
    let _service = control::Service::start(&config.state_dir, shared.clone())?;
    if args.ready_json {
        let event = serde_json::json!({
            "schema_version":control::protocol::SCHEMA_VERSION,
            "event":"ecosystem_control_ready",
            "socket":directory.join(control::protocol::SOCKET_FILE),
            "paused":shared.lock()?.store.paused()?,
            "source_revision":option_env!("WISENT_SOURCE_COMMIT")
        });
        let mut stdout = std::io::stdout().lock();
        serde_json::to_writer(&mut stdout, &event)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    let http = reqwest::Client::builder()
        .build()
        .map_err(|e| AppError::Runtime(format!("Brama transport: {e}")))?;
    let client = Arc::new(BramaClient::from_client(
        http,
        config.brama_url.clone(),
        config.brama_model.clone(),
        config.identity.agent_id.clone(),
        config.brama_secret.clone(),
        config.brama_bearer.clone(),
        config.max_tokens,
        config.temperature,
    ));
    let mut services = tokio::task::JoinSet::new();
    services.spawn(control::monitor_tools(shared.clone(), config.clone()));
    for source in shared.lock()?.policy.sources.clone() {
        services.spawn(observe::monitor(shared.clone(), source));
    }
    {
        let (shared, client, config) = (shared.clone(), client.clone(), config.clone());
        services.spawn(async move {
            let mut cadence = tokio::time::interval(config.cycle_interval);
            cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                cadence.tick().await;
                if !shared.admission_open()? {
                    continue;
                }
                match direction::select(&shared, &client).await {
                    Err(error) => shared.failure("select", &error)?,
                    Ok(true) => shared.lock()?.store.clear_issue("select")?,
                    Ok(false) => (),
                }
                if let Err(error) = direction::materialize(&shared) {
                    shared.failure("materialize", &error)?;
                }
            }
        });
    }
    {
        let (shared, client, config) = (shared.clone(), client.clone(), config.clone());
        services.spawn(async move {
            let mut cadence = tokio::time::interval(config.cycle_interval);
            cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                cadence.tick().await;
                if !shared.admission_open()? {
                    continue;
                }
                if let Err(error) = outcomes::review(&shared, &client).await {
                    shared.failure("outcomes", &error)?;
                } else {
                    shared.lock()?.store.clear_issue("outcomes")?;
                }
            }
        });
    }
    let mut workers = tokio::task::JoinSet::new();
    let mut active = std::collections::HashMap::new();
    let mut cadence = tokio::time::interval(config.cycle_interval);
    cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = cancellation.cancelled() => break,
            Some(result) = services.join_next() => {
                return Err(AppError::Runtime(format!("ecosystem background operation stopped: {result:?}")));
            },
            Some(result) = workers.join_next(), if !workers.is_empty() => {
                match result {
                    Ok((id,result)) => { active.remove(&id); if let Err(error)=result { shared.failure(&format!("execute.{id}"),&error)?; } },
                    Err(error) => { return Err(AppError::Runtime(format!("ecosystem worker failed: {error}; its durable dispatch remains unresolved"))); }
                }
            },
            _ = cadence.tick() => {
                for id in execution::eligible(&shared)? {
                    let state = shared.lock()?;
                    if active.len() >= state.policy.max_active { break; }
                    let product = state.store.get::<super::model::Initiative>("initiative", &id)?
                        .ok_or_else(|| AppError::State(format!("admitted initiative {id} disappeared")))?.product_id;
                    if active.contains_key(&id) || active.values().any(|existing| existing == &product) { continue; }
                    drop(state);
                    active.insert(id.clone(), product);
                    let shared=shared.clone();
                    workers.spawn(async move { let result=execution::advance(&shared,&id).await; (id,result) });
                }
            }
        }
    }
    // Cancellation drops in-flight clients, not their durable request identities or reservations.
    workers.abort_all();
    services.abort_all();
    while workers.join_next().await.is_some() {}
    Ok(())
}
