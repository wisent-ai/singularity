//! Resume the real runtime with a signed Las release and real declared credentials.
//! This qualifies initialization/persistence, not ticket issuance or a model cycle.
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use singularity::{Agent, Cli, Command, RuntimeConfig};
use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};
use uuid::Uuid;

type Outcome<T> = Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Launches {
    initial: Vec<String>,
    renewed: Vec<String>,
    rollback: Vec<String>,
    changed_identity: Vec<String>,
}

fn require(condition: bool, cause: impl Into<String>) -> Outcome<()> {
    if condition {
        Ok(())
    } else {
        Err(cause.into().into())
    }
}

fn configuration(arguments: &[String], root: &Path, resume: bool) -> Outcome<RuntimeConfig> {
    let parsed = Cli::try_parse_from(
        std::iter::once("singularity".to_owned()).chain(arguments.iter().cloned()),
    )?;
    let Command::Once(mut args) = parsed.command else {
        return Err("each fixture must declare the normal once command arguments".into());
    };
    require(
        args.import_file.is_none(),
        "startup import is not part of this resume fixture",
    )?;
    args.common.state_dir = root.join("state");
    args.common.workspace = root.join("workspace");
    args.common.las_release_watermark = root.join("las-watermark.json");
    args.common.resume = resume;
    RuntimeConfig::from_args(&args.common).map_err(Into::into)
}

fn state(root: &Path) -> Outcome<Value> {
    Ok(serde_json::from_slice(&fs::read(
        root.join("state/state.json"),
    )?)?)
}

fn persistent_life(mut state: Value) -> Outcome<Value> {
    let object = state
        .as_object_mut()
        .ok_or("persisted state must be an object")?;
    for field in ["identity", "status", "updated_at"] {
        require(
            object.remove(field).is_some(),
            format!("persisted state lacks {field}"),
        )?;
    }
    Ok(state)
}

async fn bootstrap_and_stop(config: RuntimeConfig) -> Outcome<()> {
    let mut agent = Agent::bootstrap(config).await?;
    agent.shutdown().await?;
    Ok(())
}

async fn refused(config: RuntimeConfig, expected: &str) -> Outcome<String> {
    match Agent::bootstrap(config).await {
        Err(error) => {
            let cause = error.to_string();
            require(
                cause.contains(expected),
                format!("wrong resume refusal: {cause}"),
            )?;
            Ok(cause)
        }
        Ok(mut agent) => {
            let cleanup = agent.shutdown().await;
            Err(format!("invalid resume was accepted; shutdown: {cleanup:?}").into())
        }
    }
}

#[tokio::test]
#[ignore = "requires real signed Las release, launch declarations and credential handoff"]
async fn renewed_launch_preserves_life_and_refuses_identity_or_policy_rollback() -> Outcome<()> {
    let revision = option_env!("WISENT_SOURCE_COMMIT").ok_or("build with WISENT_SOURCE_COMMIT")?;
    require(
        revision == env::var("SINGULARITY_TEST_SOURCE_REVISION")?,
        "compiled source revision mismatch",
    )?;
    require(
        env::var("SINGULARITY_TEST_DEPENDENCY_PURPOSE")? == "isolated-qualification",
        "dedicated Las and credential fixture required",
    )?;
    let fixture = PathBuf::from(env::var("SINGULARITY_TEST_RESUME_LAUNCHES")?);
    let launches: Launches = serde_json::from_slice(&fs::read(&fixture)?)?;
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".build/bootstrap-resume")
        .join(Uuid::new_v4().to_string());
    fs::create_dir_all(directory.join("workspace"))?;
    let mut report = json!({"source_revision": revision, "fixture": fixture, "status": "failed"});
    let outcome: Outcome<()> = async {
        let first = configuration(&launches.initial, &directory, false)?;
        let next = configuration(&launches.renewed, &directory, true)?;
        let rollback = configuration(&launches.rollback, &directory, true)?;
        let changed = configuration(&launches.changed_identity, &directory, true)?;
        require(
            first.identity.workload_public_key != next.identity.workload_public_key,
            "renewed fixture must carry a different real launch key",
        )?;
        require(
            first.identity.executable_digest != next.identity.executable_digest,
            "renewed fixture must carry a different real executable digest",
        )?;
        require(
            next.identity.policy_sequence > first.identity.policy_sequence,
            "renewed fixture must advance the declared policy sequence",
        )?;
        let mut checked = first.identity.clone();
        checked.resume_as(&next.identity)?;
        require(
            rollback.identity.policy_sequence < next.identity.policy_sequence,
            "rollback fixture must carry an older declared policy sequence",
        )?;
        let mut rollback_identity = next.identity.clone();
        rollback_identity.policy_sequence = rollback.identity.policy_sequence;
        rollback_identity.resume_as(&rollback.identity)?;
        require(
            changed.identity.policy_sequence >= next.identity.policy_sequence,
            "changed-identity refusal must not be a policy rollback",
        )?;
        require(
            next.identity.agent_id != changed.identity.agent_id,
            "changed-identity fixture must name a different being",
        )?;
        let next_identity = serde_json::to_value(&next.identity)?;
        bootstrap_and_stop(first).await?;
        let before = state(&directory)?;
        fs::write(
            directory.join("before.json"),
            serde_json::to_vec_pretty(&before)?,
        )?;
        bootstrap_and_stop(next).await?;
        let after = state(&directory)?;
        require(
            after.get("identity") == Some(&next_identity),
            "renewed identity was not persisted",
        )?;
        require(
            persistent_life(before.clone())? == persistent_life(after.clone())?,
            "resume changed the existing mind, budget, conversation, actions or creation identity",
        )?;
        fs::write(
            directory.join("after.json"),
            serde_json::to_vec_pretty(&after)?,
        )?;
        let state_path = directory.join("state/state.json");
        let accepted_bytes = fs::read(&state_path)?;
        report["rollback"] = json!(refused(rollback, "policy sequence").await?);
        require(
            fs::read(&state_path)? == accepted_bytes,
            "rollback refusal changed state bytes",
        )?;
        report["changed_identity"] = json!(refused(changed, "agent id").await?);
        require(
            fs::read(&state_path)? == accepted_bytes,
            "identity refusal changed state bytes",
        )?;
        report["accepted_identity"] = next_identity;
        Ok(())
    }
    .await;
    report["error"] = json!(outcome.as_ref().err().map(ToString::to_string));
    if outcome.is_ok() {
        report["status"] = json!("passed");
    }
    fs::write(
        directory.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", directory.join("report.json").display());
    outcome
}
