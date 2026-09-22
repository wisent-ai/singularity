use super::super::{
    Shared,
    model::{Execution, Initiative, InitiativeState},
    observe,
};
use super::source;
use crate::AppError;
use chrono::Utc;
use serde_json::{Value, json};
use std::path::Path;

pub async fn advance(
    shared: &Shared,
    initiative: &mut Initiative,
    product: &Value,
    cwd: &Path,
    execution: &Execution,
) -> Result<(), AppError> {
    let (policy, response, paused) = {
        let state = shared.lock()?;
        (
            state.policy.clone(),
            state
                .store
                .get::<Value>("execution_response", &execution.id)?
                .ok_or_else(|| AppError::State("execution has no retained response".into()))?,
            state.store.paused()?,
        )
    };
    source::accepted_receipt(&response)?;
    if !policy.allow_release {
        return Err(AppError::Config(
            "release promotion is not delegated for this portfolio".into(),
        ));
    }
    let commit = execution
        .source_revision
        .as_deref()
        .ok_or_else(|| AppError::State("execution did not identify a source revision".into()))?;
    if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(AppError::State(
            "execution source revision is not a full Git commit".into(),
        ));
    }
    let manifest: Value = serde_json::from_str(
        &source::git(cwd, &["show", &format!("{commit}:.wisent-release.json")]).await?,
    )?;
    let product_id = initiative
        .product_id
        .as_deref()
        .ok_or_else(|| AppError::State("release has no product identity".into()))?;
    if manifest["product"].as_str() != Some(product_id) {
        return Err(AppError::State(
            "release manifest belongs to a different product".into(),
        ));
    }
    let version_source = &manifest["version_source"];
    if version_source["kind"] != "regex" {
        return Err(AppError::Config(
            "source release must declare a regex version_source".into(),
        ));
    }
    let file = version_source["path"]
        .as_str()
        .ok_or_else(|| AppError::State("release version source has no path".into()))?;
    let pattern = version_source["pattern"]
        .as_str()
        .ok_or_else(|| AppError::State("release version source has no pattern".into()))?;
    let version_text = source::git(cwd, &["show", &format!("{commit}:{file}")]).await?;
    let expression = regex::Regex::new(pattern)
        .map_err(|e| AppError::State(format!("release version expression: {e}")))?;
    let version = expression
        .captures(&version_text)
        .and_then(|c| c.name("version"))
        .map(|v| v.as_str().to_owned())
        .ok_or_else(|| {
            AppError::State("committed source has no declared release version".into())
        })?;
    let key = format!("release-{}", execution.request_id);
    let previous = shared
        .lock()?
        .store
        .get::<Value>("release_dispatch", &key)?;
    let cwd_text = cwd
        .to_str()
        .ok_or_else(|| AppError::Config("release source path is not UTF-8".into()))?;
    let run = if let Some(previous) = previous {
        recorded_run(product_id, &version, commit, previous["run_id"].as_str()).await?
    } else {
        if paused || !shared.admission_open()? { return Ok(()); }
        observe::command(
            "stado",
            &["release", "catalog", "sync", "--root", cwd_text, "--json"],
        )
        .await?;
        shared.lock()?.store.put("release_dispatch",&key,&json!({"product":product_id,"source_commit":commit,"version":version,"state":"dispatching"}),Some(&initiative.id),"Recorded exact release coordinate before submission")?;
        observe::command(
            "stado",
            &[
                "release",
                "submit",
                "--source",
                cwd_text,
                "--commit",
                commit,
                "--version",
                &version,
                "--channel",
                "stable",
                "--json",
            ],
        )
        .await?
    };
    if run["source_commit"].as_str() != Some(commit)
        || run["product"].as_str() != Some(product_id)
        || run["version"] != version
    {
        return Err(AppError::State(
            "Stado release result differs from the accepted source coordinate".into(),
        ));
    }
    shared.lock()?.store.put(
        "release_dispatch",
        &key,
        &run,
        Some(&initiative.id),
        "Read back Stado's exact release state",
    )?;
    initiative.state = InitiativeState::Releasing;
    match run["state"].as_str() {
        Some("completed" | "reconciled") => (),
        Some("failed" | "superseded") => {
            return Err(AppError::State(format!(
                "release did not deliver: {}",
                run["failure"]
            )));
        }
        Some(_) => return Ok(()),
        None => return Err(AppError::State("Stado release result has no state".into())),
    }
    let installations = product["installations"]
        .as_array()
        .ok_or_else(|| AppError::State("product has no installation contract".into()))?;
    if installations.is_empty() {
        return Err(AppError::State(
            "no product surface can be installed and verified".into(),
        ));
    }
    let mut observed = Vec::new();
    for installation in installations {
        let surface = installation["surface"]
            .as_str()
            .ok_or_else(|| AppError::State("installation has no surface".into()))?;
        let repository = installation["repository"]
            .as_str()
            .ok_or_else(|| AppError::State("installation has no repository".into()))?;
        let expected = response["source_revisions"][repository]
            .as_str()
            .ok_or_else(|| {
                AppError::State(format!(
                    "accepted execution did not bind {repository} to a source revision"
                ))
            })?;
        // Required Stado deliveries own installation. Never compile or install current
        // origin/main as a substitute for the independently accepted release.
        let after = observe::products(
            shared,
            &["status", product_id, "--surface", surface, "--json"],
        )
        .await?;
        if after["source_revision"].as_str() != Some(expected) || after["readiness"]["ready"] != true {
            return Err(AppError::State(format!(
                "{product_id}/{surface} did not run the accepted revision {expected}: {after}"
            )));
        }
        observed.push(after);
    }
    shared.lock()?.store.put(
        "release",
        &initiative.id,
        &json!({"observed_at":Utc::now(),"run":run,"installations":observed}),
        Some(&initiative.id),
        "Verified installed revisions; customer outcome remains unproven",
    )?;
    initiative.state = InitiativeState::Observing;
    initiative.blocked_reason = None;
    initiative.next_review_at =
        Some(Utc::now() + chrono::Duration::seconds(policy.observation_interval_seconds as i64));
    Ok(())
}

async fn recorded_run(product: &str, version: &str, commit: &str, id: Option<&str>) -> Result<Value, AppError> {
    let mut args = vec!["release", "status", product, "--json"];
    match id {
        Some(id) => args.extend(["--run", id]),
        None => args.extend(["--version", version]),
    }
    let report = observe::command("stado", &args).await?;
    let matches = report["runs"].as_array()
        .ok_or_else(|| AppError::State("release status has no run register".into()))?
        .iter().filter(|run| run["source_commit"].as_str() == Some(commit)
            && run["version"].as_str() == Some(version) && run["channel"] == "stable"
            && id.is_none_or(|id| run["run_id"].as_str() == Some(id))).collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(AppError::State("the exact release submission outcome is unresolved; no duplicate build will be submitted".into()));
    }
    Ok(matches[0].clone())
}
