//! `singularity ticket launch`: one managed start, end to end.
//!
//! A ticket names a workload key Skarbiec knows and three capabilities issued
//! to that key's agent, and it expires within minutes, so it cannot be made
//! once and installed: every start makes its own. The launch writes a fresh
//! workload key, registers its PEM public half with `skarbiec grant issue`,
//! issues the three capabilities `singularity-bootstrap` redeems with
//! `skarbiec grant capability`, signs the ticket with `ticket sign`'s own
//! code (so it is checked the way the bootstrap checks it), and then runs the
//! bootstrap on it, which replaces this process with the being. Every number
//! and path is the caller's, stated by the service declaration that runs it.

use std::fs::DirBuilder;
use std::os::unix::fs::DirBuilderExt;
use std::path::Path;
use std::process::Command;

use serde_json::Value;
use uuid::Uuid;

use super::{absolute, file_digest, key, sign, KeyArgs, KeyHolder, LaunchArgs, SignArgs};
use crate::bootstrap::{run_bootstrap, BRAMA_BEARER_PURPOSE, BRAMA_PURPOSE, CAPABILITY_TARGET, MOST_PURPOSE};
use crate::error::AppError;

/// Make the workload key, its grant and capabilities and the ticket, then
/// become the being. Returns only with the step that refused.
pub fn launch(args: &LaunchArgs) -> Result<std::convert::Infallible, AppError> {
    absolute(&args.runtime_root, "--runtime-root")?;
    let ticket_dir = args.runtime_root.join(format!("ticket-{}", Uuid::new_v4()));
    DirBuilder::new()
        .recursive(true)
        .mode(u32::from(libc::S_IRWXU))
        .create(&ticket_dir)?;
    let workload_key = ticket_dir.join("workload.key");
    let workload_public = ticket_dir.join("workload.pub");
    key(&KeyArgs { holder: KeyHolder::Workload, out: workload_key.clone(), public_out: workload_public.clone() })?;
    skarbiec(
        args,
        "grant issue",
        &[
            "grant",
            "issue",
            &args.agent_id,
            "--capabilities",
            &args.grant_capabilities,
            "--ttl-seconds",
            &args.grant_ttl_seconds.to_string(),
            "--workload-public-key-file",
            text(&workload_public)?,
        ],
    )?;
    let brama_capability = capability(args, BRAMA_PURPOSE, &args.brama_resource)?;
    let brama_bearer_capability = capability(args, BRAMA_BEARER_PURPOSE, &args.brama_bearer_resource)?;
    let most_capability = capability(args, MOST_PURPOSE, &args.most_resource)?;
    let manifest = ticket_dir.join("manifest.json");
    let signature = ticket_dir.join("manifest.sig");
    sign(&SignArgs {
        agent_id: args.agent_id.clone(),
        role: args.role.clone(),
        environment: args.environment.clone(),
        host: args.host.clone(),
        workload_id: args.workload_id.clone(),
        workload_key,
        broker_socket: args.broker_socket.clone(),
        executable: args.executable.clone(),
        code_digest: file_digest(&args.executable, "--executable")?,
        policy_file: args.policy_file.clone(),
        policy_sequence: args.policy_sequence,
        expires_in_seconds: args.expires_in_seconds,
        brama_capability,
        brama_resource: args.brama_resource.clone(),
        brama_bearer_capability,
        brama_bearer_resource: args.brama_bearer_resource.clone(),
        most_capability,
        most_resource: args.most_resource.clone(),
        supervisor_key: args.supervisor_key.clone(),
        trust_root: args.trust_root.clone(),
        manifest_out: manifest.clone(),
        signature_out: signature.clone(),
        singularity_args: args.singularity_args.clone(),
    })?;
    run_bootstrap(&manifest, &signature, &args.trust_root, &args.runtime_root)
}

/// Issue one capability to the launch's agent for the bootstrap and return
/// its id.
fn capability(args: &LaunchArgs, purpose: &str, resource: &str) -> Result<String, AppError> {
    let answer = skarbiec(
        args,
        "grant capability",
        &[
            "grant",
            "capability",
            "--agent",
            &args.agent_id,
            "--purpose",
            purpose,
            "--resource",
            resource,
            "--target",
            CAPABILITY_TARGET,
            "--ttl",
            &args.capability_ttl_seconds.to_string(),
            "--max-uses",
            &args.capability_max_uses.to_string(),
        ],
    )?;
    let document: Value = serde_json::from_slice(&answer).map_err(|error| {
        AppError::Config(format!("skarbiec grant capability for {purpose} did not answer JSON: {error}"))
    })?;
    document
        .get("capability_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| AppError::Config(format!("skarbiec grant capability for {purpose} answered no capability_id: {document}")))
}

/// Run one skarbiec command and return its stdout; a refusal carries
/// Skarbiec's own sentence, which names the remedy.
fn skarbiec(args: &LaunchArgs, label: &str, words: &[&str]) -> Result<Vec<u8>, AppError> {
    let output = Command::new(&args.skarbiec)
        .args(words)
        .output()
        .map_err(|error| AppError::Config(format!("{} could not be started: {error}", args.skarbiec.display())))?;
    if !output.status.success() {
        return Err(AppError::Config(format!(
            "skarbiec {label} refused: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

fn text(path: &Path) -> Result<&str, AppError> {
    path.to_str()
        .ok_or_else(|| AppError::Config(format!("{} is not UTF-8", path.display())))
}
