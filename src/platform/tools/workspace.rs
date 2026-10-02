//! The workspace tools: reading and writing a file inside the agent workspace, and spawning a child.
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde_json::{Map, Value, json};
use tokio::process::Command;
use uuid::Uuid;

use super::*;
use crate::domain::{AgentState, ChildRecord};

pub(super) fn relative_path(arguments: &Map<String, Value>) -> Result<PathBuf, ToolOutcome> {
    let value = required_text(arguments, "path", 4 * 1024)?;
    let path = PathBuf::from(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(failed(
            "invalid_path",
            "path must contain only relative normal components",
        ));
    }
    Ok(path)
}

pub(super) fn file_read(workspace: &Path, arguments: Map<String, Value>) -> ToolOutcome {
    let relative = match relative_path(&arguments) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let path = workspace.join(&relative);
    let resolved = match std::fs::canonicalize(&path) {
        Ok(value) if value.starts_with(workspace) => value,
        Ok(_) => return failed("workspace_boundary", "path leaves the workspace"),
        Err(error) => return failed("file_read", &error.to_string()),
    };
    let metadata = match std::fs::symlink_metadata(&resolved) {
        Ok(value) if value.is_file() && value.len() <= MAX_WORKSPACE_FILE_BYTES => value,
        Ok(_) => return failed("file_read", "path is not a bounded regular file"),
        Err(error) => return failed("file_read", &error.to_string()),
    };
    let _ = metadata;
    match std::fs::read_to_string(resolved) {
        Ok(content) => success(json!({"path":relative,"content":content}), None, None),
        Err(error) => failed("file_read", &error.to_string()),
    }
}

pub(super) fn file_write(workspace: &Path, arguments: Map<String, Value>) -> ToolOutcome {
    let relative = match relative_path(&arguments) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let content = match arguments.get("content").and_then(Value::as_str) {
        Some(value) if value.len() as u64 <= MAX_WORKSPACE_FILE_BYTES && !value.contains('\0') => {
            value
        }
        _ => return failed("invalid_arguments", "content is invalid or too large"),
    };
    let path = workspace.join(&relative);
    let Some(parent) = path.parent() else {
        return failed("invalid_path", "path has no parent");
    };
    if let Err(error) = std::fs::create_dir_all(parent) {
        return failed("file_write", &error.to_string());
    }
    let resolved_parent = match std::fs::canonicalize(parent) {
        Ok(value) if value.starts_with(workspace) => value,
        Ok(_) => return failed("workspace_boundary", "path leaves the workspace"),
        Err(error) => return failed("file_write", &error.to_string()),
    };
    let Some(name) = path.file_name() else {
        return failed("invalid_path", "path has no file name");
    };
    let destination = resolved_parent.join(name);
    if destination
        .symlink_metadata()
        .is_ok_and(|metadata| metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return failed("file_write", "destination is not a regular file");
    }
    let temporary = resolved_parent.join(format!(".singularity-{}.tmp", Uuid::new_v4()));
    let result =
        std::fs::write(&temporary, content).and_then(|_| std::fs::rename(&temporary, &destination));
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return failed("file_write", &error.to_string());
    }
    success(json!({"path":relative,"bytes":content.len()}), None, None)
}

/// A child is a full being started by its parent, so it is handed everything
/// the parent was declared with — the Brama address and model, the Las
/// program, entrypoint and surfaces, and Most — explicitly, never by
/// inheriting whatever environment this process happened to start in.
pub(super) async fn spawn_child(
    state: &mut AgentState,
    config: &crate::config::RuntimeConfig,
    most: Option<&crate::most::MostClient>,
    arguments: Map<String, Value>,
) -> ToolOutcome {
    let state_dir = config.state_dir.as_path();
    let brama_url = &config.brama_url;
    let name = match required_text(&arguments, "name", 128) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let ticker = match required_text(&arguments, "ticker", 32) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let specialty = match required_text(&arguments, "specialty", 256) {
        Ok(value) => value,
        Err(error) => return error,
    };
    let id = Uuid::new_v4();
    let child_state = state_dir.join("children").join(id.to_string());
    if let Err(error) = std::fs::create_dir_all(&child_state) {
        return failed("child_state", &error.to_string());
    }
    let executable = match std::env::current_exe() {
        Ok(value) => value,
        Err(error) => return failed("child_executable", &error.to_string()),
    };
    let mut command = Command::new(executable);
    command
        .arg("run")
        .env("SINGULARITY_AGENT_NAME", &name)
        .env("SINGULARITY_AGENT_TICKER", &ticker)
        .env("SINGULARITY_SPECIALTY", &specialty)
        .env("SINGULARITY_STATE_DIR", &child_state)
        .env("SINGULARITY_RESUME", "false")
        .env("BRAMA_BASE_URL", brama_url.as_str())
        .env("BRAMA_MODEL", &config.brama_model)
        .env("LAS_COMMAND", &config.las_command)
        .env("LAS_MCP_ENTRYPOINT", &config.las_entrypoint)
        .env("LAS_ONLY", &config.las_only)
        .env("LAS_RELEASE_MANIFEST_FILE", &config.las_release_manifest)
        .env("LAS_RELEASE_MANIFEST_SIGNATURE_FILE", &config.las_release_manifest_signature)
        .env("LAS_RELEASE_TRUST_STORE_FILE", &config.las_release_trust_store)
        .env("LAS_RELEASE_WATERMARK_FILE", &config.las_release_watermark);
    if let Some(skip) = config.las_skip.as_deref() {
        command.env("LAS_SKIP", skip);
    }
    if let Some(most) = most {
        command.env("MOST_BASE_URL", most.base_url().as_str());
    }
    if let Err(error) = crate::bootstrap::inherit_for_child(command.as_std_mut()) {
        return failed("child_credentials", &error.to_string());
    }
    let spawned = command.spawn();
    match spawned {
        Ok(child) => {
            state.mind.children.push(ChildRecord {
                id,
                name,
                ticker,
                state_dir: child_state,
                created_at: Utc::now(),
                status: "running".into(),
            });
            success(json!({"child_id":id,"pid":child.id()}), None, None)
        }
        Err(error) => failed("child_spawn", &error.to_string()),
    }
}
