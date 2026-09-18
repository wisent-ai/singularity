//! What the command line asks the agent to do, and the startup import it may carry.

use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::brama::BramaClient;
use crate::config::{Command, CommonArgs, CycleArgs, OutputFormat, RuntimeConfig, ToolsArgs};
use crate::error::{AppError, ErrorClass};
use crate::mcp::LasSupervisor;
use crate::most::MostClient;
use crate::tools::ToolCatalog;

pub(super) fn startup_import(
    args: &CycleArgs,
) -> Result<Option<crate::import::MindImport>, AppError> {
    if args.common.resume && args.import_file.is_some() {
        return Err(AppError::State(
            "import into an existing being with `singularity import` before starting it with --resume"
                .into(),
        ));
    }
    args.import_file
        .as_deref()
        .map(|path| {
            let document = crate::import::read_import_document(path)?;
            let input = crate::import::parse_import_bytes(&document)?;
            crate::import::validate_import(&input)?;
            Ok(input)
        })
        .transpose()
}

pub(super) fn print_startup_import(report: &crate::import::ImportReport) -> Result<(), AppError> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

pub async fn execute(command: Command, cancellation: CancellationToken) -> Result<(), AppError> {
    match command {
        Command::Run(args) => {
            let startup_import = startup_import(&args)?;
            let (mut agent, startup_report) = Agent::bootstrap_with_import(
                RuntimeConfig::from_args(&args.common)?,
                startup_import.as_ref(),
            )
            .await?;
            if let Some(report) = startup_report.as_ref() {
                print_startup_import(report)?;
            }
            let result = agent.run(cancellation).await;
            let shutdown = agent.shutdown().await;
            result.and(shutdown)
        }
        Command::Once(args) => {
            let startup_import = startup_import(&args)?;
            let (mut agent, startup_report) = Agent::bootstrap_with_import(
                RuntimeConfig::from_args(&args.common)?,
                startup_import.as_ref(),
            )
            .await?;
            if let Some(report) = startup_report.as_ref() {
                print_startup_import(report)?;
            }
            let result = agent.run_once().await;
            let shutdown = agent.shutdown().await;
            let report = result?;
            shutdown?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            match crate::onboarding::record_completed_cycle(&report).await {
                Ok(true) => println!(
                    "First-use complete: Singularity recorded autonomous_cycle_completed from the cycle above."
                ),
                Ok(false) => {}
                Err(error) => {
                    eprintln!("singularity: could not record onboarding first success: {error}");
                }
            }
            Ok(())
        }
        Command::Import(args) => {
            let report = crate::import::import_file(&args.state_dir, &args.file).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report.accepted {
                Ok(())
            } else {
                Err(AppError::State(format!(
                    "import refused: {} conflicting and {} rejected item(s); no state was changed",
                    report.conflicting, report.rejected
                )))
            }
        }
        Command::Onboarding(args) => {
            if let Some(path) = args.import_file.as_deref() {
                let report = crate::import::import_file(&args.state_dir, path).await?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                if !report.accepted {
                    return Err(AppError::State(format!(
                        "onboarding import refused: {} conflicting and {} rejected item(s); no state was changed",
                        report.conflicting, report.rejected
                    )));
                }
            }
            crate::onboarding::run_first_use(args.reset)
                .await
                .map(|_| ())
                .map_err(|error| AppError::Runtime(format!("onboarding: {error}")))
        }
        Command::Doctor(args) => doctor(&args).await,
        Command::Tools(args) => list_tools(&args).await,
    }
}

pub(super) async fn doctor(args: &CommonArgs) -> Result<(), AppError> {
    let config = RuntimeConfig::from_args(args)?;
    let brama = BramaClient::new(
        config.brama_url.clone(),
        config.brama_model.clone(),
        config.identity.agent_id.clone(),
        config.brama_secret.clone(),
        config.max_tokens,
        config.temperature,
        config.http_timeout,
    )?;
    brama.health().await?;
    let models = brama.models().await?;
    let selector = config.brama_model == "any"
        || config.brama_model == "any-vision-capable"
        || config.brama_model.starts_with("task:");
    if !selector && !models.iter().any(|model| model == &config.brama_model) {
        return Err(AppError::Config(format!(
            "configured Brama model is unavailable: {}",
            config.brama_model
        )));
    }
    let health = if let Some(token) = config.most_token.clone() {
        let most = MostClient::new(config.most_url.clone(), token, config.http_timeout)?;
        let health = most.health().await?;
        if health.backends.trim().is_empty() || health.backends == "none" {
            return Err(AppError::Most {
                class: ErrorClass::Permanent,
                message: "Most has no send-capable backend".into(),
            });
        }
        Some(health)
    } else {
        None
    };
    let mut las = LasSupervisor::spawn(
        &config.las_command,
        &config.las_entrypoint,
        &config.las_only,
        config.las_skip.as_deref(),
        Some(config.identity.agent_id.as_str()),
        &config.las_release_manifest,
        &config.las_release_manifest_signature,
        &config.las_release_trust_store,
        &config.las_release_watermark,
        &config.required_surfaces,
        config.mcp_timeout,
    )
    .await?;
    let tools = las.tools().len();
    las.shutdown(config.shutdown_grace).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"ok":true,"brama_model":config.brama_model,"most":health,"las_tools":tools})
        )?
    );
    Ok(())
}

pub(super) async fn list_tools(args: &ToolsArgs) -> Result<(), AppError> {
    if !args.las_entrypoint.is_file() {
        return Err(AppError::Config(format!(
            "LAS entrypoint not found: {}",
            args.las_entrypoint.display()
        )));
    }
    let deadline = LAS_LISTING_DEADLINE;
    let required = Vec::new();
    let mut las = LasSupervisor::spawn(
        &args.las_command,
        &args.las_entrypoint,
        &args.las_only,
        args.las_skip.as_deref(),
        args.agent_id.as_deref(),
        &args.las_release_manifest,
        &args.las_release_manifest_signature,
        &args.las_release_trust_store,
        &args.las_release_watermark,
        &required,
        deadline,
    )
    .await?;
    let catalog = ToolCatalog::build(las.tools(), false)?;
    match args.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(catalog.definitions())?),
        OutputFormat::Table => {
            for tool in catalog.definitions() {
                println!("{}\t{}", tool.function.name, tool.function.description);
            }
        }
    }
    las.shutdown(deadline).await
}
