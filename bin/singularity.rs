use clap::Parser;
use singularity::config::Cli;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

/// The log filter the agent runs under when `RUST_LOG` says nothing. A `RUST_LOG` that is set
/// and unparsable is a refusal: the operator asked for a filter the subscriber cannot honour.
const DEFAULT_LOG_FILTER: &str = "info";

fn log_filter() -> Result<EnvFilter, singularity::AppError> {
    match std::env::var("RUST_LOG") {
        Ok(value) => EnvFilter::try_new(&value).map_err(|error| {
            singularity::AppError::Config(format!("RUST_LOG is not a valid filter: {error}"))
        }),
        Err(std::env::VarError::NotPresent) => Ok(EnvFilter::new(DEFAULT_LOG_FILTER)),
        Err(error) => Err(singularity::AppError::Config(format!(
            "RUST_LOG is not readable: {error}"
        ))),
    }
}

fn main() {
    let code = match run() {
        Ok(()) => i32::default(),
        Err(error) => {
            eprintln!("singularity: {error}");
            error.exit_code()
        }
    };
    if code != i32::default() {
        std::process::exit(code);
    }
}

fn run() -> Result<(), singularity::AppError> {
    singularity::bootstrap::adopt_credentials()?;
    tracing_subscriber::fmt()
        .with_env_filter(log_filter()?)
        .init();
    let cli = Cli::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let cancellation = CancellationToken::new();
        let signal = cancellation.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                signal.cancel();
            }
        });
        singularity::agent::execute(cli.command, cancellation).await
    })
}
