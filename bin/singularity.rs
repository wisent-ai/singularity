use clap::Parser;
use singularity::config::Cli;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

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
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
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
