use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "singularity-bootstrap", version)]
struct Args {
    #[arg(long, env = "SINGULARITY_BOOTSTRAP_MANIFEST")]
    manifest: PathBuf,
    #[arg(long, env = "SINGULARITY_BOOTSTRAP_MANIFEST_SIG")]
    manifest_signature: PathBuf,
    #[arg(long, env = "SINGULARITY_BOOTSTRAP_TRUST_ROOT")]
    trust_root: PathBuf,
    #[arg(long, env = "SINGULARITY_RUNTIME_ROOT")]
    runtime_root: PathBuf,
}

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("singularity-bootstrap: {error}");
            error.exit_code()
        }
    };
    if code != 0 {
        std::process::exit(code);
    }
}

fn run() -> Result<i32, singularity::AppError> {
    let args = Args::parse();
    let status = singularity::bootstrap::run_bootstrap(
        &args.manifest,
        &args.manifest_signature,
        &args.trust_root,
        &args.runtime_root,
    )?;
    Ok(status.code().unwrap_or(1))
}
