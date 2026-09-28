//! `singularity capability-preflight`: the fail-closed check a
//! capability-isolated unit runs before it starts, and, with `--exec`, the
//! launcher that replaces itself with the checked process.
//!
//! The broker is the host's Skarbiec serving only its capability socket; an
//! agent is `singularity-bootstrap` under a dedicated UID. Each is started only
//! when its environment file, the files it names, their owners and modes, and
//! the release binary's digest all match the deployment contract. A refusal
//! prints `capability-preflight: <reason>` and exits 78 (EX_CONFIG).

mod contract;
mod deployment;
mod environment;

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

use clap::{Args, ValueEnum};

use crate::AppError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PreflightKind {
    /// The Skarbiec capability broker under systemd (Linux only)
    BrokerLinux,
    /// The broker under launchd: always refused, launchd has no egress sandbox
    BrokerMacos,
    /// A capability-isolated agent running singularity-bootstrap
    Agent,
    /// The unit and environment templates in deploy/capabilities
    DeploymentStatic,
}

#[derive(Debug, Clone, Args)]
pub struct PreflightArgs {
    pub kind: PreflightKind,
    /// The unit's environment file; for deployment-static, the
    /// deploy/capabilities directory
    pub path: Option<PathBuf>,
    /// After the checks pass, replace this process with the checked one
    #[arg(long = "exec")]
    pub launch: bool,
}

/// The launched process is given a fixed search path and the C locale.
const LAUNCH_PATH: &str = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin";
const LAUNCH_LOCALE: &str = "C.UTF-8";
/// EX_CONFIG: the deployment's configuration is what is wrong.
const CONFIGURATION_EXIT_CODE: i32 = 78;

pub fn run(args: PreflightArgs) -> Result<(), AppError> {
    if let Err(reason) = check(&args) {
        eprintln!("capability-preflight: {reason}");
        std::process::exit(CONFIGURATION_EXIT_CODE);
    }
    Ok(())
}

fn check(args: &PreflightArgs) -> contract::Check<()> {
    let path = args.path.as_deref();
    let (values, program, arguments): (environment::Values, String, Vec<&str>) = match args.kind {
        PreflightKind::DeploymentStatic => {
            if args.launch {
                return Err("--exec is invalid for deployment-static".into());
            }
            return deployment::validate(
                path.ok_or("deployment-static needs the deploy/capabilities directory")?,
            );
        }
        PreflightKind::BrokerMacos => {
            return Err(if cfg!(target_os = "macos") {
                "launchd has no egress sandbox; broker startup requires an externally enforced deny-all sandbox"
            } else {
                "broker-macos is only valid on macOS"
            }
            .into());
        }
        PreflightKind::BrokerLinux => {
            if !cfg!(target_os = "linux") {
                return Err("broker-linux requires Linux systemd egress enforcement".into());
            }
            let values = environment::load_env(path.ok_or("environment file is required")?)?;
            environment::validate_broker(&values)?;
            // The host's one Skarbiec process, serving only its capability
            // socket: this deployment allows the broker no TCP.
            let program = values["SKARBIEC_BINARY"].clone();
            (values, program, vec!["serve", "--no-http"])
        }
        PreflightKind::Agent => {
            let values = environment::load_env(path.ok_or("environment file is required")?)?;
            environment::validate_agent(&values)?;
            let program = values["SINGULARITY_BOOTSTRAP_BINARY"].clone();
            (values, program, Vec::new())
        }
    };
    if !args.launch {
        return Ok(());
    }
    let error = Command::new(&program)
        .args(arguments)
        .env_clear()
        .envs(&values)
        .env("PATH", LAUNCH_PATH)
        .env("LANG", LAUNCH_LOCALE)
        .env("LC_ALL", LAUNCH_LOCALE)
        .exec();
    Err(format!("{program} could not be started: {error}"))
}
