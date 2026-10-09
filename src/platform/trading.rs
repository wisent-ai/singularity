//! What the Las `trading` surface needs to reach the trading platform agent
//! proxy as this being: the proxy address, the agent instance the proxy knows
//! the being as, and the file holding that instance secret.
//!
//! A managed being starts with a cleared environment, so these are being
//! arguments (`--trading-proxy-url` and the rest, each also readable from the
//! variable of the same name for an unmanaged run), and Las is handed them
//! only when its `trading` surface is selected.

use std::ffi::OsString;
use std::path::PathBuf;

use clap::Args;

use crate::error::AppError;

#[derive(Debug, Clone, Args)]
pub struct TradingArgs {
    /// The agent proxy route of the trading deployment this being belongs to
    #[arg(long, env = "TRADING_AUTONOMY_PROXY_URL")]
    pub trading_proxy_url: Option<String>,
    /// The agent instance the proxy knows this being as
    #[arg(long, env = "TRADING_AUTONOMY_INSTANCE_ID")]
    pub trading_instance_id: Option<String>,
    /// A file holding that instance secret; the secret itself is never an argument
    #[arg(long, env = "TRADING_AUTONOMY_AUTH_SECRET_FILE")]
    pub trading_auth_secret_file: Option<PathBuf>,
}

/// The three inputs, all stated.
#[derive(Debug, Clone)]
pub struct TradingSurface {
    pub proxy_url: String,
    pub instance_id: String,
    pub auth_secret_file: PathBuf,
}

impl TradingSurface {
    /// The environment the `trading` surface reads, by name.
    pub fn environment(&self) -> Vec<(&'static str, OsString)> {
        vec![
            ("TRADING_AUTONOMY_PROXY_URL", self.proxy_url.clone().into()),
            ("TRADING_AUTONOMY_INSTANCE_ID", self.instance_id.clone().into()),
            (
                "TRADING_AUTONOMY_AUTH_SECRET_FILE",
                self.auth_secret_file.clone().into_os_string(),
            ),
        ]
    }
}

impl TradingArgs {
    /// All three inputs, none of them, or a refusal naming what is missing.
    pub fn surface(&self) -> Result<Option<TradingSurface>, AppError> {
        match (
            &self.trading_proxy_url,
            &self.trading_instance_id,
            &self.trading_auth_secret_file,
        ) {
            (None, None, None) => Ok(None),
            (Some(proxy_url), Some(instance_id), Some(auth_secret_file)) => {
                Ok(Some(TradingSurface {
                    proxy_url: proxy_url.clone(),
                    instance_id: instance_id.clone(),
                    auth_secret_file: auth_secret_file.clone(),
                }))
            }
            (proxy_url, instance_id, auth_secret_file) => {
                let missing: Vec<&str> = [
                    ("--trading-proxy-url", proxy_url.is_none()),
                    ("--trading-instance-id", instance_id.is_none()),
                    ("--trading-auth-secret-file", auth_secret_file.is_none()),
                ]
                .into_iter()
                .filter_map(|(flag, absent)| absent.then_some(flag))
                .collect();
                Err(AppError::Config(format!(
                    "the trading surface takes all three of its inputs or none; missing {}",
                    missing.join(", ")
                )))
            }
        }
    }
}
