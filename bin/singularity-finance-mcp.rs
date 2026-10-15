#[path = "../src/finance_surface/mod.rs"]
mod finance_surface;
#[path = "../src/platform/mcp/stdio_server.rs"]
mod stdio_server;

use serde_json::Value;
use std::path::Path;
use std::process::ExitCode;

struct FinanceHost(finance_surface::FinanceService);

impl stdio_server::ToolHost for FinanceHost {
    fn server_name(&self) -> &'static str {
        "singularity-finance-mcp"
    }

    fn tools(&self) -> Value {
        finance_surface::tools()
    }

    async fn call(&self, name: &str, arguments: Value) -> Value {
        match self.0.call(name, arguments).await {
            Ok(value) => match stdio_server::tool_answer(&value) {
                Ok(answer) => answer,
                Err(error) => finance_surface::SurfaceError::invalid(format!(
                    "tool result is not serializable: {error}"
                ))
                .tool_result(),
            },
            Err(error) => error.tool_result(),
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("singularity-finance-mcp: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let service = finance_surface::load_from_environment()?;
    let mut args = std::env::args_os();
    let _program = args.next();
    if let Some(command) = args.next() {
        if command != "owner-event" {
            return Err("unknown owner-only subcommand".into());
        }
        let path = args
            .next()
            .ok_or("owner-event requires an absolute signed event file")?;
        if args.next().is_some() {
            return Err("owner-event accepts exactly one file".into());
        }
        let path = Path::new(&path);
        if !path.is_absolute() {
            return Err("owner event file must be absolute".into());
        }
        let result = service.ingest_owner_event(path)?;
        println!("{}", serde_json::to_string(&result)?);
        return Ok(());
    }
    stdio_server::serve(&FinanceHost(service)).await?;
    Ok(())
}
