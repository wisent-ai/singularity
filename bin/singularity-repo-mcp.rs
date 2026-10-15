#[path = "../src/repo_surface/mod.rs"]
mod repo_surface;
#[path = "../src/platform/mcp/stdio_server.rs"]
mod stdio_server;

use serde_json::Value;
use std::process::ExitCode;

struct RepoHost(repo_surface::RepoService);

impl stdio_server::ToolHost for RepoHost {
    fn server_name(&self) -> &'static str {
        "singularity-repo-mcp"
    }

    fn tools(&self) -> Value {
        repo_surface::tools()
    }

    async fn call(&self, name: &str, arguments: Value) -> Value {
        match self.0.call(name, arguments).await {
            Ok(value) => match stdio_server::tool_answer(&value) {
                Ok(answer) => answer,
                Err(error) => repo_surface::SurfaceError::invalid(format!(
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
    let service = match repo_surface::load_from_environment() {
        Ok(service) => service,
        Err(error) => {
            eprintln!("singularity-repo-mcp: {error}");
            return ExitCode::FAILURE;
        }
    };
    match stdio_server::serve(&RepoHost(service)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("singularity-repo-mcp: {error}");
            ExitCode::FAILURE
        }
    }
}
