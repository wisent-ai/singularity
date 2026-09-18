#[path = "../src/finance_surface/mod.rs"]
mod finance_surface;

use serde_json::{json, Value};
use std::path::Path;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

/// One JSON-RPC line is read up to 256 KiB.
const MAX_REQUEST_BYTES: usize = 256 * 1024;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("singularity-finance-mcp: {error}");
        std::process::exit(1);
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
    serve(service).await
}

async fn serve(service: finance_surface::FinanceService) -> Result<(), Box<dyn std::error::Error>> {
    let mut lines = BufReader::new(io::stdin()).lines();
    let mut stdout = io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.len() > MAX_REQUEST_BYTES {
            write_response(
                &mut stdout,
                &rpc_error(Value::Null, -32600, "request exceeds size limit"),
            )
            .await?;
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                write_response(&mut stdout, &rpc_error(Value::Null, -32700, "parse error")).await?;
                continue;
            }
        };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let response = match request.get("method").and_then(Value::as_str) {
            Some("initialize") => {
                json!({"jsonrpc":"2.0","id":id,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"singularity-finance-mcp","version":env!("CARGO_PKG_VERSION")}}})
            }
            Some("ping") => json!({"jsonrpc":"2.0","id":id,"result":{}}),
            Some("tools/list") => {
                json!({"jsonrpc":"2.0","id":id,"result":{"tools":finance_surface::tools()}})
            }
            Some("tools/call") => {
                let Some(params) = request.get("params") else {
                    write_response(
                        &mut stdout,
                        &rpc_error(id, -32602, "tools/call requires params"),
                    )
                    .await?;
                    continue;
                };
                let name = params.get("name").and_then(Value::as_str);
                let Some(arguments) = params.get("arguments").cloned() else {
                    write_response(
                        &mut stdout,
                        &rpc_error(id, -32602, "tools/call requires an arguments object"),
                    )
                    .await?;
                    continue;
                };
                let result = match name {
                    Some(name) => match service.call(name, arguments).await {
                        Ok(value) => match serde_json::to_string(&value) {
                            Ok(text) => {
                                json!({"content":[{"type":"text","text":text}],"structuredContent":value,"isError":false})
                            }
                            Err(error) => finance_surface::SurfaceError::invalid(format!(
                                "tool result is not serializable: {error}"
                            ))
                            .tool_result(),
                        },
                        Err(error) => error.tool_result(),
                    },
                    None => {
                        finance_surface::SurfaceError::invalid("tools/call requires a string name")
                            .tool_result()
                    }
                };
                json!({"jsonrpc":"2.0","id":id,"result":result})
            }
            Some(_) => rpc_error(id, -32601, "method not found"),
            None => rpc_error(id, -32600, "invalid request"),
        };
        write_response(&mut stdout, &response).await?;
    }
    Ok(())
}
async fn write_response(stdout: &mut io::Stdout, response: &Value) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(response).map_err(io::Error::other)?;
    bytes.push(b'\n');
    stdout.write_all(&bytes).await?;
    stdout.flush().await
}
fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
}
