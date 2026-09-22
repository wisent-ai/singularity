mod dependencies;
pub(super) mod protocol;
use super::{
    Shared,
    protocol::{MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, SCHEMA_VERSION, SOCKET_FILE},
};
use crate::AppError;
pub(super) use dependencies::monitor as monitor_tools;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::{FileTypeExt, PermissionsExt},
    path::{Path, PathBuf},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    task::JoinHandle,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    method: String,
    #[serde(default)]
    params: Params,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    id: Option<String>,
    kind: Option<String>,
    initiative_id: Option<String>,
    before: Option<i64>,
    limit: Option<u32>,
    offset: Option<u64>,
    bytes: Option<u32>,
    revision: Option<String>,
}

fn dispatch(shared: &Shared, request: Request) -> Result<Value, AppError> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(AppError::Config(format!(
            "unsupported ecosystem request version {}; expected {SCHEMA_VERSION}",
            request.schema_version
        )));
    }
    let state = shared.lock()?;
    let params = &request.params;
    match request.method.as_str() {
        "status" => state.store.status(&state.policy),
        "opportunities" => state.store.items(
            "opportunity",
            params.before,
            params.limit.unwrap_or(protocol::DEFAULT_PAGE_SIZE),
        ),
        "initiatives" => state.store.items(
            "initiative",
            params.before,
            params.limit.unwrap_or(protocol::DEFAULT_PAGE_SIZE),
        ),
        "records" => state.store.records(
            params.kind.as_deref(),
            params.initiative_id.as_deref(),
            params.before,
            params.limit.unwrap_or(protocol::DEFAULT_PAGE_SIZE),
        ),
        "record" => state.store.record(
            params
                .kind
                .as_deref()
                .ok_or_else(|| AppError::Config("record requires kind".into()))?,
            params
                .id
                .as_deref()
                .ok_or_else(|| AppError::Config("record requires id".into()))?,
            params.offset.unwrap_or(0),
            params.bytes.unwrap_or(protocol::DEFAULT_RECORD_BYTES),
            params.revision.as_deref(),
        ),
        "explain" => state.store.explain(
            request
                .params
                .id
                .as_deref()
                .ok_or_else(|| AppError::Config("explain requires id".into()))?,
        ),
        "pause" | "resume" => {
            state
                .store
                .set_meta("paused", &(request.method == "pause"))?;
            state.store.put(
                "control",
                &uuid::Uuid::new_v4().to_string(),
                &json!({"method":request.method}),
                None,
                &request.method,
            )?;
            state.store.status(&state.policy)
        }
        _ => Err(AppError::Config(format!(
            "unknown ecosystem method {}",
            request.method
        ))),
    }
}

async fn serve(mut stream: UnixStream, shared: Shared) -> Result<(), AppError> {
    if stream.peer_cred()?.uid() != unsafe { libc::geteuid() } {
        return Err(AppError::Config(
            "ecosystem control requires the state owner's identity".into(),
        ));
    }
    let mut bytes = Vec::new();
    BufReader::new((&mut stream).take(MAX_REQUEST_BYTES + 1))
        .read_until(b'\n', &mut bytes)
        .await?;
    let mut operation = "ecosystem.control".to_string();
    let result = if bytes.len() as u64 > MAX_REQUEST_BYTES || !bytes.ends_with(b"\n") {
        Err(AppError::Config(
            "ecosystem request must be one bounded newline-terminated JSON document".into(),
        ))
    } else {
        match serde_json::from_slice::<Request>(&bytes) {
            Ok(request) => {
                operation = format!("ecosystem.{}", request.method);
                dispatch(&shared, request)
            }
            Err(error) => Err(AppError::from(error)),
        }
    };
    let response = match result {
        Ok(result) => json!({"schema_version":SCHEMA_VERSION,"ok":true,"result":result}),
        Err(error) => {
            json!({"schema_version":SCHEMA_VERSION,"ok":false,"error":{"code":"control_refused","operation":operation,"message":error.to_string(),"retryable":false}})
        }
    };
    let mut body = serde_json::to_vec(&response)?;
    if body.len() as u64 > MAX_RESPONSE_BYTES {
        body = serde_json::to_vec(&json!({
            "schema_version":SCHEMA_VERSION,"ok":false,
            "error":{"code":"response_too_large","operation":operation,
                "message":format!("ecosystem response is {} bytes; protocol limit is {MAX_RESPONSE_BYTES} bytes",body.len()),
                "retryable":false}
        }))?;
    }
    body.push(b'\n');
    stream.write_all(&body).await?;
    stream.shutdown().await?;
    Ok(())
}

pub struct Service {
    path: PathBuf,
    task: JoinHandle<()>,
}
impl Service {
    /// The caller holds the exclusive process lock before reclaiming a stale socket.
    pub fn start(directory: &Path, shared: Shared) -> Result<Self, AppError> {
        let path = directory.join(SOCKET_FILE);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_socket() => fs::remove_file(&path)?,
            Ok(_) => {
                return Err(AppError::State(format!(
                    "{} is not a socket",
                    path.display()
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
        }
        let listener = UnixListener::bind(&path).map_err(|error| {
            AppError::State(format!(
                "bind ecosystem control socket {}: {error}",
                path.display()
            ))
        })?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        let task = tokio::spawn(async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    result = listener.accept() => match result {
                        Ok((stream,_)) => { let shared = shared.clone(); connections.spawn(async move { serve(stream,shared).await }); },
                        Err(error) => { tracing::error!("ecosystem socket accept failed: {error}"); break; }
                    },
                    Some(result) = connections.join_next(), if !connections.is_empty() => {
                        match result {
                            Ok(Ok(())) => (),
                            Ok(Err(error)) => tracing::warn!("ecosystem request failed: {error}"),
                            Err(error) => tracing::warn!("ecosystem connection failed: {error}"),
                        }
                    }
                }
            }
        });
        Ok(Self { path, task })
    }
}
impl Drop for Service {
    fn drop(&mut self) {
        self.task.abort();
        let _ = fs::remove_file(&self.path);
    }
}

pub async fn request(directory: &Path, method: &str, params: &Value) -> Result<Value, AppError> {
    let path = directory.join(SOCKET_FILE);
    let mut stream = UnixStream::connect(&path).await.map_err(|e| {
        AppError::State(format!(
            "ecosystem owner unavailable at {}: {e}; no cached state is presented as live",
            path.display()
        ))
    })?;
    if stream.peer_cred()?.uid() != unsafe { libc::geteuid() } {
        return Err(AppError::Config(
            "ecosystem owner socket belongs to another Unix principal".into(),
        ));
    }
    let mut bytes = serde_json::to_vec(
        &json!({"schema_version":SCHEMA_VERSION,"method":method,"params":params}),
    )?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_REQUEST_BYTES {
        return Err(AppError::Config(format!(
            "ecosystem request exceeds {MAX_REQUEST_BYTES} bytes"
        )));
    }
    stream.write_all(&bytes).await?;
    let mut reply = Vec::new();
    BufReader::new(stream.take(MAX_RESPONSE_BYTES + 1))
        .read_until(b'\n', &mut reply)
        .await?;
    if reply.len() as u64 > MAX_RESPONSE_BYTES || !reply.ends_with(b"\n") {
        return Err(AppError::State("invalid ecosystem response framing".into()));
    }
    let result: Value = serde_json::from_slice(&reply)?;
    if result["schema_version"] != SCHEMA_VERSION || !result["ok"].is_boolean() {
        return Err(AppError::State("invalid ecosystem response schema".into()));
    }
    Ok(result)
}
