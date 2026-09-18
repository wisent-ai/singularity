use std::fs;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::{Path, PathBuf};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::error::AppError;
use crate::import::{ImportReport, MindImport, parse_import_bytes};

const SOCKET_FILE: &str = "state-import.sock";
const WIRE_VERSION: u32 = 1;
const MAX_WIRE_BYTES: u64 = 24 * 1024 * 1024;
const MAX_RESPONSE_BYTES: u64 = 1024 * 1024;

pub struct StateImportRequest {
    pub input: MindImport,
    response: oneshot::Sender<Result<ImportReport, String>>,
}

impl StateImportRequest {
    pub fn respond(self, result: Result<ImportReport, AppError>) {
        let _ = self.response.send(result.map_err(|error| error.to_string()));
    }
}

pub struct StateImportService {
    path: PathBuf,
    task: JoinHandle<()>,
}

impl StateImportService {
    pub async fn start(
        state_dir: &Path,
    ) -> Result<(Self, mpsc::Receiver<StateImportRequest>), AppError> {
        let path = socket_path(state_dir);
        prepare_socket_path(&path).await?;
        let listener = UnixListener::bind(&path).map_err(|error| {
            AppError::State(format!("cannot bind local state import service: {error}"))
        })?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        let (sender, receiver) = mpsc::channel(32);
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let sender = sender.clone();
                tokio::spawn(async move {
                    let _ = serve(stream, sender).await;
                });
            }
        });
        Ok((Self { path, task }, receiver))
    }
}

impl Drop for StateImportService {
    fn drop(&mut self) {
        self.task.abort();
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRequest {
    version: u32,
    operation: String,
    document_base64: String,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum WireResponse {
    Success { ok: bool, result: ImportReport },
    Failure { ok: bool, error: String },
}

pub async fn import_through_service(
    state_dir: &Path,
    document: &[u8],
) -> Result<Option<ImportReport>, AppError> {
    let path = socket_path(state_dir);
    if !path.exists() {
        return Ok(None);
    }
    let mut stream = UnixStream::connect(&path).await.map_err(|error| {
        AppError::State(format!(
            "local state import service at {} is unavailable: {error}",
            path.display()
        ))
    })?;
    let request = WireRequest {
        version: WIRE_VERSION,
        operation: "mind_import".into(),
        document_base64: STANDARD.encode(document),
    };
    let bytes = serde_json::to_vec(&request)?;
    if bytes.len() as u64 > MAX_WIRE_BYTES {
        return Err(AppError::State("state import request is too large".into()));
    }
    stream.write_all(&bytes).await?;
    stream.shutdown().await?;
    let mut response = Vec::new();
    stream
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut response)
        .await?;
    if response.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(AppError::State(
            "local state import service response is too large".into(),
        ));
    }
    match serde_json::from_slice::<WireResponse>(&response)
        .map_err(|error| AppError::State(format!("invalid local state import response: {error}")))?
    {
        WireResponse::Success { ok: true, result } => Ok(Some(result)),
        WireResponse::Failure { ok: false, error } => Err(AppError::State(error)),
        _ => Err(AppError::State(
            "invalid local state import response status".into(),
        )),
    }
}

async fn serve(
    mut stream: UnixStream,
    sender: mpsc::Sender<StateImportRequest>,
) -> Result<(), AppError> {
    let mut bytes = Vec::new();
    (&mut stream)
        .take(MAX_WIRE_BYTES + 1)
        .read_to_end(&mut bytes)
        .await?;
    let response = if bytes.len() as u64 > MAX_WIRE_BYTES {
        WireResponse::Failure {
            ok: false,
            error: "state import request is too large".into(),
        }
    } else {
        handle_request(&bytes, sender).await
    };
    stream.write_all(&serde_json::to_vec(&response)?).await?;
    stream.shutdown().await?;
    Ok(())
}

async fn handle_request(
    bytes: &[u8],
    sender: mpsc::Sender<StateImportRequest>,
) -> WireResponse {
    let result = async {
        let request: WireRequest = serde_json::from_slice(bytes)
            .map_err(|error| AppError::State(format!("invalid state import request: {error}")))?;
        if request.version != WIRE_VERSION || request.operation != "mind_import" {
            return Err(AppError::State(
                "unsupported local state import operation".into(),
            ));
        }
        let document = STANDARD
            .decode(request.document_base64)
            .map_err(|_| AppError::State("state import document is not valid base64".into()))?;
        let input = parse_import_bytes(&document)?;
        let (response, receive) = oneshot::channel();
        sender
            .send(StateImportRequest { input, response })
            .await
            .map_err(|_| AppError::State("the being stopped before importing".into()))?;
        receive
            .await
            .map_err(|_| AppError::State("the being stopped before returning the import result".into()))?
            .map_err(AppError::State)
    }
    .await;
    match result {
        Ok(result) => WireResponse::Success {
            ok: true,
            result,
        },
        Err(error) => WireResponse::Failure {
            ok: false,
            error: error.to_string(),
        },
    }
}

fn socket_path(state_dir: &Path) -> PathBuf {
    state_dir.join(SOCKET_FILE)
}

async fn prepare_socket_path(path: &Path) -> Result<(), AppError> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if !metadata.file_type().is_socket() {
        return Err(AppError::State(format!(
            "local state import path is not a socket: {}",
            path.display()
        )));
    }
    if UnixStream::connect(path).await.is_ok() {
        return Err(AppError::State(
            "another Singularity runtime already owns the local state import service".into(),
        ));
    }
    fs::remove_file(path)?;
    Ok(())
}
