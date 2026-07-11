use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Permanent,
    Transient,
    Indeterminate,
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error("configuration: {0}")]
    Config(String),
    #[error("secret file: {0}")]
    Secret(String),
    #[error("state: {0}")]
    State(String),
    #[error("brama: {message}")]
    Brama { class: ErrorClass, message: String },
    #[error("mcp: {message}")]
    Mcp { class: ErrorClass, message: String },
    #[error("most: {message}")]
    Most { class: ErrorClass, message: String },
    #[error("tool: {0}")]
    Tool(String),
    #[error("runtime: {0}")]
    Runtime(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl AppError {
    pub fn exit_code(&self) -> i32 {
        let raw = match self {
            Self::Config(_) | Self::Secret(_) => "2",
            Self::Brama { .. } | Self::Mcp { .. } | Self::Most { .. } => "3",
            Self::State(_) | Self::Io(_) => "4",
            Self::Tool(_) | Self::Runtime(_) | Self::Json(_) => "5",
        };
        raw.parse().expect("static exit code is valid")
    }
}
