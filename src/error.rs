use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("configuration: {0}")]
    Config(String),
    #[error("state: {0}")]
    State(String),
    #[error("jeden: {0}")]
    Jeden(String),
    #[error("runtime: {0}")]
    Runtime(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl AppError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Config(_) => 2,
            Self::Jeden(_) => 3,
            Self::State(_) | Self::Io(_) => 4,
            Self::Runtime(_) | Self::Json(_) => 5,
        }
    }
}
