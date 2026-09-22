//! Wire compatibility and bounded local-control messages shared by CLI and Desktop.
pub const SCHEMA_VERSION: u32 = 1;
pub const SOCKET_FILE: &str = "ecosystem.sock";
pub const MAX_REQUEST_BYTES: u64 = 64 * 1024;
pub const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
