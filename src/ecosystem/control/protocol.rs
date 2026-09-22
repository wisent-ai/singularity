//! Wire compatibility and bounded local-control messages shared by CLI and Desktop.
pub const SCHEMA_VERSION: u32 = 2;
pub const SOCKET_FILE: &str = "ecosystem.sock";
pub const MAX_REQUEST_BYTES: u64 = 64 * 1024;
pub const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
pub const DEFAULT_PAGE_SIZE: u32 = 50;
pub const MAX_PAGE_SIZE: u32 = 100;
pub const DEFAULT_RECORD_BYTES: u32 = 64 * 1024;
pub const MAX_RECORD_BYTES: u32 = 256 * 1024;
