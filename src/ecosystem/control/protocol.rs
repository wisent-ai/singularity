//! Wire compatibility for local-control messages shared by CLI and Desktop: one
//! newline-terminated JSON document each way. A listing holds every record unless the caller
//! names a `limit`, and a record is read whole unless the caller names `bytes`.
pub const SCHEMA_VERSION: u32 = 2;
pub const SOCKET_FILE: &str = "ecosystem.sock";
