pub const STATE_SCHEMA_VERSION: &str = "being-v1";
/// The state keeps the last hundred actions.
const RECENT_ACTIONS_KEPT: usize = 100;

mod activity;
mod budget;
mod conversation;
mod state;

pub use activity::*;
pub use budget::*;
pub use conversation::*;
pub use state::*;
