mod policy;
mod service;
mod state;

use serde_json::{Value, json};
use std::fmt;
use std::path::PathBuf;

pub use policy::PolicyFile;
pub use service::FinanceService;
pub use state::StateStore;

pub type SurfaceResult<T> = Result<T, SurfaceError>;

#[derive(Debug)]
pub struct SurfaceError {
    kind: &'static str,
    message: String,
}
impl SurfaceError {
    pub fn invalid(v: impl Into<String>) -> Self {
        Self {
            kind: "invalid_arguments",
            message: v.into(),
        }
    }
    pub fn policy(v: impl Into<String>) -> Self {
        Self {
            kind: "policy_denied",
            message: v.into(),
        }
    }
    pub fn conflict(v: impl Into<String>) -> Self {
        Self {
            kind: "invalid_state",
            message: v.into(),
        }
    }
    pub fn state(v: impl Into<String>) -> Self {
        Self {
            kind: "state_error",
            message: v.into(),
        }
    }
    pub fn internal(v: impl Into<String>) -> Self {
        Self {
            kind: "internal_error",
            message: v.into(),
        }
    }
    pub fn tool_result(&self) -> Value {
        json!({"content":[{"type":"text","text":format!("{}: {}",self.kind,self.message)}],"isError":true})
    }
}
impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}
impl std::error::Error for SurfaceError {}

pub fn load_from_environment() -> SurfaceResult<FinanceService> {
    let policy_path = required_absolute_env("SINGULARITY_FINANCE_POLICY_FILE")?;
    let lease_path = required_absolute_env("SINGULARITY_FINANCE_ENABLE_LEASE_FILE")?;
    let state_dir = required_absolute_env("SINGULARITY_FINANCE_STATE_DIR")?;
    let key_hex = std::env::var("SINGULARITY_FINANCE_VERIFY_KEY_HEX")
        .map_err(|_| SurfaceError::policy("SINGULARITY_FINANCE_VERIFY_KEY_HEX is required"))?;
    let key = policy::verifying_key_from_hex(&key_hex)?;
    let policy = PolicyFile::load(&policy_path, &key)?;
    let state = StateStore::open(state_dir)?;
    {
        let _lock = state.lock()?;
        state.bind_policy(
            &policy.policy_id,
            policy.version,
            &policy::document_hash(&policy)?,
        )?;
    }
    Ok(FinanceService::new(policy, state, lease_path, key))
}
fn required_absolute_env(name: &str) -> SurfaceResult<PathBuf> {
    let value = std::env::var_os(name)
        .ok_or_else(|| SurfaceError::policy(format!("{name} is required")))?;
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(SurfaceError::policy(format!("{name} must be absolute")));
    }
    Ok(path)
}

pub fn tools() -> Value {
    let object = |properties: Value, required: Value| json!({"type":"object","properties":properties,"required":required,"additionalProperties":false});
    json!([
      {"name":"finance_propose","description":"Propose a policy-bound transfer by beneficiary ID. This never approves, signs, or executes.","inputSchema":object(json!({"request_id":{"type":"string"},"beneficiary_id":{"type":"string"},"asset":{"type":"string"},"amount_minor":{"type":"integer","minimum":1},"purpose":{"type":"string"},"ttl_seconds":{"type":"integer","minimum":1}}),json!(["request_id","beneficiary_id","asset","amount_minor","purpose","ttl_seconds"]))},
      {"name":"finance_status","description":"Read policy-bound proposal status without financial effect.","inputSchema":object(json!({"transaction_id":{"type":"string"}}),json!(["transaction_id"]))},
      {"name":"finance_cancel","description":"Cancel an unsigned proposal before its deadline. This cannot reverse a financial effect.","inputSchema":object(json!({"transaction_id":{"type":"string"},"request_id":{"type":"string"}}),json!(["transaction_id","request_id"]))}
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finance_contract_exposes_only_three_non_executing_tools() {
        let definitions = tools();
        let names: Vec<_> = definitions
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();

        assert_eq!(
            names,
            ["finance_propose", "finance_status", "finance_cancel"]
        );
    }
}
