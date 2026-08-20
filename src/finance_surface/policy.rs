use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use super::{SurfaceError, SurfaceResult};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedDocument {
    pub document: Value,
    pub signature_hex: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFile {
    pub policy_id: String,
    pub version: u64,
    pub valid_from: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub beneficiaries: BTreeMap<String, Beneficiary>,
    pub assets: BTreeMap<String, AssetPolicy>,
    pub approval: ApprovalPolicy,
    pub custody_authorities: CustodyAuthorities,
    pub worm_sink_dir: PathBuf,
    pub worm_sink_id: String,
    pub worm_receipt_key_hex: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Beneficiary {
    pub destination: String,
    pub allowed_assets: Vec<String>,
    pub allowed_purposes: Vec<String>,
    pub valid_from: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub per_transaction_minor: i64,
    pub rolling_window_seconds: u64,
    pub rolling_limit_minor: i64,
    pub daily_limit_minor: i64,
    pub lifetime_limit_minor: i64,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetPolicy {
    pub per_transaction_minor: i64,
    pub rolling_window_seconds: u64,
    pub rolling_limit_minor: i64,
    pub daily_limit_minor: i64,
    pub lifetime_limit_minor: i64,
    pub spendable_balance_minor: i64,
    pub protected_reserve_minor: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalPolicy {
    pub required_approvals: u32,
    pub approver_keys: BTreeMap<String, String>,
    pub timelock_seconds: u64,
    pub proposal_ttl_max_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustodyAuthorities {
    pub simulators: BTreeMap<String, String>,
    pub signers: BTreeMap<String, String>,
    pub executors: BTreeMap<String, String>,
    pub reconcilers: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnableLease {
    pub policy_id: String,
    pub policy_version: u64,
    pub lease_id: String,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub enabled: bool,
    pub kill_switch: bool,
}
pub fn document_hash<T: Serialize>(document: &T) -> SurfaceResult<String> {
    let value = serde_json::to_value(document)
        .map_err(|e| SurfaceError::internal(format!("cannot serialize authority document: {e}")))?;
    Ok(hex::encode(Sha256::digest(canonical_json(&value)?)))
}

impl PolicyFile {
    pub fn load(path: &Path, verifying_key: &VerifyingKey) -> SurfaceResult<Self> {
        let policy: Self = load_signed(path, verifying_key, "policy")?;
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> SurfaceResult<()> {
        validate_id("policy_id", &self.policy_id)?;
        if self.version == 0 || self.valid_from >= self.expires_at {
            return Err(SurfaceError::policy("invalid policy validity or version"));
        }
        if self.beneficiaries.is_empty() || self.assets.is_empty() {
            return Err(SurfaceError::policy(
                "policy must define beneficiaries and assets",
            ));
        }
        for (id, beneficiary) in &self.beneficiaries {
            validate_id("beneficiary id", id)?;
            if beneficiary.destination.is_empty() || beneficiary.destination.len() > 1024 {
                return Err(SurfaceError::policy("invalid beneficiary destination"));
            }
            if beneficiary.allowed_assets.is_empty()
                || beneficiary
                    .allowed_assets
                    .iter()
                    .any(|asset| !self.assets.contains_key(asset))
            {
                return Err(SurfaceError::policy(
                    "beneficiary has invalid allowed_assets",
                ));
            }
            if beneficiary.allowed_purposes.is_empty() {
                return Err(SurfaceError::policy(
                    "beneficiary must define allowed_purposes",
                ));
            }
            for purpose in &beneficiary.allowed_purposes {
                validate_id("beneficiary purpose", purpose)?;
            }
            if beneficiary.valid_from < self.valid_from
                || beneficiary.expires_at > self.expires_at
                || beneficiary.valid_from >= beneficiary.expires_at
                || beneficiary.per_transaction_minor <= 0
                || beneficiary.rolling_window_seconds == 0
                || beneficiary.rolling_limit_minor <= 0
                || beneficiary.daily_limit_minor <= 0
                || beneficiary.lifetime_limit_minor <= 0
                || beneficiary.per_transaction_minor > beneficiary.rolling_limit_minor
                || beneficiary.rolling_limit_minor > beneficiary.lifetime_limit_minor
                || beneficiary.daily_limit_minor > beneficiary.lifetime_limit_minor
            {
                return Err(SurfaceError::policy("invalid beneficiary policy"));
            }
        }
        for (asset, limits) in &self.assets {
            validate_asset(asset)?;
            if limits.per_transaction_minor <= 0
                || limits.rolling_window_seconds == 0
                || limits.rolling_limit_minor <= 0
                || limits.daily_limit_minor <= 0
                || limits.lifetime_limit_minor <= 0
                || limits.spendable_balance_minor < 0
                || limits.protected_reserve_minor < 0
                || limits.protected_reserve_minor > limits.spendable_balance_minor
                || limits.per_transaction_minor > limits.rolling_limit_minor
                || limits.rolling_limit_minor > limits.lifetime_limit_minor
                || limits.daily_limit_minor > limits.lifetime_limit_minor
            {
                return Err(SurfaceError::policy("invalid asset limits"));
            }
        }
        if self.approval.required_approvals == 0
            || self.approval.required_approvals as usize > self.approval.approver_keys.len()
            || self.approval.timelock_seconds == 0
            || self.approval.proposal_ttl_max_seconds == 0
            || self.approval.proposal_ttl_max_seconds > 30 * 24 * 60 * 60
        {
            return Err(SurfaceError::policy("invalid approval policy"));
        }
        let mut distinct_approver_keys = BTreeSet::new();
        for (approver_id, key) in &self.approval.approver_keys {
            validate_id("approver_id", approver_id)?;
            let parsed = verifying_key_from_hex(key)?;
            if !distinct_approver_keys.insert(parsed.to_bytes()) {
                return Err(SurfaceError::policy(
                    "approver public keys must be distinct",
                ));
            }
        }
        validate_id("worm_sink_id", &self.worm_sink_id)?;
        let worm_key = verifying_key_from_hex(&self.worm_receipt_key_hex)?;
        if !distinct_approver_keys.insert(worm_key.to_bytes()) {
            return Err(SurfaceError::policy("WORM receipt key must be independent"));
        }
        for (role, authorities) in [
            ("simulator", &self.custody_authorities.simulators),
            ("signer", &self.custody_authorities.signers),
            ("executor", &self.custody_authorities.executors),
            ("reconciler", &self.custody_authorities.reconcilers),
        ] {
            if authorities.is_empty() {
                return Err(SurfaceError::policy(format!(
                    "policy has no {role} authorities"
                )));
            }
            for (id, key) in authorities {
                validate_id(&format!("{role}_id"), id)?;
                let parsed = verifying_key_from_hex(key)?;
                if !distinct_approver_keys.insert(parsed.to_bytes()) {
                    return Err(SurfaceError::policy(
                        "all custody authority public keys must be distinct",
                    ));
                }
            }
        }
        if !self.worm_sink_dir.is_absolute() {
            return Err(SurfaceError::policy("worm_sink_dir must be absolute"));
        }
        Ok(())
    }

    pub fn require_active(&self, now: DateTime<Utc>) -> SurfaceResult<()> {
        if now < self.valid_from || now >= self.expires_at {
            return Err(SurfaceError::policy("no active signed finance policy"));
        }
        Ok(())
    }
}

impl EnableLease {
    pub fn load_active(
        path: &Path,
        verifying_key: &VerifyingKey,
        policy: &PolicyFile,
        now: DateTime<Utc>,
    ) -> SurfaceResult<Self> {
        let lease: Self = load_signed(path, verifying_key, "enable lease")?;
        validate_id("lease_id", &lease.lease_id)?;
        if lease.policy_id != policy.policy_id
            || lease.policy_version != policy.version
            || !lease.enabled
            || lease.kill_switch
            || lease.issued_at > now
            || lease.expires_at <= now
            || lease.expires_at > policy.expires_at
        {
            return Err(SurfaceError::policy(
                "finance enable lease is absent, expired, disabled, or mismatched",
            ));
        }
        Ok(lease)
    }
}

pub fn verifying_key_from_hex(value: &str) -> SurfaceResult<VerifyingKey> {
    let bytes = hex::decode(value)
        .map_err(|_| SurfaceError::policy("invalid policy verification key hex"))?;
    let key: [u8; 32] = bytes
        .try_into()
        .map_err(|_| SurfaceError::policy("policy verification key must be 32 bytes"))?;
    VerifyingKey::from_bytes(&key)
        .map_err(|_| SurfaceError::policy("invalid policy verification key"))
}

pub fn load_signed<T: for<'de> Deserialize<'de>>(
    path: &Path,
    verifying_key: &VerifyingKey,
    kind: &str,
) -> SurfaceResult<T> {
    require_owner_only_file(path)?;
    let bytes =
        fs::read(path).map_err(|e| SurfaceError::policy(format!("cannot read {kind}: {e}")))?;
    let envelope: SignedDocument = serde_json::from_slice(&bytes)
        .map_err(|e| SurfaceError::policy(format!("invalid signed {kind}: {e}")))?;
    let signature_bytes = hex::decode(&envelope.signature_hex)
        .map_err(|_| SurfaceError::policy(format!("invalid {kind} signature encoding")))?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|_| SurfaceError::policy(format!("invalid {kind} signature")))?;
    let canonical = canonical_json(&envelope.document)?;
    verifying_key
        .verify(&canonical, &signature)
        .map_err(|_| SurfaceError::policy(format!("{kind} signature verification failed")))?;
    serde_json::from_value(envelope.document)
        .map_err(|e| SurfaceError::policy(format!("invalid {kind} document: {e}")))
}

/// Deterministic signing contract: UTF-8 JSON with lexicographically sorted object
/// keys, no insignificant whitespace, serde JSON string escaping, and canonical
/// serde number rendering. Duplicate keys are rejected by typed documents after
/// signature verification; signed producers must use this exact representation.
pub fn canonical_json(value: &Value) -> SurfaceResult<Vec<u8>> {
    fn write_value(value: &Value, out: &mut Vec<u8>) -> SurfaceResult<()> {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::Number(number) => out.extend_from_slice(number.to_string().as_bytes()),
            Value::String(string) => {
                out.extend_from_slice(&serde_json::to_vec(string).map_err(|e| {
                    SurfaceError::internal(format!("cannot canonicalize string: {e}"))
                })?)
            }
            Value::Array(values) => {
                out.push(b'[');
                for (index, item) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    write_value(item, out)?;
                }
                out.push(b']');
            }
            Value::Object(map) => {
                out.push(b'{');
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
                for (index, (key, item)) in entries.into_iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(&serde_json::to_vec(key).map_err(|e| {
                        SurfaceError::internal(format!("cannot canonicalize key: {e}"))
                    })?);
                    out.push(b':');
                    write_value(item, out)?;
                }
                out.push(b'}');
            }
        }
        Ok(())
    }
    let mut output = Vec::new();
    write_value(value, &mut output)?;
    Ok(output)
}

pub fn validate_id(kind: &str, value: &str) -> SurfaceResult<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        return Err(SurfaceError::invalid(format!("invalid {kind}")));
    }
    Ok(())
}

pub fn validate_asset(value: &str) -> SurfaceResult<()> {
    if value.is_empty()
        || value.len() > 16
        || !value
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
    {
        return Err(SurfaceError::invalid(
            "asset must be 1..=16 uppercase ASCII letters/digits",
        ));
    }
    Ok(())
}

#[cfg(unix)]
pub fn require_owner_only_file(path: &Path) -> SurfaceResult<()> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| SurfaceError::policy(format!("cannot stat protected file: {e}")))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.uid() != unsafe { geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err(SurfaceError::policy(
            "protected file must be owner-only, current-user-owned, regular, and not a symlink",
        ));
    }
    Ok(())
}

#[cfg(unix)]
unsafe extern "C" {
    fn geteuid() -> u32;
}

#[cfg(not(unix))]
pub fn require_owner_only_file(_path: &Path) -> SurfaceResult<()> {
    Err(SurfaceError::policy(
        "finance policy enforcement requires Unix",
    ))
}
