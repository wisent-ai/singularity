//! The walkthrough's authorities and signed files: Ed25519 keys by role, the
//! envelope and role-message signatures the finance service verifies, the
//! progress kept between the two phases, and owner-only writes.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

pub const POLICY_ID: &str = "walkthrough-policy";
pub const VERSION: u64 = 1;
const OWNER_ONLY: u32 = 0o600;

pub struct Key(SigningKey);

impl Key {
    pub fn seed(&self) -> String {
        hex::encode(self.0.to_bytes())
    }
    pub fn public(&self) -> String {
        hex::encode(self.0.verifying_key().to_bytes())
    }
    pub fn sign(&self, message: &[u8]) -> String {
        hex::encode(self.0.sign(message).to_bytes())
    }
    pub fn envelope(&self, document: &Value) -> Outcome<String> {
        let canonical = serde_json::to_string(document)?;
        Ok(
            json!({"document": document, "signature_hex": self.sign(canonical.as_bytes())})
                .to_string(),
        )
    }
    pub fn role(&self, role: &str, tx: &str, intent: &str, reference: &str) -> String {
        self.sign(
            format!(
                "singularity-finance-{role}-v1:{POLICY_ID}:{VERSION}:{tx}:{intent}:{reference}"
            )
            .as_bytes(),
        )
    }
}

pub fn key_from(seed: &str) -> Outcome<Key> {
    let bytes: [u8; 32] = hex::decode(seed)?
        .try_into()
        .map_err(|_| "a key seed is not 32 bytes")?;
    Ok(Key(SigningKey::from_bytes(&bytes)))
}

pub fn fresh_key() -> Key {
    let mut seed = [0u8; 32];
    seed[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    seed[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    Key(SigningKey::from_bytes(&seed))
}

/// What phase one leaves for phase two: the keys by role, the first lease's
/// issue time, the transaction, and the owner-event counter.
#[derive(Serialize, Deserialize)]
pub struct Progress {
    pub seeds: Vec<String>,
    pub first_lease: DateTime<Utc>,
    pub tx: String,
    pub hash: String,
    pub events: u64,
}

pub fn ts(at: DateTime<Utc>) -> String {
    at.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()
}

pub fn sha(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

pub fn write_owner_only(path: &Path, content: &str) -> Outcome<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(OWNER_ONLY)
        .open(path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

pub fn with(intent: &Value, changes: Value) -> Value {
    let mut merged = intent.clone();
    if let (Some(target), Some(source)) = (merged.as_object_mut(), changes.as_object()) {
        target.extend(source.clone());
    }
    merged
}
