//! Reading and verifying a signed document, and the canonical JSON its signature covers.
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::Path;

use super::*;
use super::{SurfaceError, SurfaceResult};
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
