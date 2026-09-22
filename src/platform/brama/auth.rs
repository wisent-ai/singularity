use super::{brama, BramaClient};
use crate::{AppError, ErrorClass};
use hmac::{Hmac, Mac};
use secrecy::ExposeSecret;
use sha2::{Digest, Sha256};

impl BramaClient {
    pub(super) fn authenticate(&self, request: reqwest::RequestBuilder, body: &[u8]) -> Result<reqwest::RequestBuilder, AppError> {
        let timestamp = chrono::Utc::now().timestamp().to_string();
        let body_hash = hex::encode(Sha256::digest(body));
        let signed = format!("{}:{}:{}", self.agent_id, timestamp, body_hash);
        let mut mac = Hmac::<Sha256>::new_from_slice(self.secret.expose_secret().as_bytes())
            .map_err(|error| brama(ErrorClass::Permanent, format!("cannot initialize signer: {error}")))?;
        mac.update(signed.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        Ok(request.bearer_auth(self.bearer.expose_secret())
            .header("x-agent-id", &self.agent_id)
            .header("x-agent-timestamp", timestamp)
            .header("x-agent-body-sha256", body_hash)
            .header("x-agent-signature", signature))
    }
}
