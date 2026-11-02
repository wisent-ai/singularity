use super::{BramaClient, brama};
use crate::config::ModelService;
use crate::{AppError, ErrorClass};
use hmac::{Hmac, Mac};
use secrecy::ExposeSecret;
use sha2::{Digest, Sha256};

impl BramaClient {
    /// Brama's request carries its bearer and the agent's signature. Through
    /// Stado nothing is posted from here, so nothing is authenticated.
    pub(super) fn authenticate(
        &self,
        request: reqwest::RequestBuilder,
        body: &[u8],
    ) -> Result<reqwest::RequestBuilder, AppError> {
        let (secret, bearer) = match &self.service {
            ModelService::Brama { secret, bearer, .. } => (secret, bearer),
            ModelService::Stado => {
                return Err(brama(
                    ErrorClass::Permanent,
                    format!(
                        "no Brama address is named, so no request is signed here; requests go to `{}`",
                        self.service.describe()
                    ),
                ));
            }
        };
        let timestamp = chrono::Utc::now().timestamp().to_string();
        // Brama signs an empty body (a GET) with an empty body hash, not the
        // hash of no bytes (crypto::hmac_auth::verify_agent_hmac); hashing it
        // here left every signed catalogue read unverifiable.
        let body_hash = if body.is_empty() {
            String::new()
        } else {
            hex::encode(Sha256::digest(body))
        };
        let signed = format!("{}:{}:{}", self.agent_id, timestamp, body_hash);
        let mut mac =
            Hmac::<Sha256>::new_from_slice(secret.expose_secret().as_bytes()).map_err(|error| {
                brama(
                    ErrorClass::Permanent,
                    format!("cannot initialize signer: {error}"),
                )
            })?;
        mac.update(signed.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        Ok(request
            .bearer_auth(bearer.expose_secret())
            .header("x-agent-id", &self.agent_id)
            .header("x-agent-timestamp", timestamp)
            .header("x-agent-body-sha256", body_hash)
            .header("x-agent-signature", signature))
    }
}
