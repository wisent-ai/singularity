use super::constants::{CATALOG_RESPONSE_BYTES, TOKENS_PER_MILLION};
use super::{BramaClient, BramaCompletion, ModelsResponse, brama, map_network};
use crate::{AppError, ErrorClass};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Catalog {
    version: String,
    catalog_revision: String,
    degraded: bool,
    models: Vec<Entry>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Entry {
    id: String,
    available: bool,
    context_window: u64,
    max_output_tokens: u64,
    price: Price,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct Price {
    input: Decimal,
    output: Decimal,
    cache_read: Decimal,
    cache_write: Decimal,
}
#[derive(Deserialize, Serialize)]
pub struct Quote {
    model: String,
    catalog_revision: String,
    price: Price,
    pub upper_usd: Decimal,
}
impl Quote {
    pub fn cost(&self, completion: &BramaCompletion) -> Result<Decimal, AppError> {
        if completion.model != self.model {
            return Err(brama(
                ErrorClass::Permanent,
                "served model differs from the reserved concrete route; its cost remains unresolved",
            ));
        }
        let uncached = completion
            .usage
            .prompt_tokens
            .checked_sub(completion.cache_read_tokens)
            .and_then(|remaining| remaining.checked_sub(completion.cache_write_tokens))
            .ok_or_else(|| {
                brama(
                    ErrorClass::Permanent,
                    "cached usage exceeds the reported prompt usage",
                )
            })?;
        Ok((Decimal::from(uncached) * self.price.input
            + Decimal::from(completion.usage.completion_tokens) * self.price.output
            + Decimal::from(completion.cache_read_tokens) * self.price.cache_read
            + Decimal::from(completion.cache_write_tokens) * self.price.cache_write)
            / Decimal::from(TOKENS_PER_MILLION))
    }
}
impl BramaClient {
    pub async fn health(&self) -> Result<(), AppError> {
        let value: Value = self
            .http
            .get(self.endpoint("health")?)
            .send()
            .await
            .map_err(map_network)?
            .error_for_status()
            .map_err(map_network)?
            .json()
            .await
            .map_err(map_network)?;
        if value.get("status").and_then(Value::as_str) != Some("ok") {
            return Err(brama(ErrorClass::Permanent, "health response is not ok"));
        }
        Ok(())
    }

    pub async fn models(&self) -> Result<Vec<String>, AppError> {
        let response: ModelsResponse = self
            .authenticate(self.http.get(self.endpoint("v1/models")?), &[])?
            .send()
            .await
            .map_err(map_network)?
            .error_for_status()
            .map_err(map_network)?
            .json()
            .await
            .map_err(map_network)?;
        Ok(response.data.into_iter().map(|entry| entry.id).collect())
    }
    /// Fetch fresh caller-scoped capacity and prices. Unknown capacity is not an affordable call.
    pub async fn quote(&self) -> Result<Quote, AppError> {
        let mut response = self
            .authenticate(self.http.get(self.endpoint("v1/models")?), &[])?
            .header("x-jeden-schema-min", "1")
            .header("x-jeden-schema-max", "1")
            .send()
            .await
            .map_err(map_network)?;
        let status = response.status();
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(map_network)? {
            if chunk.len() > CATALOG_RESPONSE_BYTES.saturating_sub(bytes.len()) {
                return Err(brama(
                    ErrorClass::Permanent,
                    "model catalog exceeds its response bound",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            return Err(brama(
                if status.is_server_error() {
                    ErrorClass::Transient
                } else {
                    ErrorClass::Permanent
                },
                format!(
                    "GET /v1/models: HTTP {status}: {}",
                    String::from_utf8_lossy(&bytes)
                ),
            ));
        }
        let catalog: Catalog = serde_json::from_slice(&bytes)?;
        if catalog.version != "v1" || catalog.catalog_revision.is_empty() || catalog.degraded {
            return Err(brama(
                ErrorClass::Transient,
                "model catalog is degraded, unversioned or unsupported; no model cost was authorized",
            ));
        }
        let entry = catalog
            .models
            .into_iter()
            .find(|entry| entry.id == self.model)
            .ok_or_else(|| {
                brama(
                    ErrorClass::Permanent,
                    format!(
                        "configured model {} is absent from the caller-scoped catalog",
                        self.model
                    ),
                )
            })?;
        if !entry.available
            || !entry.id.contains('/')
            || entry.context_window == 0
            || entry.max_output_tokens == 0
            || u64::from(self.max_tokens) > entry.max_output_tokens
            || entry.price.input <= Decimal::ZERO
            || entry.price.output <= Decimal::ZERO
            || entry.price.cache_read < Decimal::ZERO
            || entry.price.cache_write < Decimal::ZERO
        {
            return Err(brama(
                ErrorClass::Permanent,
                "cost admission requires an available concrete route with positive prices and sufficient token ceilings",
            ));
        }
        let input = entry.price.input + entry.price.cache_read + entry.price.cache_write;
        let upper_usd = (Decimal::from(entry.context_window) * input
            + Decimal::from(self.max_tokens) * entry.price.output)
            / Decimal::from(TOKENS_PER_MILLION);
        Ok(Quote {
            model: entry.id,
            catalog_revision: catalog.catalog_revision,
            price: entry.price,
            upper_usd,
        })
    }
}
