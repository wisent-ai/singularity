use super::super::{
    Shared,
    model::{Initiative, Opportunity},
    observe,
    protocol::SCHEMA_VERSION,
};
use super::source;
use crate::AppError;
use serde_json::json;
use sha2::{Digest, Sha256};

pub async fn ensure(
    shared: &Shared,
    initiative: &mut Initiative,
    opportunity: &Opportunity,
) -> Result<(), AppError> {
    let (policy, directory, paused) = {
        let state = shared.lock()?;
        (
            state.policy.clone(),
            state.directory.clone(),
            state.store.paused()?,
        )
    };
    let id = opportunity.product_id.as_deref().ok_or_else(|| {
        AppError::Config("new-product opportunity must name its product identifier".into())
    })?;
    if !source::identifier(id) {
        return Err(AppError::Config(
            "new-product identifier must be a lowercase product name".into(),
        ));
    }
    if !policy.allow_product_creation {
        return Err(AppError::Config("product creation is not delegated".into()));
    }
    if let Some(record) = shared
        .lock()?
        .store
        .get::<serde_json::Value>("creation", &initiative.id)?
    {
        if record["state"] == "provisioned" && record["product"]["id"].as_str() == Some(id) {
            initiative.product_id = Some(id.into());
            return Ok(());
        }
    }
    if paused {
        return Err(AppError::State("new product provisioning is paused".into()));
    }
    if !shared.lock()?.store.meta::<bool>("las_catalog_ready")?.unwrap_or(false) {
        return Err(AppError::State("signed Las catalogue is unavailable; product provisioning was not dispatched".into()));
    }
    let request_id = format!(
        "creation-{}",
        hex::encode(Sha256::digest(initiative.id.as_bytes()))
    );
    let request = json!({"schema_version":SCHEMA_VERSION,"request_id":request_id,"initiative_id":initiative.id,
        "product":{"id":id,"name":opportunity.title,"description":opportunity.description,"family":"wisent","visibility":"private"},
        "repositories":[{"surface":"cli","repository":format!("wisent-ai/{id}")},
            {"surface":"desktop","repository":format!("wisent-ai/{id}-desktop")},
            {"surface":"web","repository":format!("wisent-ai/{id}-landing")}],
        "evidence_refs":opportunity.evidence_refs});
    let path = source::immutable_request(&directory, &request_id, &request)?;
    let path = path
        .to_str()
        .ok_or_else(|| AppError::Config("creation request path is not UTF-8".into()))?;
    let response =
        observe::products(shared, &["create", "--request", path, "--allow-create", "--json"]).await?;
    shared.lock()?.store.put(
        "creation",
        &initiative.id,
        &response,
        Some(&initiative.id),
        "Read back Wisent Products creation state; provisioning is not implementation",
    )?;
    if response["request_id"] != request_id
        || response["product"]["id"].as_str() != Some(id)
        || response["state"] != "provisioned"
    {
        return Err(AppError::State(format!(
            "Wisent Products did not provision the exact requested product: {response}"
        )));
    }
    initiative.product_id = Some(id.into());
    Ok(())
}
