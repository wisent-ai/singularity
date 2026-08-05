use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use wisent_onboarding_client::{
    bundle_from_canonical, ExperimentAssignment, ExperimentAssignmentRequest, FileStorage,
    JourneyBundle, JourneyClient, JourneyError, OfflineTransport, ProgressStatus, RuntimeEvent,
    ScopeKind, StadoTransport, Storage, Transport,
};

const PRODUCT_ID: &str = "singularity";
const JOURNEY_ID: &str = "first-use";
const JOURNEY_VERSION: &str = "2026-08-04.1";
const JOURNEY_VERSION_ID: &str = "f073d568-54ea-4b0a-8bb6-c412db36b3cf";
const SOURCE_REVISION: &str = "singularity:first-use:2026-08-04.1";
const FIRST_SUCCESS_FACT: &str = "agent_loop_result_observed";
const FALLBACK: &str = include_str!("onboarding_first_use.json");

static ONBOARDING_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

type Client = JourneyClient<Box<dyn Transport>, FileStorage>;

struct SingularityTransport(StadoTransport);

#[async_trait::async_trait]
impl Transport for SingularityTransport {
    async fn read_bundle(
        &self,
        product_id: &str,
        journey_id: &str,
    ) -> Result<JourneyBundle, JourneyError> {
        let bundle = self.0.read_bundle(product_id, journey_id).await?;
        let expected = bundle_from_canonical(
            FALLBACK,
            Uuid::parse_str(JOURNEY_VERSION_ID)
                .map_err(|_| JourneyError::Invalid("journey_version_id".into()))?,
        )?;
        let definition = &bundle.definition;
        if bundle.journey_version_id != expected.journey_version_id
            || bundle.content_sha256 != expected.content_sha256
            || bundle.source_revision != SOURCE_REVISION
            || definition.schema_version != 1
            || definition.product_id != PRODUCT_ID
            || definition.journey_id != JOURNEY_ID
            || definition.journey_version != JOURNEY_VERSION
            || definition.source_revision != SOURCE_REVISION
            || definition.first_success_fact != FIRST_SUCCESS_FACT
        {
            return Err(JourneyError::Invalid("singularity first-use identity".into()));
        }
        Ok(bundle)
    }

    async fn collect_event(&self, event: &RuntimeEvent) -> Result<(), JourneyError> {
        self.0.collect_event(event).await
    }

    async fn read_state(
        &self,
        product_id: &str,
        attempt_id: Uuid,
        subject_hash: &str,
    ) -> Result<Option<Value>, JourneyError> {
        self.0
            .read_state(product_id, attempt_id, subject_hash)
            .await
    }

    async fn assign_experiment(
        &self,
        request: &ExperimentAssignmentRequest,
    ) -> Result<ExperimentAssignment, JourneyError> {
        self.0.assign_experiment(request).await
    }
}

fn state_path() -> PathBuf {
    env::var_os("SINGULARITY_ONBOARDING_STATE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".singularity/onboarding-state.json"))
}

fn subject_hash(subject: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(subject.as_bytes());
    digest.update(b"\0singularity\0first-use\0user");
    hex::encode(digest.finalize())
}

fn transport() -> Box<dyn Transport> {
    let endpoint = env::var("STADO_INTEGRATION_API_URL")
        .or_else(|_| env::var("STADO_API_URL"))
        .unwrap_or_default();
    let token = env::var("SINGULARITY_STADO_INTEGRATION_TOKEN").unwrap_or_default();
    if !endpoint.trim().is_empty() && !token.trim().is_empty() {
        if let Ok(transport) = StadoTransport::new(endpoint.trim(), token) {
            return Box::new(SingularityTransport(transport));
        }
    }
    Box::new(OfflineTransport)
}

fn fallback_bundle() -> Result<JourneyBundle, String> {
    bundle_from_canonical(
        FALLBACK,
        Uuid::parse_str(JOURNEY_VERSION_ID).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

async fn start_client(subject: &str, revision: &str) -> Result<Client, String> {
    let storage = FileStorage::new(state_path());
    let fallback = fallback_bundle()?;
    storage
        .save_bundle(&fallback)
        .await
        .map_err(|error| error.to_string())?;
    let mut client = JourneyClient::new(
        PRODUCT_ID,
        JOURNEY_ID,
        subject_hash(subject),
        ScopeKind::User,
        transport(),
        storage,
        fallback,
    )
    .map_err(|error| error.to_string())?;
    client
        .start(revision)
        .await
        .map_err(|error| error.to_string())?;
    Ok(client)
}

fn current_screen(client: &Client) -> Result<wisent_onboarding_client::Screen, String> {
    let progress = client
        .progress()
        .ok_or_else(|| "onboarding progress is unavailable".to_string())?;
    client
        .bundle()
        .and_then(|bundle| {
            bundle
                .definition
                .screens
                .iter()
                .find(|screen| screen.screen_id == progress.current_screen_id)
        })
        .cloned()
        .ok_or_else(|| "onboarding screen is unavailable".to_string())
}

fn status_name(status: ProgressStatus) -> &'static str {
    match status {
        ProgressStatus::InProgress => "in_progress",
        ProgressStatus::Completed => "completed",
        ProgressStatus::Skipped => "skipped",
        ProgressStatus::Abandoned => "abandoned",
        ProgressStatus::Reset => "reset",
    }
}

pub async fn show(subject: &str) -> Result<Value, String> {
    let _guard = ONBOARDING_LOCK.lock().await;
    let mut client = start_client(subject, SOURCE_REVISION).await?;
    if client.progress().is_some_and(|progress| progress.status == ProgressStatus::InProgress) {
        loop {
            let screen = current_screen(&client)?;
            client
                .expose(SOURCE_REVISION)
                .await
                .map_err(|error| error.to_string())?;
            if screen.transitions.is_empty() {
                break;
            }
            if client
                .advance(&BTreeMap::new(), SOURCE_REVISION)
                .await
                .map_err(|error| error.to_string())?
                .is_none()
            {
                break;
            }
        }
    }
    let progress = client
        .progress()
        .cloned()
        .ok_or_else(|| "onboarding progress is unavailable".to_string())?;
    let screens = client
        .bundle()
        .ok_or_else(|| "onboarding bundle is unavailable".to_string())?
        .definition
        .screens
        .iter()
        .map(|screen| {
            json!({
                "screen_id": screen.screen_id,
                "kind": screen.screen_kind,
                "title_key": screen.title_key,
                "body_key": screen.body_key,
                "actions": screen.actions,
            })
        })
        .collect::<Vec<_>>();
    let _ = client.flush().await;
    Ok(json!({
        "product_id": PRODUCT_ID,
        "journey_id": JOURNEY_ID,
        "journey_version": JOURNEY_VERSION,
        "journey_version_id": JOURNEY_VERSION_ID,
        "source_revision": SOURCE_REVISION,
        "attempt_id": progress.attempt_id,
        "subject": subject,
        "status": status_name(progress.status),
        "current_screen_id": progress.current_screen_id,
        "evidence_revision": progress.evidence_revision,
        "experiment_id": progress.experiment_id,
        "variant_id": progress.variant_id,
        "screens": screens,
        "first_success_fact": FIRST_SUCCESS_FACT,
        "next": if progress.status == ProgressStatus::Completed {
            "Complete: a real agent cycle returned observable content or a tool result."
        } else {
            "Run singularity once or run; accepted startup alone does not complete first use."
        }
    }))
}

pub async fn change_status(subject: &str, action: &str) -> Result<Value, String> {
    let _guard = ONBOARDING_LOCK.lock().await;
    let revision = format!("{SOURCE_REVISION}:{action}");
    let mut client = start_client(subject, &revision).await?;
    match action {
        "reset" => client.reset(&revision).await,
        "resume" => client.resume(&revision).await,
        "skip" => client.skip(&revision).await,
        "abandon" => client.abandon(&revision).await,
        _ => return Err(format!("unknown onboarding action: {action}")),
    }
    .map_err(|error| error.to_string())?;
    let progress = client
        .progress()
        .ok_or_else(|| "onboarding progress is unavailable".to_string())?;
    let payload = json!({
        "product_id": PRODUCT_ID,
        "journey_id": JOURNEY_ID,
        "attempt_id": progress.attempt_id,
        "status": status_name(progress.status),
        "current_screen_id": progress.current_screen_id,
        "evidence_revision": progress.evidence_revision,
    });
    let _ = client.flush().await;
    Ok(payload)
}

pub async fn record_agent_result(
    subject: &str,
    cycle: u64,
    final_content: Option<&str>,
    actions: &[String],
) -> Result<bool, String> {
    let has_content = final_content.is_some_and(|value| !value.trim().is_empty());
    if subject.trim().is_empty() || cycle == 0 || (!has_content && actions.is_empty()) {
        return Ok(false);
    }
    let _guard = ONBOARDING_LOCK.lock().await;
    let storage = FileStorage::new(state_path());
    let subject_hash = subject_hash(subject);
    let existing = storage
        .load_progress(PRODUCT_ID, JOURNEY_ID, &subject_hash)
        .await
        .map_err(|error| error.to_string())?;
    if !existing.is_some_and(|progress| progress.status == ProgressStatus::InProgress) {
        return Ok(false);
    }
    let mut digest = Sha256::new();
    digest.update(subject.as_bytes());
    digest.update(cycle.to_be_bytes());
    digest.update(final_content.unwrap_or_default().as_bytes());
    for action in actions {
        digest.update(b"\0");
        digest.update(action.as_bytes());
    }
    let revision = format!("agent-loop-result:{}", hex::encode(digest.finalize()));
    let mut client = start_client(subject, &revision).await?;
    let evidence = BTreeMap::from([(FIRST_SUCCESS_FACT.to_string(), Value::Bool(true))]);
    while !current_screen(&client)?.transitions.is_empty() {
        if client
            .advance(&evidence, &revision)
            .await
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(false);
        }
    }
    let completed = client
        .complete(&evidence, &revision)
        .await
        .map_err(|error| error.to_string())?;
    let _ = client.flush().await;
    Ok(completed)
}
