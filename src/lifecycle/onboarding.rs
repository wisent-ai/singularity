use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;
use wisent_onboarding_client::{
    ExperimentAssignment, ExperimentAssignmentRequest, FileStorage, IntegrationTransport,
    JourneyBundle, JourneyClient, JourneyError, OfflineTransport, ProgressStatus, RuntimeEvent,
    ScopeKind, Storage, Transport, bundle_from_canonical,
};

use crate::CycleReport;

const PRODUCT_ID: &str = "singularity";
const JOURNEY_ID: &str = "first-use";
const JOURNEY_VERSION: &str = "2026-09-04.1";
const JOURNEY_VERSION_ID: &str = "ec8347d4-243f-4d15-82ab-d3fcf2c25e70";
const SOURCE_REVISION: &str = "singularity-first-use-2026-09-04.1";
const FIRST_SUCCESS_FACT: &str = "autonomous_cycle_completed";
const STATE_REVISION: &str = "cli:first-use:2026-09-04.1";
/// The journey definition this product ships; the client is started from it.
const JOURNEY_DEFINITION: &str = include_str!("onboarding_first_use.json");

type Client = JourneyClient<Box<dyn Transport>, FileStorage>;

struct SingularityTransport(IntegrationTransport);

#[async_trait]
impl Transport for SingularityTransport {
    async fn read_bundle(
        &self,
        product_id: &str,
        journey_id: &str,
    ) -> Result<JourneyBundle, JourneyError> {
        let bundle = self.0.read_bundle(product_id, journey_id).await?;
        let definition = &bundle.definition;
        if definition.schema_version != 1
            || definition.product_id != PRODUCT_ID
            || definition.journey_id != JOURNEY_ID
            || definition.journey_version != JOURNEY_VERSION
            || definition.source_revision != SOURCE_REVISION
            || definition.first_success_fact != FIRST_SUCCESS_FACT
        {
            return Err(JourneyError::Invalid(
                "Singularity first-use journey identity".into(),
            ));
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

fn transport() -> Result<Box<dyn Transport>, JourneyError> {
    let endpoint = configured("STADO_INTEGRATION_API_URL")?;
    let token = configured("SINGULARITY_STADO_INTEGRATION_TOKEN")?;
    match (endpoint, token) {
        (Some(endpoint), Some(token)) => {
            let integration = IntegrationTransport::new(&endpoint, token).map_err(|error| {
                JourneyError::Invalid(format!(
                    "STADO_INTEGRATION_API_URL {endpoint} is configured but unusable: {error}"
                ))
            })?;
            Ok(Box::new(SingularityTransport(integration)))
        }
        (None, None) => Ok(Box::new(OfflineTransport)),
        (Some(_), None) => Err(JourneyError::Invalid(
            "STADO_INTEGRATION_API_URL is set without SINGULARITY_STADO_INTEGRATION_TOKEN".into(),
        )),
        (None, Some(_)) => Err(JourneyError::Invalid(
            "SINGULARITY_STADO_INTEGRATION_TOKEN is set without STADO_INTEGRATION_API_URL".into(),
        )),
    }
}

/// A variable that is set but empty is a deployment that meant to configure something and did
/// not finish; it is refused rather than read as absent.
fn configured(name: &str) -> Result<Option<String>, JourneyError> {
    match env::var(name) {
        Ok(value) if value.trim().is_empty() => Err(JourneyError::Invalid(format!(
            "{name} is set to an empty value"
        ))),
        Ok(value) => Ok(Some(value.trim().to_owned())),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(JourneyError::Invalid(format!(
            "{name} is not readable: {error}"
        ))),
    }
}

fn state_path() -> Result<PathBuf, JourneyError> {
    if let Some(path) = env::var_os("SINGULARITY_ONBOARDING_STATE_PATH") {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("XDG_STATE_HOME") {
        return Ok(PathBuf::from(path).join("singularity/onboarding.json"));
    }
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".local/state/singularity/onboarding.json"));
    }
    Err(JourneyError::Invalid(
        "onboarding state has nowhere to live: set SINGULARITY_ONBOARDING_STATE_PATH, XDG_STATE_HOME or HOME".into(),
    ))
}

fn stable_subject_hash() -> Result<String, JourneyError> {
    let operator = env::var("USER").map_err(|error| {
        JourneyError::Invalid(format!("USER is required to identify the device: {error}"))
    })?;
    let digest = Sha256::digest(format!("{PRODUCT_ID}:device:{operator}").as_bytes());
    Ok(hex::encode(digest))
}

fn journey_bundle() -> Result<JourneyBundle, JourneyError> {
    bundle_from_canonical(
        JOURNEY_DEFINITION,
        Uuid::parse_str(JOURNEY_VERSION_ID)
            .map_err(|_| JourneyError::Invalid("Singularity journey version id".into()))?,
    )
}

async fn start_client(revision: &str) -> Result<Client, JourneyError> {
    let mut client = JourneyClient::new(
        PRODUCT_ID,
        JOURNEY_ID,
        stable_subject_hash()?,
        ScopeKind::Device,
        transport()?,
        FileStorage::new(state_path()?),
        journey_bundle()?,
    )?;
    client.start(revision).await?;
    Ok(client)
}

/// Show or resume Singularity's first-use journey. Reset goes through the
/// journey client so progress and its completion evidence are discarded
/// together before the first screen is shown again in this invocation.
pub async fn run_first_use(reset: bool) -> Result<bool, JourneyError> {
    let mut journey = start_client(STATE_REVISION).await?;
    if reset {
        journey.reset(STATE_REVISION).await?;
        println!(
            "Singularity first-use journey reset: recorded progress and evidence discarded; showing it again now."
        );
        println!();
    }

    if journey
        .progress()
        .is_some_and(|progress| progress.status == ProgressStatus::Completed)
    {
        println!(
            "Singularity first-use journey is already complete: one autonomous cycle completed."
        );
        println!("Re-run with --reset to show the walkthrough again.");
        journey.flush().await?;
        return Ok(true);
    }

    let no_evidence = BTreeMap::new();
    loop {
        journey.expose(STATE_REVISION).await?;
        render_current_step(&journey)?;
        let screen = current_screen(&journey)?;
        if screen.transitions.is_empty() {
            break;
        }
        if journey
            .advance(&no_evidence, STATE_REVISION)
            .await?
            .is_none()
        {
            return Err(JourneyError::Invalid(
                "Singularity journey cannot advance with current evidence".into(),
            ));
        }
        println!();
    }

    println!();
    println!(
        "Next: run `singularity once` with the normal workload configuration. A completed cycle records first-use success."
    );
    println!("Onboarding remains in progress until that real cycle completes.");
    journey.flush().await?;
    Ok(false)
}

/// Record first success only from the normal `once` execution path after it has
/// returned a completed cycle. Merely displaying the journey never calls this.
pub async fn record_completed_cycle(report: &CycleReport) -> Result<bool, JourneyError> {
    if report.cycle == 0 || report.status != "completed" {
        return Ok(false);
    }

    let storage = FileStorage::new(state_path()?);
    let subject_hash = stable_subject_hash()?;
    let existing = storage
        .load_progress(PRODUCT_ID, JOURNEY_ID, &subject_hash)
        .await?;
    if !existing.is_some_and(|progress| progress.status == ProgressStatus::InProgress) {
        return Ok(false);
    }

    let revision = format!("once:cycle:{}:tokens:{}", report.cycle, report.total_tokens);
    let mut journey = start_client(&revision).await?;
    let evidence = BTreeMap::from([(FIRST_SUCCESS_FACT.to_string(), Value::Bool(true))]);
    while !current_screen(&journey)?.transitions.is_empty() {
        if journey.advance(&evidence, &revision).await?.is_none() {
            return Ok(false);
        }
    }
    journey.observe_first_success(&evidence, &revision).await?;
    let completed = journey.complete(&evidence, &revision).await?;
    journey.flush().await?;
    Ok(completed)
}

fn current_screen(client: &Client) -> Result<wisent_onboarding_client::Screen, JourneyError> {
    let progress = client.progress().ok_or(JourneyError::NotStarted)?;
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
        .ok_or_else(|| JourneyError::Invalid("Singularity current screen".into()))
}

fn render_current_step(client: &Client) -> Result<(), JourneyError> {
    let screen = current_screen(client)?;
    let title = presentation_text(&screen, "title", &screen.title_key)?;
    let body = presentation_text(&screen, "body", &screen.body_key)?;
    println!("{title}");
    println!("{body}");
    Ok(())
}

fn presentation_text<'screen>(
    screen: &'screen wisent_onboarding_client::Screen,
    field: &str,
    key: &str,
) -> Result<&'screen str, JourneyError> {
    screen
        .presentation
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            JourneyError::Invalid(format!(
                "Singularity screen {} has no {field} text for {key}",
                screen.screen_id
            ))
        })
}
