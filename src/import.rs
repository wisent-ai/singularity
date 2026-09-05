use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::{ActivityEvent, ActivityStore, AgentState, AgentStatus, MemoryEntry, MemorySource};
use crate::error::AppError;

pub const IMPORT_SCHEMA_VERSION: &str = "singularity-mind-import-v1";
const MAX_IMPORT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ITEMS: usize = 1_000;
const MAX_SOURCE_BYTES: usize = 256;
const MAX_TEXT_BYTES: usize = 65_536;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MindImport {
    pub schema_version: String,
    pub source: ImportSource,
    #[serde(default)]
    pub memories: Vec<ImportItem>,
    #[serde(default)]
    pub knowledge: Vec<ImportItem>,
    #[serde(default)]
    pub profile: Vec<ImportItem>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ImportSource {
    pub kind: String,
    pub id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportItem {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportIssue {
    pub category: String,
    pub item_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportReport {
    pub accepted: bool,
    pub source_kind: String,
    pub source_id: String,
    pub imported: usize,
    pub attributed: usize,
    pub unchanged: usize,
    pub conflicting: usize,
    pub rejected: usize,
    pub issues: Vec<ImportIssue>,
}

#[derive(Debug, Clone)]
struct Candidate {
    category: &'static str,
    source: MemorySource,
    text: String,
}

pub(crate) fn read_import_document(path: &Path) -> Result<Vec<u8>, AppError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        AppError::State(format!("cannot inspect import file {}: {error}", path.display()))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(AppError::State(format!(
            "import file must be a regular file and not a symlink: {}",
            path.display()
        )));
    }
    if metadata.len() > MAX_IMPORT_BYTES {
        return Err(AppError::State(format!(
            "import file exceeds {MAX_IMPORT_BYTES} bytes"
        )));
    }
    Ok(fs::read(path)?)
}

pub fn parse_import_bytes(bytes: &[u8]) -> Result<MindImport, AppError> {
    if bytes.len() as u64 > MAX_IMPORT_BYTES {
        return Err(AppError::State(format!(
            "import file exceeds {MAX_IMPORT_BYTES} bytes"
        )));
    }
    serde_json::from_slice(bytes)
        .map_err(|error| AppError::State(format!("invalid Singularity import: {error}")))
}
pub fn validate_import(input: &MindImport) -> Result<(), AppError> {
    validate(input).map(|_| ())
}


pub async fn import_file(state_dir: &Path, path: &Path) -> Result<ImportReport, AppError> {
    let document = read_import_document(path)?;
    if let Some(report) = crate::state_service::import_through_service(state_dir, &document).await? {
        return Ok(report);
    }
    let input = parse_import_bytes(&document)?;
    let store = ActivityStore::open(state_dir)?;
    let state = store.load()?.ok_or_else(|| {
        AppError::State(format!(
            "no being state exists at {}; create the being before importing",
            store.state_path().display()
        ))
    })?;
    if matches!(
        state.status,
        AgentStatus::Starting | AgentStatus::Running | AgentStatus::Stopping
    ) {
        return Err(AppError::State(
            "the being is running; import through its local state service or stop it first".into(),
        ));
    }
    let (state, report) = apply_import(state, &input)?;
    if report.accepted {
        store.save(&state)?;
        store.append(&ActivityEvent::MindImported {
            at: Utc::now(),
            source_kind: report.source_kind.clone(),
            source_id: report.source_id.clone(),
            imported: report.imported,
            attributed: report.attributed,
            unchanged: report.unchanged,
        })?;
    }
    Ok(report)
}

pub fn apply_import(
    mut state: AgentState,
    input: &MindImport,
) -> Result<(AgentState, ImportReport), AppError> {
    let candidates = validate(input)?;
    let original = state.clone();
    let mut report = ImportReport {
        accepted: true,
        source_kind: input.source.kind.trim().to_owned(),
        source_id: input.source.id.trim().to_owned(),
        imported: 0,
        attributed: 0,
        unchanged: 0,
        conflicting: 0,
        rejected: 0,
        issues: Vec::new(),
    };

    let mut source_locations = BTreeMap::<MemorySource, usize>::new();
    for (index, memory) in state.mind.memories.iter().enumerate() {
        for source in &memory.sources {
            if let Some(previous) = source_locations.insert(source.clone(), index) {
                if previous != index {
                    return Err(AppError::State(
                        "state contains duplicate memory source attribution".into(),
                    ));
                }
            }
        }
    }

    for candidate in &candidates {
        if let Some(index) = source_locations.get(&candidate.source).copied() {
            let existing = &state.mind.memories[index];
            if existing.kind == candidate.category && existing.text == candidate.text {
                report.unchanged += 1;
            } else {
                report.conflicting += 1;
                report.issues.push(ImportIssue {
                    category: candidate.category.into(),
                    item_id: candidate.source.item_id.clone(),
                    reason: "source item was imported before with different content".into(),
                });
            }
            continue;
        }

        if let Some(existing) = state
            .mind
            .memories
            .iter_mut()
            .find(|memory| memory.kind == candidate.category && memory.text == candidate.text)
        {
            existing.sources.push(candidate.source.clone());
            report.attributed += 1;
            continue;
        }

        state.mind.memories.push(MemoryEntry {
            id: Uuid::new_v4(),
            kind: candidate.category.into(),
            text: candidate.text.clone(),
            created_at: Utc::now(),
            sources: vec![candidate.source.clone()],
        });
        report.imported += 1;
    }

    if state.mind.memories.len() > MAX_ITEMS {
        report.rejected += state.mind.memories.len() - MAX_ITEMS;
        report.issues.push(ImportIssue {
            category: "memory".into(),
            item_id: "*".into(),
            reason: format!("import would exceed the {MAX_ITEMS}-memory state limit"),
        });
    }

    if report.conflicting > 0 || report.rejected > 0 {
        report.accepted = false;
        report.imported = 0;
        report.attributed = 0;
        return Ok((original, report));
    }
    state.updated_at = Utc::now();
    Ok((state, report))
}


fn validate(input: &MindImport) -> Result<Vec<Candidate>, AppError> {
    if input.schema_version != IMPORT_SCHEMA_VERSION {
        return Err(AppError::State(format!(
            "unsupported import schema {}; expected {IMPORT_SCHEMA_VERSION}",
            input.schema_version
        )));
    }
    validate_atom(&input.source.kind, "source.kind")?;
    validate_atom(&input.source.id, "source.id")?;
    let total = input.memories.len() + input.knowledge.len() + input.profile.len();
    if total == 0 {
        return Err(AppError::State(
            "import must contain at least one memory, knowledge, or profile item".into(),
        ));
    }
    if total > MAX_ITEMS {
        return Err(AppError::State(format!(
            "import contains {total} items; maximum is {MAX_ITEMS}"
        )));
    }

    let mut seen = BTreeSet::new();
    let mut candidates = Vec::with_capacity(total);
    for (category, items) in [
        ("memory", input.memories.as_slice()),
        ("knowledge", input.knowledge.as_slice()),
        ("profile", input.profile.as_slice()),
    ] {
        for item in items {
            validate_atom(&item.id, &format!("{category}.id"))?;
            let text = item.text.trim();
            if text.is_empty() || text.len() > MAX_TEXT_BYTES || text.contains('\0') {
                return Err(AppError::State(format!(
                    "{category} item {} text must be 1..={MAX_TEXT_BYTES} bytes and contain no NUL",
                    item.id
                )));
            }
            let key = item.id.trim();
            if !seen.insert(key) {
                return Err(AppError::State(format!(
                    "duplicate source item id {} in import",
                    item.id
                )));
            }
            candidates.push(Candidate {
                category,
                source: MemorySource {
                    kind: input.source.kind.trim().into(),
                    source_id: input.source.id.trim().into(),
                    item_id: item.id.trim().into(),
                },
                text: text.into(),
            });
        }
    }
    Ok(candidates)
}

fn validate_atom(value: &str, label: &str) -> Result<(), AppError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() > MAX_SOURCE_BYTES
        || trimmed.chars().any(char::is_control)
    {
        return Err(AppError::State(format!(
            "{label} must be 1..={MAX_SOURCE_BYTES} bytes and contain no control characters"
        )));
    }
    Ok(())
}
