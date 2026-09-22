mod evidence;
mod owner;
use evidence::{Evidence, Result, SOURCE_REVISION};
use owner::Owner;
use rusqlite::{Connection, OpenFlags};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[test]
fn live_control_retains_readable_history_across_owner_interruption() {
    let mut evidence = Evidence::new().expect("create real journey evidence");
    let result = journey(&mut evidence);
    evidence
        .finish(&result)
        .expect("retain real journey result");
    if let Err(error) = result {
        panic!("{error}");
    }
}

fn journey(evidence: &mut Evidence) -> Result<()> {
    let revision = SOURCE_REVISION.ok_or("candidate lacks WISENT_SOURCE_COMMIT provenance")?;
    require(
        revision.len() == 40 && revision.bytes().all(|v| v.is_ascii_hexdigit()),
        "candidate source revision is not a full Git commit",
    )?;
    let policy_path = PathBuf::from(std::env::var_os("SINGULARITY_ECOSYSTEM_TEST_POLICY").ok_or(
        "SINGULARITY_ECOSYSTEM_TEST_POLICY must identify a real read-only delegated policy",
    )?);
    let policy_path =
        fs::canonicalize(policy_path).map_err(|e| format!("resolve real test policy: {e}"))?;
    let policy: Value = serde_json::from_slice(&fs::read(&policy_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    for capability in [
        "allow_write",
        "allow_command",
        "allow_release",
        "allow_product_creation",
    ] {
        require(
            policy[capability] == false,
            &format!("test policy must explicitly refuse {capability}"),
        )?;
    }
    require(
        policy["sources"] == json!(["product_catalog"]),
        "test policy must observe only the real product catalog",
    )?;
    let workspace = PathBuf::from(text(&policy["workspace_root"], "policy workspace root")?);
    let root = fs::canonicalize(&evidence.root).map_err(|e| e.to_string())?;
    require(
        !root.starts_with(&workspace),
        "isolated test state must be outside the policy's executor checkout root",
    )?;
    evidence.snapshot("delegated_policy", policy)?;
    let mut owner = Owner::new(evidence, policy_path);
    owner.start(true)?;
    let status = owner.cli(&["status"], true)?;
    require(
        status["result"]["paused"] == true,
        "owner admitted work despite start-paused",
    )?;
    require(
        status["result"]["runtime"]["source_revision"] == revision,
        "live owner does not identify the candidate source revision",
    )?;
    owner.cli(&["pause"], true)?;
    owner.cli(&["pause"], true)?;
    let first = owner.cli(&["records", "control", "--limit", "1"], true)?;
    let cursor = first["result"]["next_cursor"]
        .as_i64()
        .ok_or("control page omitted its continuation")?
        .to_string();
    let second = owner.cli(
        &["records", "control", "--limit", "1", "--before", &cursor],
        true,
    )?;
    require(
        first["result"]["items"].as_array().map(Vec::len) == Some(1)
            && second["result"]["items"].as_array().map(Vec::len) == Some(1),
        "control pages did not obey requested page size",
    )?;
    require(
        first["result"]["items"][0]["id"] != second["result"]["items"][0]["id"],
        "continuation repeated the same control record",
    )?;
    let launches = owner.cli(&["records", "owner_launch", "--limit", "1"], true)?;
    let id = text(
        &launches["result"]["items"][0]["id"],
        "owner launch record identity",
    )?
    .to_owned();
    let db = Connection::open_with_flags(
        owner.state.join("ecosystem.sqlite3"),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(|e| format!("inspect real owner database: {e}"))?;
    let persisted: String = db
        .query_row(
            "SELECT data FROM records WHERE kind='owner_launch' AND id=?1",
            [&id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;
    drop(db);
    let digest = format!("{:x}", Sha256::digest(persisted.as_bytes()));
    owner.evidence.snapshot(
        "persisted_launch",
        json!({"id":id,"bytes":persisted,"sha256":digest}),
    )?;
    let reconstructed = fragments(&mut owner, &id, &digest)?;
    require(
        reconstructed == persisted,
        "fragment reassembly differs from persisted SQLite bytes",
    )?;
    refusal_cases(&mut owner, &id, &digest, &persisted)?;
    owner.stop()?;
    owner.start(false)?;
    let recovered = owner.cli(&["status"], true)?;
    require(
        recovered["result"]["paused"] == true,
        "interruption lost the recorded admissions pause",
    )?;
    require(
        fragments(&mut owner, &id, &digest)? == persisted,
        "interruption changed the retained owner record",
    )?;
    owner.stop()?;
    owner.cli(&["status"], false)?;
    owner.evidence.snapshot("final_state", json!({"owner_stopped":true,"record_id":id,"content_sha256":digest,"pause_survived_interruption":true}))?;
    Ok(())
}

fn fragments(owner: &mut Owner<'_>, id: &str, digest: &str) -> Result<String> {
    let mut joined = String::new();
    let mut offset = 0_u64;
    loop {
        let value = owner.cli(
            &[
                "record",
                "owner_launch",
                id,
                "--bytes",
                "127",
                "--offset",
                &offset.to_string(),
                "--revision",
                digest,
            ],
            true,
        )?;
        let chunk = &value["result"];
        require(
            chunk["offset"] == offset && chunk["content_sha256"] == digest,
            "fragment changed offset or revision",
        )?;
        let text = text(&chunk["text"], "record fragment")?;
        require(text.len() <= 127, "fragment exceeded requested byte limit")?;
        joined.push_str(text);
        match chunk["next_offset"].as_u64() {
            Some(next) => {
                require(
                    next > offset && next == offset + text.len() as u64,
                    "fragment continuation made no progress or skipped bytes",
                )?;
                offset = next;
            }
            None => {
                require(
                    chunk["next_offset"].is_null() && chunk["total_bytes"] == joined.len() as u64,
                    "end of record did not match its declared byte count",
                )?;
                break;
            }
        }
    }
    Ok(joined)
}

fn refusal_cases(owner: &mut Owner<'_>, id: &str, digest: &str, persisted: &str) -> Result<()> {
    owner.cli(&["records", "--limit", "0"], false)?;
    owner.cli(&["record", "owner_launch", id, "--bytes", "3"], false)?;
    owner.cli(&["record", "owner_launch", id, "--offset", "1"], false)?;
    let mut wrong = digest.to_owned();
    wrong.replace_range(..1, if digest.starts_with('0') { "1" } else { "0" });
    owner.cli(&["record", "owner_launch", id, "--revision", &wrong], false)?;
    let beyond = (persisted.len() + 1).to_string();
    owner.cli(
        &[
            "record",
            "owner_launch",
            id,
            "--offset",
            &beyond,
            "--revision",
            digest,
        ],
        false,
    )?;
    let continuation = persisted
        .bytes()
        .position(|value| value & 0xc0 == 0x80)
        .ok_or("real owner record did not retain its Unicode display name")?
        .to_string();
    owner.cli(
        &[
            "record",
            "owner_launch",
            id,
            "--offset",
            &continuation,
            "--revision",
            digest,
        ],
        false,
    )?;
    let missing = uuid::Uuid::new_v4().to_string();
    owner.cli(&["record", "owner_launch", &missing], false)?;
    Ok(())
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| format!("missing {name}: {value}"))
}
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
