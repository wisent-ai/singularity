//! Reading a capability environment file and checking what it declares for
//! the broker or an agent.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::Value;

use super::contract::{
    is_digest, is_socket, reject_symlinks, require_absolute, require_secure, verify_binary, Check,
    Kind, BROKER_SOCKET_MODE, CLIENT_GROUP, CONTRACT_TARGETS,
};

pub(super) type Values = BTreeMap<String, String>;

const BROKER_PATHS: [(&str, Kind); 10] = [
    ("SKARBIEC_VAULT_FILE", Kind::File),
    ("SKARBIEC_CAP_POLICY", Kind::File),
    ("SKARBIEC_CAP_POLICY_SIG", Kind::File),
    ("SKARBIEC_CAP_TRUST_ROOT", Kind::File),
    ("SKARBIEC_WORKLOAD_REGISTRY", Kind::File),
    ("SKARBIEC_WORKLOAD_REGISTRY_SIG", Kind::File),
    ("SKARBIEC_CAP_STATE", Kind::Dir),
    ("SKARBIEC_WORM_RECEIPT_DIR", Kind::Dir),
    ("SKARBIEC_WORM_RECEIPT_COMMAND", Kind::Executable),
    ("SKARBIEC_WORM_CHECKPOINT", Kind::File),
];
const BROKER_REQUIRED: [&str; 5] = [
    "SKARBIEC_BINARY",
    "SKARBIEC_BINARY_SHA256",
    "SKARBIEC_CAP_SOCKET",
    "SKARBIEC_CAP_SOCKET_GID",
    "SKARBIEC_MCP_AGENT_ID",
];
const AGENT_PATHS: [(&str, Kind); 5] = [
    ("SINGULARITY_BOOTSTRAP_MANIFEST", Kind::File),
    ("SINGULARITY_BOOTSTRAP_MANIFEST_SIG", Kind::File),
    ("SINGULARITY_BOOTSTRAP_TRUST_ROOT", Kind::File),
    ("SINGULARITY_RUNTIME_ROOT", Kind::Dir),
    ("SKARBIEC_WORKLOAD_SIGNING_KEY_FILE", Kind::File),
];
const AGENT_REQUIRED: [&str; 4] = [
    "SINGULARITY_BOOTSTRAP_BINARY",
    "SINGULARITY_BOOTSTRAP_BINARY_SHA256",
    "SKARBIEC_CAP_SOCKET",
    "SKARBIEC_WORKLOAD_ID",
];
const MANIFEST_FIELDS: [&str; 5] = [
    "workload_private_key_file",
    "singularity_executable",
    "executable_digest",
    "policy_digest",
    "broker_socket",
];
/// Names that would carry a secret by value; a name ending in one of the
/// reference suffixes carries a path to one, which is what is allowed.
const SECRET_NAME: &str = r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY)";
const REFERENCE_SUFFIXES: [&str; 4] = ["_FILE", "_PATH", "_DIR", "_SOCKET"];
/// The markers an example file uses where a deployment must put a real value.
const UNRESOLVED_MARKER: &str =
    r"(?i)(replace|placeholder|example|changeme|todo|insert[-_ ]?here|<[^>]+>)";

fn euid() -> u32 {
    // SAFETY: geteuid and getegid read the calling process's credentials and cannot fail.
    unsafe { libc::geteuid() }
}

fn egid() -> u32 {
    // SAFETY: as above.
    unsafe { libc::getegid() }
}

fn group_id(name: &str) -> Option<u32> {
    let name = CString::new(name).ok()?;
    // SAFETY: getgrnam reads the group database; the pointer is checked before use
    // and its gr_gid copied out before any other call can reuse the buffer.
    let entry = unsafe { libc::getgrnam(name.as_ptr()) };
    (!entry.is_null()).then(|| unsafe { (*entry).gr_gid })
}

fn accessible(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: access only reads the NUL-terminated path.
    unsafe { libc::access(path.as_ptr(), libc::R_OK | libc::W_OK) == 0 }
}

pub(super) fn load_env(path: &Path) -> Check<Values> {
    require_absolute(path, "environment file")?;
    require_secure(path, Kind::File, Some(euid()), None)?;
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let name_rule = Regex::new(r"^[A-Z][A-Z0-9_]*$").map_err(|error| error.to_string())?;
    let secret_name = Regex::new(SECRET_NAME).map_err(|error| error.to_string())?;
    let marker = Regex::new(UNRESOLVED_MARKER).map_err(|error| error.to_string())?;
    let mut values = Values::new();
    for (index, raw) in text.lines().enumerate() {
        let at = format!("{}:{}", path.display(), index + 1);
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line
            .split_once('=')
            .filter(|_| !line.starts_with("export "))
        else {
            return Err(format!("{at}: only literal NAME=VALUE entries are allowed"));
        };
        if !name_rule.is_match(name) || values.contains_key(name) {
            return Err(format!("{at}: invalid or duplicate environment name"));
        }
        if value.is_empty()
            || value != value.trim()
            || value.starts_with(['\'', '"'])
            || value.contains(['$', '`', '\\', '\n', '\r', '\0'])
        {
            return Err(format!(
                "{at}: values must be nonempty unquoted literals without expansion"
            ));
        }
        if marker.is_match(value) {
            return Err(format!("{at}: unresolved deployment marker"));
        }
        if secret_name.is_match(name)
            && !REFERENCE_SUFFIXES
                .iter()
                .any(|suffix| name.ends_with(suffix))
        {
            return Err(format!(
                "{at}: raw secret-bearing environment variable is forbidden: {name}"
            ));
        }
        values.insert(name.to_string(), value.to_string());
    }
    Ok(values)
}

fn require_entries<'a>(values: &Values, required: impl IntoIterator<Item = &'a str>) -> Check<()> {
    let missing: Vec<&str> = required
        .into_iter()
        .filter(|name| !values.contains_key(*name))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!("missing required entries: {}", missing.join(", ")))
    }
}

fn validate_paths(values: &Values, contracts: &[(&str, Kind)]) -> Check<()> {
    require_entries(values, contracts.iter().map(|(name, _)| *name))?;
    for (name, kind) in contracts {
        let owner = if matches!(kind, Kind::Executable) {
            0
        } else {
            euid()
        };
        require_secure(Path::new(&values[*name]), *kind, Some(owner), None)?;
    }
    Ok(())
}

pub(super) fn validate_broker(values: &Values) -> Check<()> {
    require_entries(values, BROKER_REQUIRED)?;
    validate_paths(values, &BROKER_PATHS)?;
    let configured = values["SKARBIEC_CAP_SOCKET_GID"].parse::<u32>().ok();
    let deployed = group_id(CLIENT_GROUP);
    let (Some(configured), Some(deployed)) = (configured, deployed) else {
        return Err(format!(
            "{CLIENT_GROUP} must exist and SKARBIEC_CAP_SOCKET_GID must be its numeric GID"
        ));
    };
    if configured != deployed || configured != egid() {
        return Err(
            "broker socket GID must equal the broker effective GID and deployed client-group GID"
                .into(),
        );
    }
    let agent_id =
        Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$").map_err(|error| error.to_string())?;
    if !agent_id.is_match(&values["SKARBIEC_MCP_AGENT_ID"]) {
        return Err("SKARBIEC_MCP_AGENT_ID must be an explicit non-wildcard identity".into());
    }
    let socket = PathBuf::from(&values["SKARBIEC_CAP_SOCKET"]);
    require_absolute(&socket, "SKARBIEC_CAP_SOCKET")?;
    reject_symlinks(&socket)?;
    let parent = socket
        .parent()
        .ok_or("SKARBIEC_CAP_SOCKET has no parent directory")?;
    require_secure(parent, Kind::SharedDir, Some(euid()), Some(configured))?;
    if let Some((_, false)) = is_socket(&socket)? {
        return Err(format!(
            "existing socket target is not a Unix socket: {}",
            socket.display()
        ));
    }
    verify_binary(
        &values["SKARBIEC_BINARY"],
        &values["SKARBIEC_BINARY_SHA256"],
        "broker",
    )?;
    Ok(())
}

pub(super) fn validate_agent(values: &Values) -> Check<()> {
    require_entries(values, AGENT_REQUIRED)?;
    validate_paths(values, &AGENT_PATHS)?;
    if !CONTRACT_TARGETS.contains(&values["SKARBIEC_WORKLOAD_ID"].as_str()) {
        return Err("SKARBIEC_WORKLOAD_ID must be an exact capability-contract target".into());
    }
    verify_binary(
        &values["SINGULARITY_BOOTSTRAP_BINARY"],
        &values["SINGULARITY_BOOTSTRAP_BINARY_SHA256"],
        "bootstrap",
    )?;
    let manifest_path = Path::new(&values["SINGULARITY_BOOTSTRAP_MANIFEST"]);
    let manifest: Value = std::fs::read_to_string(manifest_path)
        .map_err(|error| error.to_string())
        .and_then(|text| serde_json::from_str(&text).map_err(|error| error.to_string()))
        .map_err(|error| format!("invalid bootstrap manifest: {error}"))?;
    let manifest = manifest
        .as_object()
        .ok_or("bootstrap manifest must be a JSON object")?;
    let field = |name: &str| {
        manifest
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
    };
    if let Some(missing) = MANIFEST_FIELDS.iter().find(|name| field(name).is_none()) {
        return Err(format!("bootstrap manifest is missing {missing}"));
    }
    let signing_key = PathBuf::from(&values["SKARBIEC_WORKLOAD_SIGNING_KEY_FILE"]);
    if Path::new(field("workload_private_key_file").unwrap_or_default()) != signing_key {
        return Err(
            "manifest signing-key path must equal SKARBIEC_WORKLOAD_SIGNING_KEY_FILE".into(),
        );
    }
    require_secure(&signing_key, Kind::File, Some(euid()), None)?;
    verify_binary(
        field("singularity_executable").unwrap_or_default(),
        field("executable_digest").unwrap_or_default(),
        "runtime",
    )?;
    if !is_digest(field("policy_digest").unwrap_or_default()) {
        return Err("manifest policy_digest must be lowercase SHA-256".into());
    }
    let socket = PathBuf::from(&values["SKARBIEC_CAP_SOCKET"]);
    if Path::new(field("broker_socket").unwrap_or_default()) != socket {
        return Err("manifest broker_socket must equal SKARBIEC_CAP_SOCKET".into());
    }
    require_absolute(&socket, "SKARBIEC_CAP_SOCKET")?;
    reject_symlinks(&socket)?;
    if is_socket(&socket)? != Some((BROKER_SOCKET_MODE, true)) {
        return Err("broker socket must be a 0660 Unix socket".into());
    }
    if !accessible(&socket) {
        return Err("broker socket is not accessible to this workload UID".into());
    }
    Ok(())
}
