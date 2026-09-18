"""Reading a capability environment file and checking the entries it declares."""

import grp
import os
import stat
from pathlib import Path

from preflight.contract import (
    AGENT_PATHS,
    AGENT_REQUIRED,
    BROKER_PATHS,
    BROKER_REQUIRED,
    BROKER_SOCKET_MODE,
    CLIENT_GROUP,
    MCP_AGENT_ID,
    NAME,
    REJECTED_VALUE,
    SAFE_REFERENCE_SUFFIXES,
    SKARBIEC_MCP_ENV_NAMES,
    CONTRACT_TARGETS,
    HEX64,
    UNSAFE_NAME,
    fail,
    reject_symlinks,
    require_absolute,
    require_secure,
    verify_binary,
)
def load_env(path: Path) -> dict[str, str]:
    require_absolute(path, "environment file")
    require_secure(path, "file", owner=os.geteuid())
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or "=" not in line:
            fail(f"{path}:{number}: only literal NAME=VALUE entries are allowed")
        name, value = line.split("=", 1)
        if not NAME.fullmatch(name) or name in values:
            fail(f"{path}:{number}: invalid or duplicate environment name")
        if not value or value != value.strip() or value[0] in "'\"" or any(c in value for c in "$`\\\n\r\x00"):
            fail(f"{path}:{number}: values must be nonempty unquoted literals without expansion")
        if REJECTED_VALUE.search(value):
            fail(f"{path}:{number}: unresolved deployment marker")
        if UNSAFE_NAME.search(name) and not name.endswith(SAFE_REFERENCE_SUFFIXES):
            fail(f"{path}:{number}: raw secret-bearing environment variable is forbidden: {name}")
        values[name] = value
    return values
def require_entries(values: dict[str, str], required: set[str] | frozenset[str]) -> None:
    missing = sorted(required - set(values))
    if missing:
        fail("missing required entries: " + ", ".join(missing))


def validate_paths(values: dict[str, str], contracts: dict[str, str]) -> None:
    require_entries(values, frozenset(contracts))
    for name, kind in contracts.items():
        path = Path(values[name])
        require_secure(path, kind, owner=os.geteuid() if kind != "executable" else 0)

def validate_broker(values: dict[str, str]) -> None:
    require_entries(values, BROKER_REQUIRED)
    validate_paths(values, BROKER_PATHS)
    try:
        configured_gid = int(values["SKARBIEC_CAP_SOCKET_GID"], 10)
        deployed_gid = grp.getgrnam(CLIENT_GROUP).gr_gid
    except (ValueError, KeyError):
        fail(f"{CLIENT_GROUP} must exist and SKARBIEC_CAP_SOCKET_GID must be its numeric GID")
    if configured_gid != deployed_gid or configured_gid != os.getegid():
        fail("broker socket GID must equal the broker effective GID and deployed client-group GID")
    if not MCP_AGENT_ID.fullmatch(values["SKARBIEC_MCP_AGENT_ID"]):
        fail("SKARBIEC_MCP_AGENT_ID must be an explicit non-wildcard identity")
    socket = Path(values["SKARBIEC_CAP_SOCKET"])
    require_absolute(socket, "SKARBIEC_CAP_SOCKET")
    reject_symlinks(socket)
    require_secure(socket.parent, "shared-dir", owner=os.geteuid(), group=configured_gid)
    if socket.exists() and not stat.S_ISSOCK(os.lstat(socket).st_mode):
        fail(f"existing socket target is not a Unix socket: {socket}")
    verify_binary(values["SKARBIEC_BINARY"], values["SKARBIEC_BINARY_SHA256"], "broker")


def validate_agent(values: dict[str, str]) -> None:
    require_entries(values, AGENT_REQUIRED)
    validate_paths(values, AGENT_PATHS)
    workload_id = values["SKARBIEC_WORKLOAD_ID"]
    if workload_id not in CONTRACT_TARGETS:
        fail("SKARBIEC_WORKLOAD_ID must be an exact capability-contract target")
    verify_binary(values["SINGULARITY_BOOTSTRAP_BINARY"], values["SINGULARITY_BOOTSTRAP_BINARY_SHA256"], "bootstrap")
    manifest_path = Path(values["SINGULARITY_BOOTSTRAP_MANIFEST"])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"invalid bootstrap manifest: {error}")
    if not isinstance(manifest, dict):
        fail("bootstrap manifest must be a JSON object")
    for field in ("workload_private_key_file", "singularity_executable", "executable_digest", "policy_digest", "broker_socket"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            fail(f"bootstrap manifest is missing {field}")
    signing_key = Path(values["SKARBIEC_WORKLOAD_SIGNING_KEY_FILE"])
    if Path(manifest["workload_private_key_file"]) != signing_key:
        fail("manifest signing-key path must equal SKARBIEC_WORKLOAD_SIGNING_KEY_FILE")
    require_secure(signing_key, "file", owner=os.geteuid())
    verify_binary(manifest["singularity_executable"], manifest["executable_digest"], "runtime")
    if not HEX64.fullmatch(manifest["policy_digest"]):
        fail("manifest policy_digest must be lowercase SHA-256")
    socket = Path(values["SKARBIEC_CAP_SOCKET"])
    if Path(manifest["broker_socket"]) != socket:
        fail("manifest broker_socket must equal SKARBIEC_CAP_SOCKET")
    require_absolute(socket, "SKARBIEC_CAP_SOCKET")
    reject_symlinks(socket)
    info = os.lstat(socket)
    if not stat.S_ISSOCK(info.st_mode) or stat.S_IMODE(info.st_mode) != BROKER_SOCKET_MODE:
        fail("broker socket must be a 0660 Unix socket")
    if not os.access(socket, os.R_OK | os.W_OK):
        fail("broker socket is not accessible to this workload UID")

