#!/usr/bin/env python3
"""Fail-closed deployment preflight for capability-isolated processes."""

import argparse
import hashlib
import grp
import json
import os
import re
import stat
import sys
from pathlib import Path

NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNSAFE_NAME = re.compile(r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY)")
SAFE_REFERENCE_SUFFIXES = ("_FILE", "_PATH", "_DIR", "_SOCKET")
REJECTED_VALUE = re.compile(r"(?i)(replace|placeholder|example|changeme|todo|insert[-_ ]?here|<[^>]+>)")
CONTRACT_TARGETS = frozenset(("weles", "most-service", "brama", "singularity-bootstrap"))
CLIENT_GROUP = "skarbiec-capability-clients"
MCP_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SKARBIEC_MCP_ENV_NAMES = (
    "SKARBIEC_VAULT_FILE",
    "SKARBIEC_CAP_POLICY",
    "SKARBIEC_CAP_POLICY_SIG",
    "SKARBIEC_CAP_TRUST_ROOT",
    "SKARBIEC_WORKLOAD_REGISTRY",
    "SKARBIEC_WORKLOAD_REGISTRY_SIG",
    "SKARBIEC_CAP_STATE",
    "SKARBIEC_CAP_SOCKET",
    "SKARBIEC_WORM_RECEIPT_DIR",
    "SKARBIEC_WORM_CHECKPOINT",
    "SKARBIEC_WORM_RECEIPT_COMMAND",
    "SKARBIEC_MCP_AGENT_ID",
)

BROKER_PATHS = {
    "SKARBIEC_VAULT_FILE": "file",
    "SKARBIEC_CAP_POLICY": "file",
    "SKARBIEC_CAP_POLICY_SIG": "file",
    "SKARBIEC_CAP_TRUST_ROOT": "file",
    "SKARBIEC_WORKLOAD_REGISTRY": "file",
    "SKARBIEC_WORKLOAD_REGISTRY_SIG": "file",
    "SKARBIEC_CAP_STATE": "dir",
    "SKARBIEC_WORM_RECEIPT_DIR": "dir",
    "SKARBIEC_WORM_RECEIPT_COMMAND": "executable",
    "SKARBIEC_WORM_CHECKPOINT": "file",
}
BROKER_REQUIRED = frozenset((
    "SKARBIEC_BINARY",
    "SKARBIEC_BINARY_SHA256",
    "SKARBIEC_CAP_SOCKET",
    "SKARBIEC_CAP_SOCKET_GID",
    "SKARBIEC_MCP_AGENT_ID",
))
AGENT_PATHS = {
    "SINGULARITY_BOOTSTRAP_MANIFEST": "file",
    "SINGULARITY_BOOTSTRAP_MANIFEST_SIG": "file",
    "SINGULARITY_BOOTSTRAP_TRUST_ROOT": "file",
    "SINGULARITY_RUNTIME_ROOT": "dir",
    "SKARBIEC_WORKLOAD_SIGNING_KEY_FILE": "file",
}
AGENT_REQUIRED = frozenset((
    "SINGULARITY_BOOTSTRAP_BINARY",
    "SINGULARITY_BOOTSTRAP_BINARY_SHA256",
    "SKARBIEC_CAP_SOCKET",
    "SKARBIEC_WORKLOAD_ID",
))


def fail(message: str) -> "None":
    raise ValueError(message)


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


def require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        fail(f"{label} must be an absolute normalized path")


def reject_symlinks(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                fail(f"symlink path component is forbidden: {current}")
        except FileNotFoundError:
            break


def require_secure(
    path: Path,
    kind: str,
    owner: int | None = None,
    group: int | None = None,
) -> None:
    require_absolute(path, str(path))
    reject_symlinks(path)
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        fail(f"symlink is forbidden: {path}")
    if owner is not None and info.st_uid != owner:
        fail(f"wrong owner for {path}: expected uid {owner}")
    if group is not None and info.st_gid != group:
        fail(f"wrong group for {path}: expected gid {group}")
    mode = stat.S_IMODE(info.st_mode)
    if kind == "file":
        if not stat.S_ISREG(info.st_mode) or mode & 0o077:
            fail(f"owner-only regular file required: {path}")
    elif kind == "dir":
        if not stat.S_ISDIR(info.st_mode) or mode != 0o700:
            fail(f"0700 directory required: {path}")
    elif kind == "shared-dir":
        if not stat.S_ISDIR(info.st_mode) or mode != 0o750:
            fail(f"0750 shared directory required: {path}")
    elif kind == "executable":
        if not stat.S_ISREG(info.st_mode) or mode & 0o022 or not mode & 0o100:
            fail(f"non-writable owner-executable regular file required: {path}")
    else:
        fail(f"internal error: unsupported path kind {kind}")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_binary(path_text: str, expected: str, label: str) -> Path:
    path = Path(path_text)
    require_secure(path, "executable", owner=0)
    if not HEX64.fullmatch(expected):
        fail(f"{label} digest must be lowercase SHA-256")
    if digest(path) != expected:
        fail(f"{label} release binary digest mismatch")
    return path


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
    if not stat.S_ISSOCK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o660:
        fail("broker socket must be a 0660 Unix socket")
    if not os.access(socket, os.R_OK | os.W_OK):
        fail("broker socket is not accessible to this workload UID")

def require_fragments(path: Path, fragments: tuple[str, ...]) -> str:
    text = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        fail(f"{path}: missing deployment controls: {', '.join(missing)}")
    return text


def validate_deployment(root: Path) -> None:
    root = root.resolve(strict=True)
    systemd = root / "systemd"
    broker = require_fragments(systemd / "skarbiec-capability-broker.service", (
        "User=skarbiec-capability",
        "Group=skarbiec-capability-clients",
        "RuntimeDirectoryMode=0750",
        "ProtectSystem=strict",
        "NoNewPrivileges=true",
        "CapabilityBoundingSet=\n",
        "RestrictAddressFamilies=AF_UNIX",
        "IPAddressDeny=any",
        "SocketBindDeny=any",
    ))
    if "PrivateUsers=" in broker:
        fail("broker must see host peer UID/GID; PrivateUsers is forbidden")
    require_fragments(systemd / "wisent-agent@.service", (
        "User=wisent-agent-%i",
        "SupplementaryGroups=skarbiec-capability-clients",
        "RuntimeDirectoryMode=0700",
        "StateDirectoryMode=0700",
        "ProtectSystem=strict",
        "NoNewPrivileges=true",
        "CapabilityBoundingSet=\n",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=any",
        "SocketBindDeny=any",
    ))
    sysusers = require_fragments(systemd / "capabilities.sysusers", (
        "g skarbiec-capability-clients -",
        "m skarbiec-capability skarbiec-capability-clients",
    ))
    tmpfiles = (systemd / "capabilities.tmpfiles").read_text(encoding="utf-8")
    for target in sorted(CONTRACT_TARGETS):
        account = f"wisent-agent-{target}"
        if f"u {account} " not in sysusers or f"m {account} {CLIENT_GROUP}" not in sysusers:
            fail(f"missing dedicated UID/client-group membership for {target}")
        if f"d /etc/wisent/agents/{target} 0700 {account} {account} -" not in tmpfiles:
            fail(f"missing owner-only configuration directory for {target}")
        allowlist = systemd / f"wisent-agent@{target}.service.d" / "20-egress-allowlist.conf.example"
        allowlist_text = require_fragments(allowlist, ("IPAddressDeny=any", "IPAddressAllow="))
        if "IPAddressAllow=0.0.0.0/0" in allowlist_text or "IPAddressAllow=::/0" in allowlist_text:
            fail(f"broad egress route is forbidden: {allowlist}")
    require_fragments(systemd / "capabilities.tmpfiles", (
        "d /etc/wisent/capabilities 0700 skarbiec-capability skarbiec-capability -",
        "d /run/skarbiec-capability 0750 skarbiec-capability skarbiec-capability-clients -",
    ))
    require_fragments(root / "environment" / "skarbiec-broker.env.example", (
        "SKARBIEC_CAP_SOCKET_GID=",
        "SKARBIEC_CAP_POLICY=",
        "SKARBIEC_WORKLOAD_REGISTRY=",
    ))
    require_fragments(root / "environment" / "wisent-agent.env.example", (
        "SKARBIEC_CAP_SOCKET=",
        "SKARBIEC_WORKLOAD_ID=",
        "SKARBIEC_WORKLOAD_SIGNING_KEY_FILE=",
    ))
    broker_env = root / "environment" / "skarbiec-broker.env.example"
    broker_env_names = tuple(
        line.split("=", 1)[0]
        for line in broker_env.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and line.split("=", 1)[0] in SKARBIEC_MCP_ENV_NAMES
    )
    if broker_env_names != SKARBIEC_MCP_ENV_NAMES:
        fail("broker environment example must preserve exact signed Skarbiec MCP env_names order")
    require_fragments(root / "launchd" / "com.wisent.skarbiec-capability-broker.plist.template", (
        "<string>broker-macos</string>",
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("broker-linux", "broker-macos", "agent", "deployment-static"))
    parser.add_argument("environment_file", type=Path, nargs="?")
    parser.add_argument("--exec", action="store_true", dest="launch")
    args = parser.parse_args()
    if args.kind == "deployment-static":
        if args.launch:
            fail("--exec is invalid for deployment-static")
        validate_deployment(args.environment_file or Path(__file__).resolve().parent)
        return 0
    if args.kind == "broker-linux" and not sys.platform.startswith("linux"):
        fail("broker-linux requires Linux systemd egress enforcement")
    if args.kind == "broker-macos" and sys.platform != "darwin":
        fail("broker-macos is only valid on macOS")
    if args.kind == "broker-macos":
        fail("launchd has no egress sandbox; broker startup requires an externally enforced deny-all sandbox")
    if args.environment_file is None:
        fail("environment_file is required")
    values = load_env(args.environment_file)
    if args.kind == "broker-linux":
        validate_broker(values)
        executable = values["SKARBIEC_BINARY"]
        argv = [executable, "capability-serve"]
    else:
        validate_agent(values)
        executable = values["SINGULARITY_BOOTSTRAP_BINARY"]
        argv = [executable]
    if args.launch:
        os.environ.clear()
        os.environ.update(values)
        os.environ["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
        os.environ["LANG"] = "C.UTF-8"
        os.environ["LC_ALL"] = "C.UTF-8"
        os.execv(executable, argv)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"capability-preflight: {error}", file=sys.stderr)
        raise SystemExit(78)
