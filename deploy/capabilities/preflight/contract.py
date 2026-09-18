"""The deployment contract a capability process is checked against."""
import hashlib
import os
import re
import stat
from pathlib import Path

NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNSAFE_NAME = re.compile(r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY)")
SAFE_REFERENCE_SUFFIXES = ("_FILE", "_PATH", "_DIR", "_SOCKET")
REJECTED_VALUE = re.compile(r"(?i)(replace|placeholder|example|changeme|todo|insert[-_ ]?here|<[^>]+>)")
CONTRACT_TARGETS = frozenset(("weles", "most-service", "brama", "singularity-bootstrap"))
CLIENT_GROUP = "skarbiec-capability-clients"
MCP_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# The modes a capability-isolated deployment is held to: secret files grant the group
# and others nothing, private directories are 0700, shared ones 0750, executables are
# not group/world writable and are owner-executable, the broker socket is 0660.
GROUP_OR_OTHER_ACCESS = 0o077
PRIVATE_DIR_MODE = 0o700
SHARED_DIR_MODE = 0o750
GROUP_OR_OTHER_WRITE = 0o022
OWNER_EXECUTE = 0o100
BROKER_SOCKET_MODE = 0o660
HASH_BLOCK_BYTES = 1024 * 1024
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
        if not stat.S_ISREG(info.st_mode) or mode & GROUP_OR_OTHER_ACCESS:
            fail(f"owner-only regular file required: {path}")
    elif kind == "dir":
        if not stat.S_ISDIR(info.st_mode) or mode != PRIVATE_DIR_MODE:
            fail(f"0700 directory required: {path}")
    elif kind == "shared-dir":
        if not stat.S_ISDIR(info.st_mode) or mode != SHARED_DIR_MODE:
            fail(f"0750 shared directory required: {path}")
    elif kind == "executable":
        if not stat.S_ISREG(info.st_mode) or mode & GROUP_OR_OTHER_WRITE or not mode & OWNER_EXECUTE:
            fail(f"non-writable owner-executable regular file required: {path}")
    else:
        fail(f"internal error: unsupported path kind {kind}")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
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



# The launched process is given a fixed search path and the C locale; a configuration
# refusal exits 78 (EX_CONFIG).
LAUNCH_PATH = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
LAUNCH_LOCALE = "C.UTF-8"
CONFIGURATION_EXIT_CODE = 78
