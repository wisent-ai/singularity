"""Checking the launchd and systemd manifests the deployment ships."""

from pathlib import Path

from preflight.contract import (
    CLIENT_GROUP,
    CONTRACT_TARGETS,
    SKARBIEC_MCP_ENV_NAMES,
    fail,
)
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
        "serve --no-http",
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


