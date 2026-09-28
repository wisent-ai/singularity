//! `deployment-static`: the launchd and systemd manifests this repository
//! ships under deploy/capabilities still carry every isolation control the
//! capability contract depends on.

use std::path::Path;

use super::contract::{Check, CLIENT_GROUP, CONTRACT_TARGETS, SKARBIEC_MCP_ENV_NAMES};

fn read(path: &Path) -> Check<String> {
    std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn require_fragments(path: &Path, fragments: &[&str]) -> Check<String> {
    let text = read(path)?;
    let missing: Vec<&str> = fragments
        .iter()
        .copied()
        .filter(|fragment| !text.contains(fragment))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{}: missing deployment controls: {}",
            path.display(),
            missing.join(", ")
        ));
    }
    Ok(text)
}

pub(super) fn validate(root: &Path) -> Check<()> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("{}: {error}", root.display()))?;
    let systemd = root.join("systemd");
    let broker = require_fragments(
        &systemd.join("skarbiec-capability-broker.service"),
        &[
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
        ],
    )?;
    if broker.contains("PrivateUsers=") {
        return Err("broker must see host peer UID/GID; PrivateUsers is forbidden".into());
    }
    require_fragments(
        &systemd.join("wisent-agent@.service"),
        &[
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
        ],
    )?;
    let sysusers = require_fragments(
        &systemd.join("capabilities.sysusers"),
        &[
            "g skarbiec-capability-clients -",
            "m skarbiec-capability skarbiec-capability-clients",
        ],
    )?;
    let tmpfiles = require_fragments(
        &systemd.join("capabilities.tmpfiles"),
        &[
            "d /etc/wisent/capabilities 0700 skarbiec-capability skarbiec-capability -",
            "d /run/skarbiec-capability 0750 skarbiec-capability skarbiec-capability-clients -",
        ],
    )?;
    for target in CONTRACT_TARGETS {
        let account = format!("wisent-agent-{target}");
        if !sysusers.contains(&format!("u {account} "))
            || !sysusers.contains(&format!("m {account} {CLIENT_GROUP}"))
        {
            return Err(format!(
                "missing dedicated UID/client-group membership for {target}"
            ));
        }
        if !tmpfiles.contains(&format!(
            "d /etc/wisent/agents/{target} 0700 {account} {account} -"
        )) {
            return Err(format!(
                "missing owner-only configuration directory for {target}"
            ));
        }
        let allowlist = systemd
            .join(format!("wisent-agent@{target}.service.d"))
            .join("20-egress-allowlist.conf.example");
        let text = require_fragments(&allowlist, &["IPAddressDeny=any", "IPAddressAllow="])?;
        if text.contains("IPAddressAllow=0.0.0.0/0") || text.contains("IPAddressAllow=::/0") {
            return Err(format!(
                "broad egress route is forbidden: {}",
                allowlist.display()
            ));
        }
    }
    let environment = root.join("environment");
    let broker_env = require_fragments(
        &environment.join("skarbiec-broker.env.example"),
        &[
            "SKARBIEC_CAP_SOCKET_GID=",
            "SKARBIEC_CAP_POLICY=",
            "SKARBIEC_WORKLOAD_REGISTRY=",
        ],
    )?;
    require_fragments(
        &environment.join("wisent-agent.env.example"),
        &[
            "SKARBIEC_CAP_SOCKET=",
            "SKARBIEC_WORKLOAD_ID=",
            "SKARBIEC_WORKLOAD_SIGNING_KEY_FILE=",
        ],
    )?;
    let declared: Vec<&str> = broker_env
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.split_once('=').map_or(line, |(name, _)| name))
        .filter(|name| SKARBIEC_MCP_ENV_NAMES.contains(name))
        .collect();
    if declared != SKARBIEC_MCP_ENV_NAMES {
        return Err(
            "broker environment example must preserve exact signed Skarbiec MCP env_names order"
                .into(),
        );
    }
    require_fragments(
        &root
            .join("launchd")
            .join("com.wisent.skarbiec-capability-broker.plist.template"),
        &["<string>broker-macos</string>"],
    )?;
    Ok(())
}
