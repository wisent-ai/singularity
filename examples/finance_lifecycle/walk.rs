//! The walkthrough's session: the MCP child it talks JSON-RPC to, the owner
//! events it signs and applies, and the leases and WORM receipts it writes.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use chrono::{DateTime, Duration, Utc};
use serde_json::{json, Value};

use super::signing::{key_from, ts, write_owner_only, Key, Outcome, Progress, POLICY_ID, VERSION};

fn summary(value: &Value, keys: &[&str]) {
    let picked: serde_json::Map<String, Value> = keys
        .iter()
        .filter_map(|key| {
            value
                .get(*key)
                .map(|found| (key.to_string(), found.clone()))
        })
        .collect();
    println!("<- {}\n", Value::Object(picked));
}

pub struct Walk {
    pub dir: PathBuf,
    pub keys: Vec<Key>,
    pub progress: Progress,
    mcp: PathBuf,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    child: Child,
    next_id: u64,
}

impl Walk {
    pub fn start(dir: PathBuf, mcp: PathBuf, progress: Progress) -> Outcome<Self> {
        let keys = progress
            .seeds
            .iter()
            .map(|seed| key_from(seed))
            .collect::<Outcome<Vec<_>>>()?;
        let mut child = Command::new(&mcp)
            .envs(environment(&dir, &keys[0]))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().ok_or("the MCP child has no stdin")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("the MCP child has no stdout")?);
        let next_id = u64::default();
        let mut walk = Self {
            dir,
            keys,
            progress,
            mcp,
            stdin,
            stdout,
            child,
            next_id,
        };
        let info = walk.rpc(
            "initialize",
            json!({"protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "walkthrough", "version": "0"}}),
        )?;
        let tools = walk.rpc("tools/list", json!({}))?;
        let names: Vec<&str> = tools["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|tool| tool["name"].as_str())
            .collect();
        println!(
            "server: {} {}\ntools: {}\n",
            info["serverInfo"]["name"],
            info["serverInfo"]["version"],
            names.join(", ")
        );
        Ok(walk)
    }

    pub fn finish(mut self) -> Outcome<()> {
        write_owner_only(
            &self.dir.join("progress.json"),
            &serde_json::to_string(&self.progress)?,
        )?;
        drop(self.stdin);
        self.child.wait()?;
        Ok(())
    }

    fn rpc(&mut self, method: &str, params: Value) -> Outcome<Value> {
        self.next_id += 1;
        writeln!(
            self.stdin,
            "{}",
            json!({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        )?;
        self.stdin.flush()?;
        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        let answer: Value = serde_json::from_str(&line)?;
        answer
            .get("result")
            .cloned()
            .ok_or_else(|| format!("{method} answered without a result: {line}").into())
    }

    pub fn call(&mut self, label: &str, name: &str, arguments: Value) -> Outcome<Option<Value>> {
        println!("### {label}\n-> {arguments}");
        let result = self.rpc("tools/call", json!({"name": name, "arguments": arguments}))?;
        if result["isError"].as_bool() == Some(true) {
            println!(
                "<- ERROR {}\n",
                result["content"][0]["text"].as_str().unwrap_or_default()
            );
            return Ok(None);
        }
        let value = result["structuredContent"].clone();
        summary(
            &value,
            &[
                "transaction_id",
                "status",
                "intent_hash",
                "approval_count",
                "timelock_until",
                "reconciliation_required",
            ],
        );
        Ok(Some(value))
    }

    pub fn owner_event(&mut self, label: &str, action: Value, at: Option<String>) -> Outcome<()> {
        self.progress.events += 1;
        let event = json!({
            "event_id": format!("evt-{:03}", self.progress.events),
            "transaction_id": self.progress.tx, "intent_hash": self.progress.hash,
            "occurred_at": at.unwrap_or_else(|| ts(Utc::now())), "action": action,
        });
        let name = format!("evt-{:03}.json", self.progress.events);
        let path = self.dir.join(&name);
        write_owner_only(&path, &self.keys[0].envelope(&event)?)?;
        println!("$ singularity-finance-mcp owner-event {name}   # {label}");
        let run = Command::new(&self.mcp)
            .arg("owner-event")
            .arg(&path)
            .envs(environment(&self.dir, &self.keys[0]))
            .output()?;
        if !run.status.success() {
            println!("<- ERROR {}\n", String::from_utf8_lossy(&run.stderr).trim());
            return Ok(());
        }
        summary(
            &serde_json::from_slice(&run.stdout)?,
            &[
                "status",
                "approval_count",
                "timelock_until",
                "reconciliation_required",
            ],
        );
        Ok(())
    }

    pub fn lease(&self, id: &str, issued: DateTime<Utc>, kill_switch: bool) -> Outcome<()> {
        let lease = json!({
            "policy_id": POLICY_ID, "policy_version": VERSION, "lease_id": id,
            "issued_at": ts(issued), "expires_at": ts(issued + Duration::hours(1)),
            "enabled": true, "kill_switch": kill_switch,
        });
        write_owner_only(
            &self.dir.join("lease.json"),
            &self.keys[0].envelope(&lease)?,
        )
    }

    /// A WORM receipt signed by the receipt key (the last of the seven).
    pub fn receipt(&self, kind: &str, reference: &str, at: &str) -> Outcome<String> {
        let receipt = json!({
            "sink_id": "walkthrough-worm", "receipt_id": format!("receipt-{kind}"), "event_kind": kind,
            "transaction_id": self.progress.tx, "intent_hash": self.progress.hash,
            "reference_hash": reference, "recorded_at": at,
        });
        let path = self.dir.join(format!("receipt-{kind}.json"));
        write_owner_only(&path, &self.keys[6].envelope(&receipt)?)?;
        Ok(path.display().to_string())
    }
}

fn environment(dir: &Path, document: &Key) -> Vec<(String, String)> {
    let path = |name: &str| dir.join(name).display().to_string();
    vec![
        (
            "SINGULARITY_FINANCE_POLICY_FILE".into(),
            path("policy.json"),
        ),
        (
            "SINGULARITY_FINANCE_ENABLE_LEASE_FILE".into(),
            path("lease.json"),
        ),
        ("SINGULARITY_FINANCE_STATE_DIR".into(), path("state")),
        (
            "SINGULARITY_FINANCE_VERIFY_KEY_HEX".into(),
            document.public(),
        ),
        (
            "SINGULARITY_FINANCE_EXECUTOR".into(),
            "/usr/bin/false".into(),
        ),
    ]
}
