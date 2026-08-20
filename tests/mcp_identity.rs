use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use singularity::{AppError, ErrorClass, LasSupervisor};

const REQUEST_DEADLINE: Duration = Duration::from_secs(10);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(1);
const AMBIENT_CHILD_MARKER: &str = "SINGULARITY_MCP_IDENTITY_AMBIENT_CHILD";

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct McpFixture {
    directory: PathBuf,
    script: PathBuf,
}

impl McpFixture {
    fn new() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "singularity-mcp-identity-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let script = directory.join("identity-mcp.sh");
        std::fs::write(
            &script,
            r#"initialize_agent_id=''
while IFS= read -r request
do
    case "$request" in
        *'"method":"initialize"'*)
            initialize_agent_id=$(printf '%s\n' "$request" | sed -n 's/.*"agentId":"\([^"]*\)".*/\1/p')
            printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05"}}'
            ;;
        *'"method":"tools/list"'*)
            printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"identity__read","inputSchema":{"type":"object"}}]}}'
            ;;
        *'"method":"tools/call"'*)
            printf '{"jsonrpc":"2.0","id":3,"result":{"environmentAgentId":"%s","initializeAgentId":"%s"}}\n' "$SKARBIEC_MCP_AGENT_ID" "$initialize_agent_id"
            ;;
    esac
done
"#,
        )
        .unwrap();
        Self { directory, script }
    }
}

impl Drop for McpFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

async fn spawn_skarbiec(
    fixture: &McpFixture,
    agent_id: Option<&str>,
) -> Result<LasSupervisor, AppError> {
    LasSupervisor::spawn(
        "/bin/sh",
        &fixture.script,
        "skarbiec",
        None,
        agent_id,
        &fixture.script,
        &fixture.script,
        &fixture.script,
        &fixture.script,
        &[],
        REQUEST_DEADLINE,
    )
    .await
}

async fn observed_identity(supervisor: &mut LasSupervisor) -> Value {
    supervisor
        .call_tool("identity__read", json!({}))
        .await
        .unwrap()
}

#[tokio::test]
async fn distinct_supervisors_bind_their_passed_identity_to_initialization_and_child_environment() {
    let fixture = McpFixture::new();
    let mut alpha = spawn_skarbiec(&fixture, Some("agent-alpha")).await.unwrap();
    let mut beta = spawn_skarbiec(&fixture, Some("agent-beta")).await.unwrap();

    assert_eq!(
        observed_identity(&mut alpha).await,
        json!({
            "environmentAgentId": "agent-alpha",
            "initializeAgentId": "agent-alpha",
        })
    );
    assert_eq!(
        observed_identity(&mut beta).await,
        json!({
            "environmentAgentId": "agent-beta",
            "initializeAgentId": "agent-beta",
        })
    );

    alpha.shutdown(SHUTDOWN_GRACE).await.unwrap();
    beta.shutdown(SHUTDOWN_GRACE).await.unwrap();
}

#[test]
fn ambient_agent_identity_cannot_override_the_explicit_supervisor_identity() {
    let status = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("ambient_spoof_child_observes_only_the_explicit_identity")
        .arg("--ignored")
        .env(AMBIENT_CHILD_MARKER, "1")
        .env("SKARBIEC_MCP_AGENT_ID", "spoofed-global-agent")
        .status()
        .unwrap();

    assert!(status.success(), "isolated ambient-spoof assertion failed");
}

#[tokio::test]
#[ignore = "invoked in an isolated subprocess by ambient_agent_identity_cannot_override_the_explicit_supervisor_identity"]
async fn ambient_spoof_child_observes_only_the_explicit_identity() {
    assert_eq!(std::env::var(AMBIENT_CHILD_MARKER).as_deref(), Ok("1"));
    assert_eq!(
        std::env::var("SKARBIEC_MCP_AGENT_ID").as_deref(),
        Ok("spoofed-global-agent")
    );

    let fixture = McpFixture::new();
    let mut supervisor = spawn_skarbiec(&fixture, Some("trusted-passed-agent"))
        .await
        .unwrap();

    assert_eq!(
        observed_identity(&mut supervisor).await,
        json!({
            "environmentAgentId": "trusted-passed-agent",
            "initializeAgentId": "trusted-passed-agent",
        })
    );
    supervisor.shutdown(SHUTDOWN_GRACE).await.unwrap();
}

#[tokio::test]
async fn active_skarbiec_rejects_missing_or_malformed_identity_before_process_spawn() {
    let fixture_path = Path::new("/fixture-would-not-be-read");
    let invalid_identities = [
        (
            "explicit Skarbiec selection",
            "skarbiec",
            None,
            "Skarbiec requires an explicit immutable Las agent identity",
        ),
        (
            "empty LAS_ONLY selects all surfaces",
            "",
            None,
            "Skarbiec requires an explicit immutable Las agent identity",
        ),
        (
            "empty identity",
            "skarbiec",
            Some(""),
            "invalid immutable Las agent identity",
        ),
        (
            "wildcard identity",
            "skarbiec",
            Some("*"),
            "invalid immutable Las agent identity",
        ),
        (
            "path-like identity",
            "skarbiec",
            Some("agent/escape"),
            "invalid immutable Las agent identity",
        ),
        (
            "identity exceeding 128 bytes",
            "skarbiec",
            Some(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            "invalid immutable Las agent identity",
        ),
    ];

    for (case, only, agent_id, expected_message) in invalid_identities {
        let error = match LasSupervisor::spawn(
            "/definitely/not/a/real/mcp-command",
            fixture_path,
            only,
            None,
            agent_id,
            fixture_path,
            fixture_path,
            fixture_path,
            fixture_path,
            &[],
            REQUEST_DEADLINE,
        )
        .await
        {
            Ok(_) => panic!("{case} spawned a process for identity {agent_id:?}"),
            Err(error) => error,
        };

        assert!(
            matches!(
                error,
                AppError::Mcp {
                    class: ErrorClass::Permanent,
                    ref message,
                } if message == expected_message
            ),
            "unexpected rejection for {case} with identity {agent_id:?}: {error}"
        );
    }
}
