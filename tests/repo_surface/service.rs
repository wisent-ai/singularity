// Repository service stories, attached to src/repo_surface/service.rs.

use std::collections::BTreeMap;

use super::*;
use std::fs;
use std::path::Path;

use serde_json::json;

use super::checks::{fresh_seal, write_tree};
use crate::repo_surface::command::git;

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("wisent-service-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn git_ok(cwd: &Path, args: &[&str]) -> super::super::command::CommandOutput {
    let output = git(cwd, args, None, 30).await.unwrap();
    assert!(
        output.success,
        "git {args:?} failed with stderr: {}",
        output.stderr
    );
    output
}

#[tokio::test]
async fn warsztat_contract_staged_tree_change_invalidates_seal() {
    let directory = TestDirectory::new();
    git_ok(&directory.0, &["init", "--quiet"]).await;
    let tracked = directory.0.join("proposal.txt");
    fs::write(&tracked, "sealed\n").unwrap();
    git_ok(&directory.0, &["add", "--", "proposal.txt"]).await;
    let sealed = write_tree(&directory.0).await.unwrap();
    let state = WorkspaceState {
        id: "workspace".into(),
        repo_id: "repo".into(),
        branch: "proposal/workspace".into(),
        base_commit: String::new(),
        worktree: directory.0.clone(),
        created_at: String::new(),
        sealed_fingerprint: Some(sealed),
        checks: BTreeMap::new(),
        commit: None,
        published: false,
    };
    fs::write(tracked, "changed after seal\n").unwrap();
    git_ok(&directory.0, &["add", "--", "proposal.txt"]).await;

    let error = fresh_seal(&state).await.unwrap_err();

    assert_eq!(
        error.to_string(),
        "invalid_state: staged tree changed after seal"
    );
}

#[test]
fn warsztat_contract_request_replay_requires_identical_input() {
    let directory = TestDirectory::new();
    let state = StateStore::open(directory.0.join("state")).unwrap();
    let service = RepoService::new(
        PolicyFile {
            repositories: BTreeMap::new(),
        },
        state.clone(),
    );
    let response = json!({"workspace_id":"workspace","applied":true});
    state
        .save_request(
            "request",
            &RequestRecord {
                operation: "workspace_apply_patch".into(),
                workspace_id: "workspace".into(),
                input_fingerprint: "fingerprint".into(),
                response: response.clone(),
            },
        )
        .unwrap();

    assert_eq!(
        service
            .replay(
                "request",
                "workspace_apply_patch",
                "workspace",
                "fingerprint"
            )
            .unwrap(),
        Some(response)
    );
    let error = service
        .replay(
            "request",
            "workspace_apply_patch",
            "workspace",
            "different-fingerprint",
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid_state: request_id was already used for different input"
    );
}
