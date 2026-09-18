// Repository policy stories, attached to src/repo_surface/policy.rs.

use super::*;

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("wisent-policy-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(path.join(".git")).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn warsztat_contract_policy_binds_head_owner_to_repository_owner() {
    let root = TestDirectory::new();
    let mut policy = RepoPolicy {
        root: root.0.clone(),
        remote: "origin".into(),
        base_branch: "main".into(),
        branch_prefix: "proposal/".into(),
        github_repository: "wisent-ai/singularity".into(),
        github_head_owner: "attacker".into(),
        allowed_paths: vec![PathBuf::from("src")],
        checks: BTreeMap::new(),
        required_checks: Vec::new(),
    };

    let error = policy.validate().unwrap_err();

    assert_eq!(
        error.to_string(),
        "policy_denied: github_head_owner must match the owner of github_repository"
    );
}
