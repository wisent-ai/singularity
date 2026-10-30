//! A being launched with no Brama address takes the model service Stado's
//! capability resolution hands it, through the real `singularity doctor`.
//! The real Stado `SINGULARITY_TEST_STADO` runs first on PATH in a home of
//! the case's own, whose registry places Brama nowhere. With nothing declared
//! the launch is refused naming every inference alternative. With the real
//! OpenAI-compatible endpoint `SINGULARITY_TEST_MODEL_SERVICE_URL` declared in
//! Brama's place, it serves `SINGULARITY_TEST_MODEL_SERVICE_MODEL`, and its key
//! is read from the owner-only file `SINGULARITY_TEST_MODEL_SERVICE_KEY_FILE`
//! into the case's own credential store as a role. Doctor checks that
//! endpoint's health and models and names it as the model service, with no
//! Brama credential declared. Las is the real one the release fixture names:
//! `SINGULARITY_TEST_LAS_COMMAND`, `_ENTRYPOINT`, `_ONLY`, `_MANIFEST`,
//! `_SIGNATURE`, `_TRUST_STORE` and `_REQUIRED_SURFACES` (surfaces the signed
//! release serves).
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::Value;

fn declared(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("declare {name}"))
}

/// The case's root under the package's build directory.
fn root(case: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".build")
        .join(format!("resolved-{case}-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).expect("remove the previous case");
    }
    for directory in ["home/tmp", "home/storage", "bin", "state", "workspace"] {
        std::fs::create_dir_all(root.join(directory)).expect("create the case's directories");
    }
    std::os::unix::fs::symlink(declared("SINGULARITY_TEST_STADO"), root.join("bin/stado"))
        .expect("link the real stado");
    root
}

/// The environment every command of the case runs in: its own home, the
/// real Stado first on PATH, and no inherited route or credential.
fn isolate(command: &mut Command, root: &Path) {
    let mut path = root.join("bin").into_os_string();
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    command
        .env_clear()
        .env("HOME", root.join("home"))
        .env("TMPDIR", root.join("home/tmp"))
        .env("WC_STORAGE_BACKEND", "local")
        .env("WC_LOCAL_STORAGE_PATH", root.join("home/storage"))
        .env("PATH", path);
}

fn stado(root: &Path, arguments: &[&str], input: &str) -> String {
    let mut command = Command::new(root.join("bin/stado"));
    isolate(&mut command, root);
    let mut child = command
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start stado");
    child
        .stdin
        .take()
        .expect("stado's stdin")
        .write_all(input.as_bytes())
        .expect("write stado's stdin");
    let output = child.wait_with_output().expect("wait for stado");
    assert!(
        output.status.success(),
        "stado {} refused: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// `singularity doctor` with every declaration but Brama's address, model
/// and credentials.
fn doctor(root: &Path) -> Output {
    let digest = "a".repeat(64);
    let mut command = Command::new(env!("CARGO_BIN_EXE_singularity"));
    isolate(&mut command, root);
    command.arg("doctor");
    for (flag, value) in [
        ("--agent-id", "resolved-test"),
        ("--agent-name", "Resolved test"),
        ("--agent-ticker", "RSLV"),
        ("--role", "test"),
        ("--environment", "test"),
        ("--host", "test-host"),
        ("--workload-id", "test-workload"),
        ("--workload-public-key", digest.as_str()),
        ("--executable-digest", digest.as_str()),
        ("--code-digest", digest.as_str()),
        ("--policy-digest", digest.as_str()),
        ("--policy-sequence", "1"),
        ("--starting-balance", "1"),
        ("--instance-price", "0"),
        ("--cycle-interval-secs", "60"),
    ] {
        command.args([flag, value]);
    }
    for (flag, variable) in [
        ("--las-command", "SINGULARITY_TEST_LAS_COMMAND"),
        ("--las-entrypoint", "SINGULARITY_TEST_LAS_ENTRYPOINT"),
        ("--las-only", "SINGULARITY_TEST_LAS_ONLY"),
        ("--las-release-manifest", "SINGULARITY_TEST_LAS_MANIFEST"),
        (
            "--las-release-manifest-signature",
            "SINGULARITY_TEST_LAS_SIGNATURE",
        ),
        (
            "--las-release-trust-store",
            "SINGULARITY_TEST_LAS_TRUST_STORE",
        ),
        (
            "--required-surfaces",
            "SINGULARITY_TEST_LAS_REQUIRED_SURFACES",
        ),
    ] {
        command.args([flag, declared(variable).as_str()]);
    }
    command
        .arg("--las-release-watermark")
        .arg(root.join("las-watermark.json"))
        .arg("--state-dir")
        .arg(root.join("state"))
        .arg("--workspace")
        .arg(root.join("workspace"));
    command.output().expect("run the real singularity CLI")
}

#[test]
fn with_no_brama_address_the_being_takes_the_model_service_stado_resolves() {
    let root = root("doctor");
    std::fs::write(
        root.join("las-watermark.json"),
        "{\"version\":1,\"sequence\":0}\n",
    )
    .expect("write the watermark");
    std::fs::set_permissions(
        root.join("las-watermark.json"),
        std::os::unix::fs::PermissionsExt::from_mode(0o600),
    )
    .expect("make the watermark owner-only");
    stado(&root, &["config", "init"], "");

    // Nothing answers: the launch is refused naming every alternative.
    let refused = doctor(&root);
    let said = String::from_utf8_lossy(&refused.stderr).into_owned();
    assert!(!refused.status.success(), "doctor answered: {refused:?}");
    for named in [
        "no BRAMA_BASE_URL (--brama-url) is named and no inference alternative answers singularity",
        "brama at ",
        "model-endpoint at capabilities.inference.endpoint.url",
    ] {
        assert!(said.contains(named), "{named:?} is not named: {said}");
    }

    // The real endpoint answers in Brama's place.
    let model = declared("SINGULARITY_TEST_MODEL_SERVICE_MODEL");
    let key = std::fs::read_to_string(declared("SINGULARITY_TEST_MODEL_SERVICE_KEY_FILE"))
        .expect("read the model service key");
    let store = root.join("credentials.json");
    stado(
        &root,
        &[
            "config",
            "set",
            "credentials.store",
            &format!("file://{}", store.display()),
        ],
        "",
    );
    stado(
        &root,
        &[
            "credentials",
            "put",
            "--role",
            "singularity-journey-endpoint",
        ],
        &serde_json::json!({ "token": key.trim() }).to_string(),
    );
    for (field, value) in [
        (
            "capabilities.inference.endpoint.url",
            declared("SINGULARITY_TEST_MODEL_SERVICE_URL"),
        ),
        ("capabilities.inference.endpoint.model", model.clone()),
        (
            "capabilities.inference.endpoint.key_role",
            "singularity-journey-endpoint".to_owned(),
        ),
    ] {
        stado(&root, &["config", "set", field, &value], "");
    }
    let answered = doctor(&root);
    assert!(
        answered.status.success(),
        "doctor refused: {}",
        String::from_utf8_lossy(&answered.stderr)
    );
    let report: Value = serde_json::from_slice(&answered.stdout).expect("doctor answers JSON");
    assert_eq!(report["model"], model.as_str(), "{report}");
    assert_eq!(
        report["model_service"]["provider"], "model-endpoint",
        "{report}"
    );
    let record: Value =
        serde_json::from_str(&stado(&root, &["capabilities", "inference", "--json"], ""))
            .expect("stado answers JSON");
    assert_eq!(
        record["alternatives"][0]["served"]["singularity"]["provider"], "model-endpoint",
        "{record}"
    );
    std::fs::remove_dir_all(&root).expect("remove the case");
}
