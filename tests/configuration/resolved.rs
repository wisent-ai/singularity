//! A being launched with no Brama address hands its model requests to
//! Stado's inference alternatives, through the real `singularity` CLI. The
//! real Stado `SINGULARITY_TEST_STADO` runs first on PATH in a home of the
//! case's own, whose registry places Brama nowhere. With no Stado at all,
//! `doctor` is refused naming the command that could not run; with Stado,
//! `doctor` names Stado and the order of its alternatives. One real `once`
//! cycle is refused with Stado's sentence (Brama's cause and the command that
//! answers the kept request); a statement recorded for exactly that request
//! then answers the same cycle, which completes, and Stado records that the
//! attestation alternative served the being. No Brama address, credential or
//! catalog is declared. Las is the real one a signed release names:
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
    for directory in ["home/tmp", "home/storage", "bin", "no-stado", "workspace"] {
        std::fs::create_dir_all(root.join(directory)).expect("create the case's directories");
    }
    std::os::unix::fs::symlink(declared("SINGULARITY_TEST_STADO"), root.join("bin/stado"))
        .expect("link the real stado");
    root
}

/// The environment every command of the case runs in: its own home, the
/// real Stado first on PATH (or none, when `bin` is the empty directory),
/// and no inherited route or credential.
fn isolate(command: &mut Command, root: &Path, bin: &str) {
    let mut path = root.join(bin).into_os_string();
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    command
        .env_clear()
        .env("HOME", root.join("home"))
        .env("TMPDIR", root.join("home/tmp"))
        .env("WC_STORAGE_BACKEND", "local")
        .env("WC_LOCAL_STORAGE_PATH", root.join("home/storage"))
        .env("STADO_BIN", root.join(bin).join("stado"))
        .env("PATH", path);
}

fn stado(root: &Path, arguments: &[&str], input: &str) -> String {
    let mut command = Command::new(root.join("bin/stado"));
    isolate(&mut command, root, "bin");
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

/// `singularity <verb>` with every declaration but Brama's address and
/// credentials, its state in `state`, Stado found in `bin`.
fn singularity(root: &Path, verb: &str, bin: &str, state: &str) -> Output {
    let digest = "a".repeat(64);
    let mut command = Command::new(env!("CARGO_BIN_EXE_singularity"));
    isolate(&mut command, root, bin);
    command.arg(verb);
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
        ("--brama-model", "any"),
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
    std::fs::create_dir_all(root.join(state)).expect("create the state directory");
    command
        .arg("--las-release-watermark")
        .arg(root.join("las-watermark.json"))
        .arg("--state-dir")
        .arg(root.join(state))
        .arg("--workspace")
        .arg(root.join("workspace"));
    let output = command.output().expect("run the real singularity CLI");
    std::fs::write(
        root.join(format!("{verb}-{state}-{bin}.json")),
        serde_json::json!({
            "command": format!("{command:?}"), "status": output.status.to_string(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
        })
        .to_string(),
    )
    .expect("keep the run");
    output
}

#[test]
fn with_no_brama_address_the_being_hands_its_requests_to_stados_alternatives() {
    let root = root("call");
    // The watermark the signed release's sequence must exceed, owner-only.
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

    // No Stado to hand requests to: refused naming the command.
    let absent = singularity(&root, "doctor", "no-stado", "doctor");
    let said = String::from_utf8_lossy(&absent.stderr).into_owned();
    assert!(!absent.status.success(), "doctor answered: {absent:?}");
    assert!(
        said.contains("no BRAMA_BASE_URL (--brama-url) is named and `")
            && said.contains("capabilities inference --json` could not run"),
        "{said}"
    );

    stado(&root, &["config", "init"], "");
    let checked = singularity(&root, "doctor", "bin", "doctor");
    assert!(
        checked.status.success(),
        "doctor refused: {}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let report: Value = serde_json::from_slice(&checked.stdout).expect("doctor answers JSON");
    assert_eq!(report["model_service"]["provider"], "stado", "{report}");
    assert_eq!(
        report["model_service"]["order"],
        serde_json::json!(["brama", "attestation"]),
        "{report}"
    );

    // Nothing answers the cycle's request yet: Stado's sentence.
    let refused = singularity(&root, "once", "bin", "refused");
    let said = String::from_utf8_lossy(&refused.stderr).into_owned();
    assert!(!refused.status.success(), "once answered: {refused:?}");
    for named in [
        "no inference alternative answered `",
        "capabilities call inference --consumer singularity --json`",
        "brama at ",
        "stado capabilities attestation record inference --consumer singularity",
    ] {
        assert!(said.contains(named), "{named:?} is not named: {said}");
    }

    // The statement for exactly that request answers the same cycle.
    let kept: Value = serde_json::from_str(&stado(
        &root,
        &[
            "capabilities",
            "attestation",
            "list",
            "inference",
            "--consumer",
            "singularity",
            "--json",
        ],
        "",
    ))
    .expect("the attestation list is JSON");
    let request = kept[0]["request"]
        .as_str()
        .expect("one kept request")
        .to_owned();
    stado(
        &root,
        &[
            "capabilities",
            "attestation",
            "record",
            "inference",
            "--consumer",
            "singularity",
            &request,
            "--checked",
            "the opening cycle has no opportunity worth a tool call",
        ],
        "Nothing to act on in this cycle.",
    );
    let answered = singularity(&root, "once", "bin", "answered");
    assert!(
        answered.status.success(),
        "once refused: {}",
        String::from_utf8_lossy(&answered.stderr)
    );
    // The cycle's report carries the statement's text as the model's answer.
    let printed = String::from_utf8_lossy(&answered.stdout);
    assert!(
        printed.contains("Nothing to act on in this cycle."),
        "{printed}"
    );
    let record: Value =
        serde_json::from_str(&stado(&root, &["capabilities", "inference", "--json"], ""))
            .expect("stado answers JSON");
    assert_eq!(
        record["alternatives"][0]["served"]["singularity"]["provider"], "attestation",
        "{record}"
    );
    std::fs::remove_dir_all(&root).expect("remove the case");
}
