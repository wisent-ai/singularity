//! `singularity ticket key|sign|launch` through the real binaries: a ticket the
//! issuer writes is accepted by `singularity-bootstrap` up to the broker it
//! names (a socket nobody listens on, so redemption is the first refusal),
//! and the issuer's refusals leave nothing behind. Every file lives under
//! Cargo's ignored `target/tmp` directory and is removed by the case.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use sha2::{Digest, Sha256};

const SINGULARITY: &str = env!("CARGO_BIN_EXE_singularity");
const BOOTSTRAP: &str = env!("CARGO_BIN_EXE_singularity-bootstrap");

/// One case's own directory under Cargo's `target/tmp`, removed when the case ends.
struct Case(PathBuf);

impl Case {
    fn start(name: &str) -> Self {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("ticket-{name}-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir_all(&root).unwrap();
        Self(root)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(program: &str, args: &[&str]) -> Output {
    Command::new(program).env_clear().args(args).output().unwrap()
}

fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// A distinct lowercase 64-hex id derived from `seed`.
fn hex_id(seed: &str) -> String {
    hex::encode(Sha256::digest(seed.as_bytes()))
}

/// Whether a file grants its group and others nothing.
fn owner_only(path: &Path) -> bool {
    let mode = fs::metadata(path).unwrap().permissions().mode();
    let shared = mode & u32::from(libc::S_IRWXG | libc::S_IRWXO);
    // `shared & !shared` has no bit set, so it equals `shared` only when
    // `shared` has none either.
    shared == shared & !shared
}

fn key(case: &Case, holder: &str) -> (PathBuf, PathBuf) {
    let (private, public) = (case.path(&format!("{holder}.key")), case.path(&format!("{holder}.pub")));
    let answer = run(
        SINGULARITY,
        &["ticket", "key", "--holder", holder, "--out", text(&private), "--public-out", text(&public)],
    );
    assert!(answer.status.success(), "{}", String::from_utf8_lossy(&answer.stderr));
    (private, public)
}

/// `ticket sign` with every input stated; `lifetime` is the seconds the ticket lives.
fn sign(case: &Case, lifetime: &str) -> (Output, PathBuf, PathBuf, PathBuf) {
    let (workload, _) = key(case, "workload");
    let (supervisor, trust_root) = key(case, "supervisor");
    let policy = case.path("policy.json");
    fs::write(&policy, "{}").unwrap();
    let (manifest, signature) = (case.path("manifest.json"), case.path("manifest.sig"));
    let (brama, bearer, most) = (hex_id("brama"), hex_id("bearer"), hex_id("most"));
    let code = hex_id("code");
    let socket = case.path("absent-broker.sock");
    let answer = run(
        SINGULARITY,
        &[
            "ticket", "sign",
            "--agent-id", "ticket-test", "--role", "test", "--environment", "test",
            "--host", "test-host", "--workload-id", "ticket-test-workload",
            "--workload-key", text(&workload), "--broker-socket", text(&socket),
            "--executable", SINGULARITY, "--code-digest", &code,
            "--policy-file", text(&policy), "--policy-sequence", "1",
            "--expires-in-seconds", lifetime,
            "--brama-capability", &brama, "--brama-resource", "brama:hmac",
            "--brama-bearer-capability", &bearer, "--brama-bearer-resource", "brama:bearer",
            "--most-capability", &most, "--most-resource", "most:token",
            "--supervisor-key", text(&supervisor), "--trust-root", text(&trust_root),
            "--manifest-out", text(&manifest), "--signature-out", text(&signature),
            "--", "doctor",
        ],
    );
    (answer, manifest, signature, trust_root)
}

#[test]
fn an_issued_ticket_is_accepted_by_the_bootstrap_up_to_its_broker() {
    let case = Case::start("issued");
    let (answer, manifest, signature, trust_root) = sign(&case, "300");
    assert!(answer.status.success(), "{}", String::from_utf8_lossy(&answer.stderr));
    let report: serde_json::Value = serde_json::from_slice(&answer.stdout).unwrap();
    assert_eq!(report["checked_by"], "singularity-bootstrap check_ticket");
    for written in [&manifest, &signature] {
        assert!(owner_only(written), "{} is owner-only", written.display());
    }
    let runtime = case.path("runtime");
    let started = run(
        BOOTSTRAP,
        &[
            "--manifest", text(&manifest), "--manifest-signature", text(&signature),
            "--trust-root", text(&trust_root), "--runtime-root", text(&runtime),
        ],
    );
    let stderr = String::from_utf8_lossy(&started.stderr);
    assert!(!started.status.success(), "{stderr}");
    // Signature, manifest rules and executable digest all passed: the first
    // refusal is the broker socket nobody listens on.
    assert!(stderr.contains("capability redemption denied"), "{stderr}");
}

#[test]
fn an_already_expired_ticket_is_refused_and_nothing_is_written() {
    let case = Case::start("expired");
    let (answer, manifest, signature, _) = sign(&case, "0");
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("bootstrap manifest is invalid or expired"), "{stderr}");
    assert!(!manifest.exists() && !signature.exists(), "no ticket is written");
}

#[test]
fn a_key_is_never_written_over_an_existing_file() {
    let case = Case::start("existing-key");
    let (private, public) = key(&case, "supervisor");
    let before = fs::read(&private).unwrap();
    let answer = run(
        SINGULARITY,
        &["ticket", "key", "--holder", "supervisor", "--out", text(&private), "--public-out", text(&public)],
    );
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("could not be created as a new owner-only file"), "{stderr}");
    assert_eq!(fs::read(&private).unwrap(), before, "the existing key is unchanged");
}

#[test]
fn a_workload_public_key_is_the_pem_skarbiec_verifies_proofs_with() {
    let case = Case::start("workload-pem");
    let (_, public) = key(&case, "workload");
    let written = fs::read_to_string(&public).unwrap();
    assert!(written.contains("-----BEGIN PUBLIC KEY-----"), "{written}");
    // Skarbiec hands the registered key to `openssl pkeyutl -verify -pubin`;
    // the same reader must accept it here.
    let parsed = run("openssl", &["pkey", "-pubin", "-in", text(&public), "-noout"]);
    assert!(parsed.status.success(), "{}", String::from_utf8_lossy(&parsed.stderr));
}

#[test]
fn a_launch_without_skarbiec_stops_before_any_capability_or_ticket() {
    let case = Case::start("launch-no-skarbiec");
    let (supervisor, trust_root) = key(&case, "supervisor");
    let policy = case.path("policy.json");
    fs::write(&policy, "{}").unwrap();
    let runtime = case.path("runtime");
    let absent = case.path("absent-skarbiec");
    let answer = run(
        SINGULARITY,
        &[
            "ticket", "launch",
            "--agent-id", "ticket-test", "--role", "test", "--environment", "test",
            "--host", "test-host", "--workload-id", "ticket-test-workload",
            "--skarbiec", text(&absent), "--grant-capabilities", "acquire:brama#hmac",
            "--grant-ttl-seconds", "60", "--capability-ttl-seconds", "60", "--capability-max-uses", "1",
            "--broker-socket", text(&case.path("absent-broker.sock")),
            "--executable", SINGULARITY,
            "--policy-file", text(&policy), "--policy-sequence", "1", "--expires-in-seconds", "300",
            "--brama-resource", "brama:hmac", "--brama-bearer-resource", "brama:bearer",
            "--most-resource", "most:token",
            "--supervisor-key", text(&supervisor), "--trust-root", text(&trust_root),
            "--runtime-root", text(&runtime),
            "--", "doctor",
        ],
    );
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("could not be started"), "{stderr}");
    let tickets: Vec<_> = fs::read_dir(&runtime).unwrap().map(|entry| entry.unwrap().path()).collect();
    for ticket in tickets {
        assert!(!ticket.join("manifest.json").exists(), "no ticket is signed without capabilities");
    }
}
