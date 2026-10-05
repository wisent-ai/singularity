use std::process::Command;

// Exercise the real CLI with no inherited accounting or credentials. A missing
// secret path makes the pre-fix path refuse locally rather than contacting a
// provider; the required-option refusal must happen before that path is read.
#[test]
fn an_undeclared_budget_price_or_cadence_is_refused_before_runtime_startup() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let missing_secret = root.join(".build/missing-launch-declaration-secret");
    assert!(
        !missing_secret.exists(),
        "the refusal test's missing path must not exist"
    );
    let digest = "a".repeat(64);
    let required = [
        ("--agent-id", "declaration-test"),
        ("--agent-name", "Declaration test"),
        ("--agent-ticker", "DECL"),
        ("--role", "test"),
        ("--environment", "test"),
        ("--host", "test-host"),
        ("--workload-id", "test-workload"),
        ("--workload-public-key", digest.as_str()),
        ("--executable-digest", digest.as_str()),
        ("--code-digest", digest.as_str()),
        ("--policy-digest", digest.as_str()),
        ("--policy-sequence", "1"),
        ("--brama-url", "https://brama.wisent.com"),
        ("--brama-model", "any"),
        ("--las-command", "las"),
        ("--las-only", ""),
    ];
    let declarations = [
        ("--starting-balance", "7.125"),
        ("--instance-price", "0.375"),
        ("--cycle-interval-secs", "23"),
    ];
    for (omitted, _) in declarations {
        let mut command = Command::new(env!("CARGO_BIN_EXE_singularity"));
        command.env_clear().arg("once");
        for (flag, value) in required {
            command.args([flag, value]);
        }
        for flag in [
            "--brama-secret-file",
            "--las-entrypoint",
            "--las-release-manifest",
            "--las-release-manifest-signature",
            "--las-release-trust-store",
            "--las-release-watermark",
        ] {
            command.arg(flag).arg(&missing_secret);
        }
        command
            .arg("--workspace")
            .arg(root)
            .arg("--state-dir")
            .arg(&missing_secret);
        for (flag, value) in declarations {
            if flag != omitted {
                command.args([flag, value]);
            }
        }
        let output = command.output().expect("run the actual singularity CLI");
        assert_eq!(output.status.code(), Some(2), "{omitted}: {output:?}");
        let error = String::from_utf8(output.stderr).expect("UTF-8 CLI diagnostic");
        assert!(
            error.contains(omitted),
            "missing declaration {omitted} was not named: {error}"
        );
        assert!(!missing_secret.exists(), "a refused launch created state");
    }
}
