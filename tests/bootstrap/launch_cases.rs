use super::*;

fn launch(case: &Case, being_args: Option<&str>) -> (Output, PathBuf) {
    let (supervisor, trust_root) = key(case, "supervisor");
    let policy = case.path("policy.json");
    fs::write(&policy, "{}").unwrap();
    let runtime = case.path("runtime");
    let absent = case.path("absent-skarbiec");
    let socket = case.path("absent-broker.sock");
    let mut arguments = vec![
        "ticket",
        "launch",
        "--agent-id",
        "ticket-test",
        "--role",
        "test",
        "--environment",
        "test",
        "--host",
        "test-host",
        "--workload-id",
        "ticket-test-workload",
        "--skarbiec",
        text(&absent),
        "--grant-capabilities",
        "acquire:brama#hmac",
        "--grant-ttl-seconds",
        "60",
        "--capability-ttl-seconds",
        "60",
        "--capability-max-uses",
        "1",
        "--broker-socket",
        text(&socket),
        "--executable",
        SINGULARITY,
        "--policy-file",
        text(&policy),
        "--policy-sequence",
        "1",
        "--expires-in-seconds",
        "300",
        "--brama-resource",
        "brama:hmac",
        "--brama-bearer-resource",
        "brama:bearer",
        "--most-resource",
        "most:token",
        "--supervisor-key",
        text(&supervisor),
        "--trust-root",
        text(&trust_root),
        "--runtime-root",
        text(&runtime),
    ];
    match being_args {
        Some(json) => arguments.extend(["--being-args", json]),
        None => arguments.extend(["--", "doctor"]),
    }
    (run(SINGULARITY, &arguments), runtime)
}

#[test]
fn a_launch_without_skarbiec_stops_before_any_capability_or_ticket() {
    let case = Case::start("launch-no-skarbiec");
    let (answer, runtime) = launch(&case, None);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("could not be started"), "{stderr}");
    for entry in fs::read_dir(&runtime).unwrap() {
        assert!(
            !entry.unwrap().path().join("manifest.json").exists(),
            "no ticket is signed without capabilities"
        );
    }
}

#[test]
fn empty_being_arguments_refuse_before_creating_runtime_or_requesting_grants() {
    let case = Case::start("empty-being-args");
    let (answer, runtime) = launch(&case, Some("[]"));
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("SINGULARITY_LAUNCH_BEING_ARGS"), "{stderr}");
    assert!(stderr.contains("empty array"), "{stderr}");
    assert!(
        !runtime.try_exists().unwrap(),
        "invalid arguments created runtime state"
    );
}

#[test]
fn malformed_being_arguments_refuse_before_creating_runtime_or_requesting_grants() {
    let case = Case::start("malformed-being-args");
    let (answer, runtime) = launch(&case, Some("{}"));
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("SINGULARITY_LAUNCH_BEING_ARGS"), "{stderr}");
    assert!(stderr.contains("JSON array of strings"), "{stderr}");
    assert!(
        !runtime.try_exists().unwrap(),
        "invalid arguments created runtime state"
    );
}

#[test]
fn declared_being_command_reaches_the_actual_grant_dependency() {
    let case = Case::start("declared-being-command");
    let (answer, runtime) = launch(&case, Some("[\"doctor\"]"));
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(
        stderr.contains(text(&case.path("absent-skarbiec"))),
        "{stderr}"
    );
    assert!(stderr.contains("could not be started"), "{stderr}");
    assert!(
        runtime.is_dir(),
        "valid arguments did not enter launch preparation"
    );
}
