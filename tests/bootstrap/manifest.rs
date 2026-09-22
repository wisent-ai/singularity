// These boundary regressions do not qualify the complete managed-launch flow.
use super::*;
fn manifest_with_expiry(expires_at: DateTime<Utc>) -> BootstrapManifest {
    BootstrapManifest {
        version: "singularity.bootstrap.v2".to_owned(),
        issued_at: expires_at - Duration::seconds(60),
        expires_at,
        agent_id: "agent-42".to_owned(),
        role: "publisher".to_owned(),
        environment: "test".to_owned(),
        host: "test-host".to_owned(),
        workload_id: "workload-42".to_owned(),
        workload_public_key: "1".repeat(64),
        executable_digest: "2".repeat(64),
        code_digest: "3".repeat(64),
        policy_digest: "4".repeat(64),
        policy_sequence: 1,
        broker_socket: PathBuf::from("/tmp/bootstrap-test-broker.sock"),
        workload_private_key_file: PathBuf::from("/tmp/bootstrap-test-workload.key"),
        singularity_executable: PathBuf::from("/tmp/bootstrap-test-executable"),
        singularity_args: Vec::new(),
        capabilities: BootstrapCapabilities {
            brama: BootstrapCapability {
                id: "5".repeat(64),
                target: "singularity-bootstrap".to_owned(),
                purpose: "singularity.brama.bootstrap".to_owned(),
                resource: "brama:agent-42".to_owned(),
            },
            brama_bearer: BootstrapCapability {
                id: "7".repeat(64),
                target: "singularity-bootstrap".to_owned(),
                purpose: "singularity.brama.authorization".to_owned(),
                resource: "brama:agent-42".to_owned(),
            },
            most: BootstrapCapability {
                id: "6".repeat(64),
                target: "singularity-bootstrap".to_owned(),
                purpose: "singularity.most.bootstrap".to_owned(),
                resource: "most:publisher-42".to_owned(),
            },
        },
    }
}

#[test]
fn verify_manifest_requires_an_exact_domain_separated_signature() {
    let temp = TempDirectory::new();
    let trust_root_path = temp.path().join("manifest-trust-root.hex");
    let signature_path = temp.path().join("manifest-signature.hex");
    let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
    write_owner_only(
        &trust_root_path,
        hex::encode(signing_key.verifying_key().as_bytes()).as_bytes(),
    );

    let manifest_bytes = br#"{"version":"singularity.bootstrap.v2","agent_id":"agent-42"}"#;
    let mut domain_separated = Vec::with_capacity(MANIFEST_DOMAIN.len() + manifest_bytes.len());
    domain_separated.extend_from_slice(MANIFEST_DOMAIN);
    domain_separated.extend_from_slice(manifest_bytes);
    let valid_signature = signing_key.sign(&domain_separated);
    write_owner_only(
        &signature_path,
        hex::encode(valid_signature.to_bytes()).as_bytes(),
    );

    assert!(
        verify_manifest(manifest_bytes, &signature_path, &trust_root_path).is_ok(),
        "the exact domain-separated manifest bytes must verify"
    );

    let tampered_bytes = br#"{"version":"singularity.bootstrap.v2","agent_id":"agent-43"}"#;
    assert!(
        verify_manifest(tampered_bytes, &signature_path, &trust_root_path).is_err(),
        "changing the signed manifest bytes must invalidate the signature"
    );

    let unscoped_signature = signing_key.sign(manifest_bytes);
    write_owner_only(
        &signature_path,
        hex::encode(unscoped_signature.to_bytes()).as_bytes(),
    );
    assert!(
        verify_manifest(manifest_bytes, &signature_path, &trust_root_path).is_err(),
        "a signature over the payload without MANIFEST_DOMAIN must be rejected"
    );
}

#[test]
fn validate_manifest_rejects_expiry_and_excessive_lifetime() {
    let now = Utc::now();
    let expired = manifest_with_expiry(now - Duration::seconds(1));
    let mut excessive_lifetime = manifest_with_expiry(now + Duration::seconds(250));
    excessive_lifetime.issued_at =
        excessive_lifetime.expires_at - Duration::seconds(MAX_MANIFEST_LIFETIME + 1);
    let cases = [
        ("expired", expired),
        ("lifetime over 300 seconds", excessive_lifetime),
    ];

    for (name, manifest) in cases {
        assert!(
            validate_manifest(&manifest).is_err(),
            "bootstrap must reject a manifest that is {name}"
        );
    }
}

#[test]
fn verify_executable_requires_the_exact_sha256_of_a_regular_file() {
    let temp = TempDirectory::new();
    let executable_path = temp.path().join("singularity-test-executable");
    let executable_bytes = b"deterministic executable fixture\n";
    write_owner_only(&executable_path, executable_bytes);
    let expected = hex::encode(Sha256::digest(executable_bytes));

    assert!(
        verify_executable(&executable_path, &expected).is_ok(),
        "a regular file with the exact expected SHA-256 must verify"
    );

    let wrong_digest = hex::encode(Sha256::digest(b"different executable bytes"));
    assert!(
        verify_executable(&executable_path, &wrong_digest).is_err(),
        "an executable whose bytes do not match the manifest digest must be rejected"
    );
}
