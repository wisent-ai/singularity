use super::*;
use ed25519_dalek::{Signer, SigningKey};

#[test]
fn a_key_is_never_written_over_an_existing_file() {
    let case = Case::start("existing-key");
    let (private, public) = key(&case, "supervisor");
    let private_before = fs::read(&private).unwrap();
    let public_before = fs::read(&public).unwrap();
    let answer = run(
        SINGULARITY,
        &[
            "ticket",
            "key",
            "--holder",
            "supervisor",
            "--out",
            text(&private),
            "--public-out",
            text(&public),
        ],
    );
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(
        stderr.contains("could not be created as a new owner-only file"),
        "{stderr}"
    );
    assert_eq!(
        fs::read(&private).unwrap(),
        private_before,
        "the existing private key changed"
    );
    assert_eq!(
        fs::read(&public).unwrap(),
        public_before,
        "the existing public key changed"
    );
}

#[test]
fn a_refused_public_destination_removes_only_the_new_private_key() {
    let case = Case::start("public-refusal");
    let private = case.path("workload.key");
    let public = case.path("occupied.pub");
    let retained = b"existing public destination";
    fs::write(&public, retained).unwrap();
    let answer = run(
        SINGULARITY,
        &[
            "ticket",
            "key",
            "--holder",
            "workload",
            "--out",
            text(&private),
            "--public-out",
            text(&public),
        ],
    );
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains(text(&public)), "{stderr}");
    assert!(
        !private.exists(),
        "refused public write left its new private key"
    );
    assert_eq!(
        fs::read(&public).unwrap(),
        retained,
        "existing destination changed"
    );
    fs::remove_file(&public).unwrap();
    let recovered = run(
        SINGULARITY,
        &[
            "ticket",
            "key",
            "--holder",
            "workload",
            "--out",
            text(&private),
            "--public-out",
            text(&public),
        ],
    );
    assert!(
        recovered.status.success(),
        "{}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert!(owner_only(&private) && owner_only(&public));
    verify_key_pair(&case, &private, &public);
}

fn verify_key_pair(case: &Case, private: &Path, public: &Path) {
    let private_bytes = hex::decode(fs::read_to_string(private).unwrap()).unwrap();
    let signing = SigningKey::from_bytes(&private_bytes.try_into().unwrap());
    let payload = case.path("proof-input");
    let signature = case.path("proof-signature");
    let message = b"workload key correspondence";
    fs::write(&payload, message).unwrap();
    fs::write(&signature, signing.sign(message).to_bytes()).unwrap();
    let args = [
        "pkeyutl",
        "-verify",
        "-pubin",
        "-inkey",
        text(public),
        "-rawin",
        "-in",
        text(&payload),
        "-sigfile",
        text(&signature),
    ];
    let verified = run("openssl", &args);
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    fs::write(&payload, b"changed workload proof").unwrap();
    let refused = run("openssl", &args);
    assert!(!refused.status.success(), "changed proof was accepted");
}

#[test]
fn a_workload_public_key_verifies_proofs_from_its_private_half() {
    let case = Case::start("workload-pem");
    let (private, public) = key(&case, "workload");
    verify_key_pair(&case, &private, &public);
}
