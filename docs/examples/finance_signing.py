"""The signing contract the finance walkthrough shares with src/finance_surface/policy.rs."""
import hashlib
import json
import os
from datetime import datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

OWNER_ONLY_MODE = 0o600
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def keypair():
    key = Ed25519PrivateKey.generate()
    return key, key.public_key().public_bytes_raw().hex()


def signed_envelope(document, key) -> str:
    return json.dumps(
        {"document": document, "signature_hex": key.sign(canonical(document)).hex()}
    )


def write_owner_only(path, content):
    with open(path, "w", opener=lambda p, f: os.open(p, f, OWNER_ONLY_MODE)) as fh:
        fh.write(content)
    os.chmod(path, OWNER_ONLY_MODE)


def ts(dt: datetime) -> str:
    return dt.strftime(TIMESTAMP_FORMAT)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
