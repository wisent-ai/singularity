#!/usr/bin/env python3
"""Drive one transaction through singularity-finance-mcp end to end:
propose, refuse, simulate, approve, timelock, sign, dispatch to a failing
executor, reconcile, submit, confirm — then flip and roll back the enable
lease. This is the executed transcript behind walkthrough-finance.md.

No network, no custody system, no real money: all seven Ed25519 authority
keys are generated here, the executor is /usr/bin/false, and everything
lives in a mktemp directory printed at the end.

Requires: python3 with the `cryptography` package, and the
singularity-finance-mcp binary (SINGULARITY_FINANCE_MCP env, default:
target/release/singularity-finance-mcp relative to the repo root).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

MCP = os.environ.get(
    "SINGULARITY_FINANCE_MCP",
    os.path.join(os.path.dirname(__file__), "../../target/release/singularity-finance-mcp"),
)
MCP = os.path.abspath(MCP)

# --- signing: the exact canonical-JSON contract of src/finance_surface/policy.rs
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
    with open(path, "w", opener=lambda p, f: os.open(p, f, 0o600)) as fh:
        fh.write(content)
    os.chmod(path, 0o600)

def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

# --- seven distinct authorities, as the policy validator requires
doc_key, doc_pub = keypair()          # signs policy, lease, owner events
approver_key, approver_pub = keypair()
sim_key, sim_pub = keypair()
signer_key, signer_pub = keypair()
exec_key, exec_pub = keypair()
recon_key, recon_pub = keypair()
receipt_key, receipt_pub = keypair()  # signs external WORM receipts

FIN = tempfile.mkdtemp(prefix="singularity-finance.")
os.makedirs(f"{FIN}/worm", mode=0o700)
os.chmod(f"{FIN}/worm", 0o700)  # the service verifies the sink is owner-only
now = datetime.now(timezone.utc)
POLICY_ID, VERSION = "walkthrough-policy", 1

policy = {
    "policy_id": POLICY_ID,
    "version": VERSION,
    "valid_from": ts(now - timedelta(hours=1)),
    "expires_at": ts(now + timedelta(days=1)),
    "beneficiaries": {
        "infra-vendor": {
            "destination": "vendor-account-0001",
            "allowed_assets": ["USD"],
            "allowed_purposes": ["invoice"],
            "valid_from": ts(now - timedelta(hours=1)),
            "expires_at": ts(now + timedelta(days=1)),
            "per_transaction_minor": 5000,
            "rolling_window_seconds": 3600,
            "rolling_limit_minor": 8000,
            "daily_limit_minor": 8000,
            "lifetime_limit_minor": 20000,
            "enabled": True,
        }
    },
    "assets": {
        "USD": {
            "per_transaction_minor": 5000,
            "rolling_window_seconds": 3600,
            "rolling_limit_minor": 8000,
            "daily_limit_minor": 8000,
            "lifetime_limit_minor": 20000,
            "spendable_balance_minor": 10000,
            "protected_reserve_minor": 4000,
        }
    },
    "approval": {
        "required_approvals": 1,
        "approver_keys": {"treasury-owner": approver_pub},
        "timelock_seconds": 2,
        "proposal_ttl_max_seconds": 3600,
    },
    "custody_authorities": {
        "simulators": {"sim-1": sim_pub},
        "signers": {"signer-1": signer_pub},
        "executors": {"exec-1": exec_pub},
        "reconcilers": {"rec-1": recon_pub},
    },
    "worm_sink_dir": f"{FIN}/worm",
    "worm_sink_id": "walkthrough-worm",
    "worm_receipt_key_hex": receipt_pub,
}
write_owner_only(f"{FIN}/policy.json", signed_envelope(policy, doc_key))

def write_lease(lease_id, issued_at, enabled=True, kill_switch=False):
    lease = {
        "policy_id": POLICY_ID,
        "policy_version": VERSION,
        "lease_id": lease_id,
        "issued_at": ts(issued_at),
        "expires_at": ts(issued_at + timedelta(hours=1)),
        "enabled": enabled,
        "kill_switch": kill_switch,
    }
    write_owner_only(f"{FIN}/lease.json", signed_envelope(lease, doc_key))

lease1_issued = now - timedelta(seconds=30)
write_lease("lease-1", lease1_issued)

env = dict(
    os.environ,
    SINGULARITY_FINANCE_POLICY_FILE=f"{FIN}/policy.json",
    SINGULARITY_FINANCE_ENABLE_LEASE_FILE=f"{FIN}/lease.json",
    SINGULARITY_FINANCE_STATE_DIR=f"{FIN}/state",
    SINGULARITY_FINANCE_VERIFY_KEY_HEX=doc_pub,
    SINGULARITY_FINANCE_EXECUTOR="/usr/bin/false",
)

server = subprocess.Popen(
    [MCP], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True
)
rpc_id = 0

def rpc(method, params):
    global rpc_id
    rpc_id += 1
    server.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}) + "\n")
    server.stdin.flush()
    return json.loads(server.stdout.readline())["result"]

def call(label, name, arguments):
    print(f"### {label}")
    print(f"-> {json.dumps(arguments)}")
    result = rpc("tools/call", {"name": name, "arguments": arguments})
    if result.get("isError"):
        print(f"<- ERROR {result['content'][0]['text']}\n")
        return None
    value = result["structuredContent"]
    keys = ["transaction_id", "status", "intent_hash", "approval_count",
            "timelock_until", "reconciliation_required"]
    print(f"<- {json.dumps({k: value[k] for k in keys if k in value})}\n")
    return value

event_counter = 0

def owner_event(label, transaction_id, intent_hash, action, occurred_at=None):
    global event_counter
    event_counter += 1
    occurred_at = occurred_at or ts(datetime.now(timezone.utc))
    event = {
        "event_id": f"evt-{event_counter:03}",
        "transaction_id": transaction_id,
        "intent_hash": intent_hash,
        "occurred_at": occurred_at,
        "action": action,
    }
    path = f"{FIN}/evt-{event_counter:03}.json"
    write_owner_only(path, signed_envelope(event, doc_key))
    print(f"$ singularity-finance-mcp owner-event {os.path.basename(path)}   # {label}")
    run = subprocess.run([MCP, "owner-event", path], env=env, capture_output=True, text=True)
    if run.returncode != 0:
        print(f"<- ERROR {run.stderr.strip()}\n")
        return None
    value = json.loads(run.stdout)
    keys = ["status", "approval_count", "timelock_until", "reconciliation_required"]
    print(f"<- {json.dumps({k: value[k] for k in keys if k in value})}\n")
    return value

def role_signature(key, role, transaction_id, intent_hash, reference):
    message = f"singularity-finance-{role}-v1:{POLICY_ID}:{VERSION}:{transaction_id}:{intent_hash}:{reference}"
    return key.sign(message.encode()).hex()

def worm_receipt(event_kind, transaction_id, intent_hash, reference_hash, occurred_at):
    receipt = {
        "sink_id": "walkthrough-worm",
        "receipt_id": f"receipt-{event_kind}",
        "event_kind": event_kind,
        "transaction_id": transaction_id,
        "intent_hash": intent_hash,
        "reference_hash": reference_hash,
        "recorded_at": occurred_at,
    }
    path = f"{FIN}/receipt-{event_kind}.json"
    write_owner_only(path, signed_envelope(receipt, receipt_key))
    return path

# --- handshake
info = rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "walkthrough", "version": "0"}})
tools = rpc("tools/list", {})
print(f"server: {info['serverInfo']['name']} {info['serverInfo']['version']}")
print("tools:", ", ".join(tool["name"] for tool in tools["tools"]), "\n")

# --- propose, replay, refuse
intent = {"request_id": "req-invoice-001", "beneficiary_id": "infra-vendor",
          "asset": "USD", "amount_minor": 2500, "purpose": "invoice", "ttl_seconds": 900}
tx = call("finance_propose (accepted)", "finance_propose", intent)
TX, HASH = tx["transaction_id"], tx["intent_hash"]

call("finance_propose replay (same request_id, same intent)", "finance_propose", intent)
call("finance_propose refusal (request_id reused with different intent)",
     "finance_propose", dict(intent, amount_minor=2600))
call("finance_propose refusal (unknown beneficiary)",
     "finance_propose", dict(intent, request_id="req-bad-001", beneficiary_id="unknown-vendor"))
call("finance_propose refusal (per-transaction limit)",
     "finance_propose", dict(intent, request_id="req-bad-002", amount_minor=5001))
call("finance_propose refusal (protected reserve)",
     "finance_propose", dict(intent, request_id="req-bad-003", amount_minor=4000))
call("finance_propose refusal (TTL beyond policy)",
     "finance_propose", dict(intent, request_id="req-bad-004", ttl_seconds=4000))
call("finance_propose refusal (protected intent fields in parameters)",
     "finance_propose", dict(intent, request_id="req-bad-005", parameters={"destination": "elsewhere"}))
call("finance_execute refusal (not signed yet)", "finance_execute", {"transaction_id": TX})

# --- owner events carry it forward
evidence = sha("simulation-evidence")
owner_event("simulation_accepted by sim-1", TX, HASH, {
    "type": "simulation_accepted", "evidence_hash": evidence, "simulator_id": "sim-1",
    "simulator_signature_hex": role_signature(sim_key, "simulation", TX, HASH, evidence)})

approval_message = f"singularity-finance-approval-v1:{POLICY_ID}:{VERSION}:{TX}:{HASH}:{evidence}"
owner_event("approval_granted by treasury-owner", TX, HASH, {
    "type": "approval_granted", "approver_id": "treasury-owner",
    "approval_signature_hex": approver_key.sign(approval_message.encode()).hex()})

time.sleep(2.5)  # the policy's 2-second timelock
call("finance_status after timelock (ready)", "finance_status", {"transaction_id": TX})

attestation = sha("signer-attestation")
owner_event("signed by signer-1", TX, HASH, {
    "type": "signed", "signer_attestation_hash": attestation, "signer_id": "signer-1",
    "signer_signature_hex": role_signature(signer_key, "signing", TX, HASH, attestation)})

call("finance_cancel refusal (already signed)", "finance_cancel",
     {"transaction_id": TX, "request_id": "req-cancel-001"})

# --- dispatch marks indeterminate before the executor breathes
call("finance_execute (executor /usr/bin/false refuses after dispatch)",
     "finance_execute", {"transaction_id": TX})
call("finance_status after failed dispatch", "finance_status", {"transaction_id": TX})

reconciliation = sha("nothing-was-submitted")
owner_event("reconciled_not_submitted by rec-1", TX, HASH, {
    "type": "reconciled_not_submitted", "reconciliation_hash": reconciliation,
    "reconciler_id": "rec-1",
    "reconciler_signature_hex": role_signature(recon_key, "not_submitted", TX, HASH, reconciliation)})

# --- submission and confirmation, receipt-bound
reference = sha("custody-reference")
at = ts(datetime.now(timezone.utc))
owner_event("submitted by exec-1 + WORM receipt", TX, HASH, {
    "type": "submitted", "executor_reference_hash": reference, "executor_id": "exec-1",
    "executor_signature_hex": role_signature(exec_key, "submission", TX, HASH, reference),
    "worm_receipt_file": worm_receipt("submitted", TX, HASH, reference, at)}, occurred_at=at)

confirmation = sha("chain-confirmation")
at = ts(datetime.now(timezone.utc))
owner_event("confirmed by rec-1 + WORM receipt", TX, HASH, {
    "type": "confirmed", "reconciliation_hash": confirmation, "reconciler_id": "rec-1",
    "reconciler_signature_hex": role_signature(recon_key, "confirmation", TX, HASH, confirmation),
    "worm_receipt_file": worm_receipt("confirmed", TX, HASH, confirmation, at)}, occurred_at=at)

# --- the lease is the hand on the switch
write_lease("lease-2", datetime.now(timezone.utc), kill_switch=True)
call("finance_propose refusal (kill switch lease)", "finance_propose",
     dict(intent, request_id="req-invoice-002", amount_minor=2400))

time.sleep(1)  # lease-3 must be issued strictly after lease-1
write_lease("lease-3", datetime.now(timezone.utc))
call("finance_propose accepted again under a fresh lease (anchor advances)",
     "finance_propose", dict(intent, request_id="req-invoice-002", amount_minor=2400))

write_lease("lease-1", lease1_issued)
call("finance_propose refusal (older lease restored: rollback detected)",
     "finance_propose", dict(intent, request_id="req-invoice-003", amount_minor=2300))

server.stdin.close()
server.wait(timeout=10)
print(f"inspect the state store (walkthrough-state.md): FIN={FIN}")
