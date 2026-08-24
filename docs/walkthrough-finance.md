# Walkthrough: one transaction through the finance boundary

Can the whole money path — propose, simulate, approve, timelock, sign,
dispatch, reconcile, submit, confirm — be exercised without a custody
system, a credential, or a cent? Yes: every authority in the signed policy
is a key **you** generate in a sandbox, the executor is `/usr/bin/false`,
and the WORM receipts are signed by your own throwaway receipt key. The
service cannot tell the difference, which is the point: everything below is
`singularity-finance-mcp` 0.5.0 verbatim, and the same transcript is
reproducible with [examples/finance-lifecycle.py](examples/finance-lifecycle.py).

## Setup

Seven Ed25519 keypairs are generated locally: the document key (policy,
lease, owner events), one approver, one simulator, one signer, one
executor, one reconciler, one WORM receipt key — the policy validator
requires all of them distinct. The policy allows one beneficiary
(`infra-vendor`, purpose `invoice`, asset `USD`), per-transaction limit
5000 minor units, spendable balance 10000 with a protected reserve of 4000,
one required approval, and a 2-second timelock. Policy and lease are
written as `{"document":…,"signature_hex":…}` envelopes over canonical JSON
(sorted keys, no whitespace), mode `0600`, then:

```bash
export SINGULARITY_FINANCE_POLICY_FILE=$FIN/policy.json \
  SINGULARITY_FINANCE_ENABLE_LEASE_FILE=$FIN/lease.json \
  SINGULARITY_FINANCE_STATE_DIR=$FIN/state \
  SINGULARITY_FINANCE_VERIFY_KEY_HEX=<document-public-key-hex> \
  SINGULARITY_FINANCE_EXECUTOR=/usr/bin/false
singularity-finance-mcp   # stdio JSON-RPC
```

## Propose, replay, refuse

```
### finance_propose (accepted)
-> {"request_id": "req-invoice-001", "beneficiary_id": "infra-vendor", "asset": "USD",
    "amount_minor": 2500, "purpose": "invoice", "ttl_seconds": 900}
<- {"transaction_id": "fin_fac7a76ce60a9aa6f56acb703bdfd834", "status": "approval_pending",
    "intent_hash": "fac7a76c…ada98a2", "approval_count": 0, "reconciliation_required": false, …}
```

The transaction id is the first 32 hex of the canonical intent hash —
`fin_<hash>` — so the same intent cannot be double-opened. Replaying the
same `request_id` with the same input returns the recorded response
byte-for-byte; changing the input is a conflict:

```
### finance_propose replay (same request_id, same intent)
<- {"transaction_id": "fin_fac7a76ce60a9aa6f56acb703bdfd834", "status": "approval_pending", …}

### finance_propose refusal (request_id reused with different intent)
<- ERROR invalid_state: request_id was already used with different intent
```

The policy refusals, each one sentence:

```
<- ERROR policy_denied: beneficiary is not in signed policy
<- ERROR policy_denied: per-transaction limit exceeded            # amount_minor 5001
<- ERROR policy_denied: protected reserve would be breached       # 4000 asked, 2500 already reserved, 10000−4000 spendable
<- ERROR policy_denied: proposal TTL exceeds signed policy        # ttl_seconds 4000 > 3600
<- ERROR policy_denied: parameters cannot override protected intent fields   # {"destination": …}
```

And the execute gate holds while the transaction is merely proposed:

```
### finance_execute refusal (not signed yet)
<- ERROR invalid_state: execution requires signed state and completed reconciliation
```

## Owner events carry it forward

The model can only propose, read, and cancel. Everything else enters as a
signed file through `singularity-finance-mcp owner-event <absolute-file>`,
one process invocation per event, each verified against the corresponding
authority key in the policy:

```
$ singularity-finance-mcp owner-event evt-sim-001.json      # simulation_accepted by sim-1
<- {"status":"approval_pending", …}                         # simulated, back to awaiting approval

$ singularity-finance-mcp owner-event evt-approve-001.json  # approval_granted by treasury-owner
<- {"status":"timelocked", "approval_count":1, "timelock_until":"2026-08-24T22:07:55.745164Z", …}
```

The approval signature covers the exact intent hash **and** the accepted
simulation evidence hash — an approver cannot approve a transaction whose
simulation they have not seen. After the 2-second timelock:

```
### finance_status after timelock (ready)
<- {"status": "ready", …}

$ singularity-finance-mcp owner-event evt-signed-001.json   # signed by signer-1
<- {"status":"signed", …}

### finance_cancel refusal (already signed)
<- ERROR invalid_state: transaction can no longer be cancelled
```

## Dispatch marks indeterminate before the executor breathes

`finance_execute` durably commits the `indeterminate` state and sets
`reconciliation_required` **before** starting the executor, so a crash or
timeout can never cause a silent duplicate effect. With `/usr/bin/false` as
the executor:

```
### finance_execute (executor /usr/bin/false refuses after dispatch)
<- ERROR internal_error: executor refused:

### finance_status after failed dispatch
<- {"status": "indeterminate", "reconciliation_required": true, …}
```

The transaction is now frozen until a reconciler answers what actually
happened. Here, nothing was submitted:

```
$ singularity-finance-mcp owner-event evt-recon-001.json    # reconciled_not_submitted by rec-1
<- {"status":"signed", "reconciliation_required":false, …}
```

## Submission and confirmation, receipt-bound

A real submission arrives as an owner event whose WORM receipt — signed by
the policy's receipt key — must bind the exact sink id, event kind,
transaction id, intent hash, reference hash, and timestamp:

```
$ singularity-finance-mcp owner-event evt-submit-001.json   # submitted by exec-1 + receipt
<- {"status":"submitted", …}

$ singularity-finance-mcp owner-event evt-confirm-001.json  # confirmed by rec-1 + receipt
<- {"status":"confirmed", …}
```

`confirmed` is terminal; the amount keeps counting against rolling, daily,
lifetime, and reserve arithmetic. What all of this deposited on disk —
transactions, the hash-chained audit, commit journal, WORM copies, anchors
— is inspected in [walkthrough-state](walkthrough-state.md).

## The lease is the hand on the switch

Flip the [enable lease](concepts/lease.md) to `kill_switch: true`, sign a
fresh one, then try to sneak the old one back:

```
### finance_propose refusal (kill switch lease)
<- ERROR policy_denied: finance enable lease is absent, expired, disabled, or mismatched

### finance_propose accepted again under a fresh lease (anchor advances)
<- {"transaction_id": "fin_9564d5dceb3b6c6b3d219e59447418e2", "status": "approval_pending", …}

### finance_propose refusal (older lease restored: rollback detected)
<- ERROR policy_denied: signed enable lease rollback or equivocation detected
```

The state store remembers the newest lease it has ever accepted; presenting
an older one is not a downgrade, it is an incident. The full boundary
design — three processes, who holds which key, why the model process never
sees any of them — is [the finance boundary](finance.md).
