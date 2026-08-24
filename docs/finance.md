# The finance boundary

Money is the one place where "the model decided" is never sufficient. The
financial surface is split across three processes so that no single
compromise moves funds: the model process proposes; `singularity-finance-mcp`
enforces a signed policy and a durable state machine; an isolated executor —
the release ships `singularity-finance-executor-http` — owns the credentials
and performs the real operation. The model process never receives a signing
key, and [the loop](loop.md) credits revenue only from this surface's
reported results.

## The four tools

`singularity-finance-mcp` is a stdio MCP server exposing exactly:

| Tool | Contract |
|---|---|
| `finance_propose` | Propose a policy-bound transfer, trade, swap or liquidity action: `request_id`, `beneficiary_id`, `asset`, `amount_minor` (positive integer), `purpose`, optional `parameters`, `ttl_seconds`. |
| `finance_status` | Read a transaction's status without financial effect. |
| `finance_cancel` | Cancel an unsigned transaction before its deadline. |
| `finance_execute` | Execute a signed, approved and reconciled transaction through the configured isolated executor. |

`request_id` makes `finance_propose` and `finance_cancel` idempotent: the
same id with the same input returns the recorded response; the same id with
different intent is a conflict.

## The signed policy

The service refuses to start without `SINGULARITY_FINANCE_POLICY_FILE`, an
Ed25519-signed document verified against
`SINGULARITY_FINANCE_VERIFY_KEY_HEX` over canonical JSON (sorted keys, no
insignificant whitespace). The policy declares:

- **Beneficiaries** — each with a destination, allowed assets and purposes,
  a validity window, per-transaction, rolling, daily and lifetime limits,
  and an enabled flag. Proposals name a `beneficiary_id`; there is no
  free-form destination.
- **Assets** — per-asset transaction, rolling, daily and lifetime limits,
  plus `spendable_balance_minor` and `protected_reserve_minor`. A proposal
  that would dip into the protected reserve — counting every non-terminal
  transaction as still reserving its funds — is refused.
- **Approval** — `required_approvals`, the approver public keys (all
  distinct), a mandatory `timelock_seconds`, and a proposal TTL ceiling of
  at most 30 days.
- **Custody authorities** — separate key sets for simulators, signers,
  executors and reconcilers; every key in the policy, including the WORM
  receipt key, must be distinct.
- **WORM sink** — the directory and sink id for write-once receipts.

## The enable lease

Proposing and executing additionally require
`SINGULARITY_FINANCE_ENABLE_LEASE_FILE`, a second signed document that must
match the policy id and version, be inside its validity window, be
`enabled`, and not have `kill_switch` set. Deleting or expiring the lease
halts new proposals and all execution without touching the policy;
`finance_status` and `finance_cancel` keep working so in-flight
transactions can still be read and withdrawn.

## Transaction lifecycle

State lives in the owner-only `SINGULARITY_FINANCE_STATE_DIR`, guarded by a
file lock, committed atomically, and chained into a hash-linked audit log.
Statuses:

```
proposed → policy_accepted → simulated → approval_pending → approved
  → timelocked → ready → signed → submitted → confirmed
```

with `rejected`, `cancelled`, `expired`, `failed`, `indeterminate`, and
`quarantined` as exits. `confirmed`, `rejected`, `cancelled`, `expired` and
`failed` are terminal; everything else still reserves funds against the
limits and the protected reserve. The transaction id is derived from the
canonical intent hash (`fin_<first-32-hex>`), so the same intent cannot be
double-opened.

Progress past proposal is driven by **owner events**, not by the model:
`singularity-finance-mcp owner-event <absolute-file>` ingests one signed
event — simulation accepted, approval granted, signed, submitted, confirmed,
rejected, failed, indeterminate, reconciled-not-submitted, or quarantined —
each verified against the corresponding custody authority or approver key.

## Execution

`finance_execute` accepts only a transaction in `signed` state with no
unresolved reconciliation requirement. Then, in order:

1. Under the state lock, the transaction is durably marked `indeterminate`
   with `reconciliation_required` set, and the dispatch is committed —
   before the executor starts, so a timeout or crash can never cause a
   silent duplicate effect.
2. The canonical intent (`singularity.finance.execute.v1`: transaction id,
   intent hash, beneficiary, destination, asset, amount, purpose,
   parameters, expiry, policy id and version) is written to the stdin of
   the absolute executable named by `SINGULARITY_FINANCE_EXECUTOR`.
3. The executor's response must carry an executor id present in the
   policy's executor authorities, a valid signature over its reference
   hash, and a WORM receipt file that verifies against the policy's receipt
   key, sink id, transaction and intent hash.
4. Only then does the service append its own WORM record and transition the
   transaction to `submitted`. Confirmation still requires a reconciler's
   owner event.

## The HTTP executor adapter

`singularity-finance-executor-http` is the concrete executor shipped with
the release. It reads the canonical intent from stdin (64 KiB cap), posts
it to `SINGULARITY_FINANCE_CUSTODY_URL` — which must be a credential-free
HTTPS URL — authenticated with a bearer token from the owner-only
`SINGULARITY_FINANCE_CUSTODY_TOKEN_FILE`, with ambient proxies and
redirects disabled and a 120 s timeout. It validates the custody response
(executor id, 64-hex reference hash, 128-hex signature, absolute WORM
receipt path, 64 KiB cap) before writing it to stdout, and zeroizes the
token in memory on every path. The custody service behind that URL — not
the adapter, not the MCP service, and never the model process — holds the
signing keys.

The environment contract for both processes is listed in
[configuration](configuration.md).
