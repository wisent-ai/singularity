# Walkthrough: inspecting the state stores

Singularity's honesty lives on disk: a being, a finance surface, and a repo
surface each keep an owner-only store you can read with `jq` and `ls`. This
page inspects the stores produced by
[walkthrough-first-cycle](walkthrough-first-cycle.md) and
[walkthrough-finance](walkthrough-finance.md), verbatim.

## The being: `state.json`

One file is the entire being. After two failed-at-Brama cycles in the
sandbox:

```
$ jq . $SBX/state/state.json
{
  "schema_version": "being-v1",
  "identity": {
    "agent_id": "walkthrough-being",
    "name": "MyAgent",
    "ticker": "AGENT",
    "agent_type": "general",
    "specialty": "general",
    "role": "walkthrough",
    "environment": "local-walkthrough",
    "host": "local",
    "workload_id": "walkthrough-workload",
    "workload_public_key": "aaaa…aaaa",
    "executable_digest": "bbbb…bbbb",
    "code_digest": "cccc…cccc",
    "policy_digest": "dddd…dddd",
    "policy_sequence": 1
  },
  "mind": {
    "system_prompt": "You are MyAgent, an autonomous digital being specialized in general. …",
    "rules": [],
    "learnings": [],
    "memories": [],
    "children": [],
    "current_model": "any"
  },
  "status": "stopped",
  "cycle": 2,
  "budget": {
    "starting": "10",
    "remaining": "10",
    "api_spent": "0",
    "instance_spent": "0",
    "total_tokens": 0,
    "earned": "0"
  },
  "conversation": [
    {
      "role": "user",
      "content": "Begin autonomous cycle 1. Balance: 10 USD. Earned: 0 USD. Net profit: 0 USD. Current model: any. Recent actions: . Inspect opportunities and choose the next useful action. A plain response ends only this cycle; the being continues living while solvent."
    },
    { "role": "user", "content": "Begin autonomous cycle 2. …" }
  ],
  "recent_actions": [],
  "created_resources": { "chat_ids": [], "message_ids": [] },
  "started_at": "2026-08-24T22:05:12.573198Z",
  "updated_at": "2026-08-24T22:05:13.708994Z"
}
```

Worth reading closely:

- **`cycle: 2` with an untouched budget** — two framing messages were saved
  before their cognition calls failed; the debit happens only after a
  completion, so a gateway outage costs nothing.
- **Budget values are strings** — exact decimals, never floats.
- **The conversation already carries both framing messages** — state is
  saved after framing, before the model call; a crash loses at most the
  round in flight.
- **`status: "stopped"`, not `failed`** — a transport error ends the run,
  but the shutdown path still persisted a clean final state.

## The being: `activity.jsonl`

The journal is append-only, one fsynced JSON object per line:

```
$ cat $SBX/state/activity.jsonl
{"type":"started","at":"2026-08-24T22:05:12.791934Z"}
{"type":"cycle_started","at":"2026-08-24T22:05:12.862103Z","cycle":1}
{"type":"stopped","at":"2026-08-24T22:05:12.930187Z","cycle":1,"status":"stopped"}
{"type":"started","at":"2026-08-24T22:05:13.654686Z"}
{"type":"cycle_started","at":"2026-08-24T22:05:13.671315Z","cycle":2}
{"type":"stopped","at":"2026-08-24T22:05:13.708996Z","cycle":2,"status":"stopped"}
```

A healthy earning cycle adds `model_completed`, `cost_debited`,
`tool_finished`, and `revenue_credited` lines between `cycle_started` and
`stopped`; a tolerated failure appears as `{"type":"warning", …}` with the
error string as `message`.

## The finance store

`SINGULARITY_FINANCE_STATE_DIR` after the finance walkthrough:

```
$ ls $FIN/state
audit  audit-head.json  commit-applied  commits  finance.lock
lease-anchor.json  policy-anchor.json  requests  transactions
```

The confirmed transaction records its complete history — every transition
with its actor:

```
$ jq '{status, transitions: [.transitions[] | {status, actor}]}' \
    $FIN/state/transactions/fin_fac7a76ce60a9aa6f56acb703bdfd834.json
{
  "status": "confirmed",
  "transitions": [
    {"status": "proposed",         "actor": "finance_monitor"},
    {"status": "policy_accepted",  "actor": "finance_monitor"},
    {"status": "approval_pending", "actor": "finance_monitor"},
    {"status": "simulated",        "actor": "external_simulator"},
    {"status": "approval_pending", "actor": "finance_monitor"},
    {"status": "approved",         "actor": "operator"},
    {"status": "timelocked",       "actor": "finance_monitor"},
    {"status": "ready",            "actor": "finance_monitor"},
    {"status": "signed",           "actor": "external_signer"},
    {"status": "indeterminate",    "actor": "finance_executor_dispatch"},
    {"status": "signed",           "actor": "external_reconciler"},
    {"status": "submitted",        "actor": "external_executor"},
    {"status": "confirmed",        "actor": "external_reconciler"}
  ]
}
```

The `indeterminate → signed` pair in the middle is the failed
`/usr/bin/false` dispatch and its reconciliation — the store keeps the
scar.

The audit chain is one file per event, named by zero-padded sequence and
content hash, each record carrying `previous_hash`:

```
$ ls $FIN/state/audit
00000000000000000001-ba12a4e62665ea01c75fe0055136289f54105f33757fdafb6290236dc5f444ee.json
00000000000000000002-46dca54e5a6c3916a9d42fe2977d184d378cef5c736815d2fb8f1750d335aab8.json
…
00000000000000000010-c0b918f19c024681e08cbc942395711755392a3824e08c5979ff87b7b3f59840.json

$ jq '{sequence, previous_hash, hash}' $FIN/state/audit-head.json
{
  "sequence": 10,
  "previous_hash": "c2b7b0be46ec44996db2af095d328aa3fb5ad588bb683abe8e273f8508089a04",
  "hash": "c0b918f19c024681e08cbc942395711755392a3824e08c5979ff87b7b3f59840"
}
```

Every service start re-validates the whole chain — sequence, linkage, file
names, and per-record hash — and refuses to serve over a corrupted record
(`state_error: invalid audit record: …`, `state_error: audit hash chain
validation failed`). `commits/` and `commit-applied/` are the write-ahead
journal and its applied markers: a commit written but not applied when the
process died is replayed on the next open, which is why a crash between
"journaled" and "applied" changes nothing.

The two anchors pin the newest accepted policy and lease — the rollback
refusals in [concepts/lease](concepts/lease.md) compare against exactly
these:

```
$ jq -c . $FIN/state/policy-anchor.json $FIN/state/lease-anchor.json
{"policy_id":"walkthrough-policy","version":1,"document_hash":"a45bff7e…"}
{"lease_id":"lease-3","issued_at":"2026-08-24T22:07:57Z","document_hash":"c2d3b048…"}
```

And the WORM sink holds one content-addressed copy per submission and
confirmation:

```
$ ls $FIN/worm
427c7ac73825ddcd40e21e452aa1bfa5f596103c8d13cb491ff0552add1ebd17.json
cbb855dde4c3e19df65ae9db91c6c1de9b1d21c380b68c32ecb4722c5311f470.json
```

## The repo store

`JEDEN_REPO_STATE_DIR` keeps `workspaces/` (the actual git worktrees),
`records/` (one JSON per workspace: branch, base commit, sealed
fingerprint, check evidence, commit, published flag), `requests/` (the
idempotency ledger), and `locks/` + `request-locks/` (flock files). The
full proposal lifecycle that populates it is captured in
[repo-surface](repo-surface.md).

## Permissions are part of the contract

Every store directory is `0700`, every file `0600`, and the finance and
repo services *verify* rather than assume: a group-readable policy or
transaction file is refused with its own sentence (see
[runbook](runbook.md)). If an inspection tool of yours rewrites permissions,
the services will tell you.
