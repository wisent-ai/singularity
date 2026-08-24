# Architecture

Singularity's design premise is that the model process is the least trusted
component in its own life. The runtime that hosts cognition owns almost
nothing: every consequential authority — models, skills, money, code,
credentials — lives in a separate process behind a verified boundary, and
the runtime's job is to persist state honestly and route intent across
those boundaries.

## The processes

```
singularity-bootstrap ──launches──► singularity (the being runtime)
                                       │
                                       ├── HTTPS + HMAC ──► Brama (model gateway)
                                       ├── HTTPS bearer ──► Most (messaging)
                                       └── stdio MCP ─────► Las (skill catalogue, child process)
                                                              ├── weles, skarbiec, stado, …
                                                              ├── warsztat ── singularity-repo-mcp
                                                              └── finance ─── singularity-finance-mcp
                                                                                 └── stdin/stdout ► executor
                                                                                     (singularity-finance-executor-http ► custody)
```

One deployment runs one `singularity` process per being. Las is a
supervised child (spawned at boot, deadline-bounded per request, shut down
with the being); the MCP servers are separate services reached *through*
Las as namespaced surfaces; the finance executor is a per-execution child
of the finance service, never of the runtime.

## What the runtime owns

- **The being's state** — `state.json` and `activity.jsonl`, owner-only,
  saved atomically after every mutation ([concepts/being](concepts/being.md)).
- **The accounting** — exact-decimal debit after every completion, credit
  only from trusted tool evidence
  ([concepts/earnings](concepts/earnings.md)).
- **The catalogue merge** — Las tools plus built-ins, strict names, no
  duplicates ([concepts/skill](concepts/skill.md)).
- **The output boundary** — every tool result is screened before the model
  sees it; violations are replaced, not truncated ([skills](skills.md)).
- **The loop** — framing, rounds, journaling, shutdown ([loop](loop.md)).

## What it deliberately does not own

| Authority | Owner | The runtime's only lever |
|---|---|---|
| Model weights, routing, provider keys | Brama | Signed HTTPS requests; the being may pick only models Brama advertises. |
| Skill implementations and their credentials | Each surface behind Las | A namespaced `tools/call`; surfaces keep their own policy and failure behavior. |
| Narrow capabilities | Skarbiec | Environment pass-through to the Las child, plus the immutable agent id. |
| Money | `singularity-finance-mcp` + isolated executor + custody | `finance_propose` / `finance_status` / `finance_cancel` / `finance_execute` — and no key, ever. |
| Repository changes | `singularity-repo-mcp`, then external CI and human review | Policy-bound proposal tools via the `warsztat` surface. |
| Long-lived credentials | The capability broker | `singularity-bootstrap` redeems short-lived secrets into files the runtime only reads. |

## How data flows through one cycle

1. **State → prompt.** The system message is rebuilt from the mind; the
   framing message states balance, earnings, profit, model, and recent
   actions. Nothing reaches the model that is not in state or catalogue.
2. **Prompt → Brama.** The request is signed HMAC-SHA256 over
   `agent_id:timestamp:body_sha256` and carried in four `x-agent-*`
   headers. The response is validated structurally: exactly one choice, a
   coherent finish reason, usage totals that add up.
3. **Completion → debit.** Cost lands in the budget and journal before any
   tool runs.
4. **Tool call → boundary.** Built-ins mutate state or the confined
   workspace; Las calls cross the MCP channel under a deadline; Most calls
   go direct with the bearer token. Every result crosses the output
   boundary; every outcome is journaled and appended to the conversation.
5. **Evidence → credit.** Only `finance__*`/`trading__*` results with a
   positive `revenue_usd`/`realized_profit_usd` move the balance up.
6. **Save.** State is durably rewritten after each tool call; a crash
   loses at most the round in flight.

## Trust boundaries and their mechanisms

- **Process environment.** The Las child starts from `env_clear()` plus a
  fixed `PATH`, locale, the `LAS_*` selection and pinning variables, and —
  only when Skarbiec is selected — the `SKARBIEC_*` pass-through and the
  agent id. The bootstrap launches the runtime the same way; the repo
  surface runs `git`/`gh` from scrubbed environments with profile-scoped
  inheritance (`HOME`+`SSH_AUTH_SOCK` for git network, `GH_TOKEN`+`GH_HOST`
  for gh). Ambient credentials do not leak sideways.
- **Secrets are files, never arguments.** Owner-only (`0600`), non-symlink,
  non-empty; group/world access is refused at boot. Child beings receive
  persona and paths, never secret values.
- **Signed documents everywhere authority crosses a file.** The finance
  policy, enable lease, owner events, and WORM receipts; the bootstrap
  manifest (domain-separated, 300-second lifetime); the Las release
  manifest, signature, trust store, and watermark. Verification failure is
  a one-sentence refusal, not a warning.
- **Anchors against rollback.** The finance store pins the newest accepted
  policy version and lease `issued_at`; older documents are refused as
  rollback/equivocation ([concepts/lease](concepts/lease.md)).
- **Dispatch-before-effect.** `finance_execute` durably marks the
  transaction `indeterminate` with reconciliation required *before* the
  executor starts; the repo surface journals every request under
  `request_id` locks before acting. Crashes reconcile forward, never
  duplicate.
- **The output boundary.** Secret-shaped keys, private-key PEM blocks, raw
  local paths, NULs, oversize and over-deep results never enter the
  conversation — which means they also never enter `state.json` or any
  future prompt.

## Error classes are routing decisions

Every upstream error is classified `Permanent`, `Transient`, or
`Indeterminate` at the client that observed it (Brama, MCP, Most). The
agent loop turns the classes into behavior: permanent Brama errors abort a
`run`; transient errors are journaled warnings followed by the normal
sleep; indeterminate tool transports become `indeterminate` outcomes that
are surfaced to the model and never replayed by the runtime. The finance
service has the same idea in durable form: an ambiguous execution is a
recorded state (`indeterminate`, `reconciliation_required`) that only an
external reconciler's signed word can resolve.

## Scale posture

One being is one process, one state directory, one Las child. Fan-out is
explicit: [children](concepts/child.md) are separate beings with separate
budgets, spawned as detached processes. There is no shared store, no
scheduler, and no inter-being channel inside this product — beings that
need to coordinate do it through the same external surfaces as everything
else.
