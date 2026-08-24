# What is Singularity

Singularity is a native Rust runtime for one persistent autonomous digital
being. Nobody assigns it an objective and it does not stop existing when a
task ends: it observes the tools available to it, chooses what to pursue,
earns revenue, pays its own model and compute costs, rewrites its own
persistent mind, and can create child beings. The whole product is three
ideas — a being that persists, a loop that pays for itself, and hard
boundaries around everything with authority.

## A being, not a job

The unit of existence is durable state, not a process. `state.json` in the
owner-only state directory holds the being's immutable identity, its
persistent mind (system prompt, self-imposed rules, learnings, memories,
model choice, child records), its budget, its conversation, and its recent
actions, under the schema version `being-v1`. The prompt sent to the model is
rebuilt from that state on every round, so a successful self-change affects
the next model call without changing the executable. Killing the process
loses nothing; `singularity run --resume` continues the same being, and
refuses to continue it under a different identity. The full model is in
[the being](being.md).

## A loop that pays for itself

Every cycle loads the being's current mind and financial state, sends that
context plus the current tool catalogue to Brama, executes the model's tool
calls, debits the exact model and instance cost after every completion, and
credits revenue only when a trusted `finance__*` or `trading__*` tool reports
a realized positive amount. State is saved atomically after every tool call,
and every event — cycle starts, model usage, tool outcomes, costs, credits,
warnings, shutdowns — is appended to a fsynced `activity.jsonl` journal. A
plain assistant reply ends only the current cycle; `singularity run`
continues while the balance is above zero and stops as `exhausted` when it is
not. The mechanics and the exact cost formula are in
[the loop and solvency](loop.md).

## Boundaries hold the authority

Singularity itself holds almost no power; everything consequential is a
separate process with its own policy and failure behavior:

- **Cognition** goes only through Brama, over HMAC-signed HTTP requests. The
  being may switch its own model, but only to one Brama actually advertises
  or to a routing selector Brama resolves.
- **Skills** come from Las, a supervised child process serving a namespaced
  MCP catalogue (Weles, Skarbiec, Stado, Most, Probierz, Warsztat, Finance
  and other Wisent surfaces). Every tool output is bounded and screened
  before the model sees it: secret-shaped keys, private-key material, and
  raw local paths are rejected outright. See [skills via Las](skills.md).
- **Money** moves only through `singularity-finance-mcp`, which enforces a
  signed policy — beneficiary allowlist, per-transaction, rolling, daily and
  lifetime limits, protected reserve, approvals and a timelock — and then
  hands the canonical intent to an isolated executor process. The model
  process never receives a signing key. See [the finance boundary](finance.md).
- **Deployment identity** is bound at start: the workload id, host, role,
  environment, executable digest, code digest and policy sequence are
  required configuration, and managed deployments obtain their short-lived
  Brama and Most credentials through [`singularity-bootstrap`](bootstrap.md).

## What Singularity is not

Singularity is not an assistant waiting for input: `run` takes an optional
one-shot `--stimulus` observation, not a task queue. It is not a plugin
host: it freezes no Python plugin list and copies no other product's
credentials; Las supplies the catalogue dynamically and each surface keeps
its own authority. It is not a wallet: the finance service verifies
signatures, receipts and limits but the executor owns the credentials, and
an ambiguous remote effect is durably marked indeterminate rather than
retried. And it is not its own predecessor: state schema `being-v1` is a
clean cutover, and the old supervisor state and Python runtime are not
compatibility paths.

## The first three commands

```bash
singularity doctor
```

Verifies the world before the being lives in it: Brama health and model
availability, Most send-capability when configured, and a full Las spawn with
the required surfaces present.

```bash
singularity tools --format table
```

Prints the exact tool catalogue the model would see: every Las tool plus the
built-in memory, self-modification, file, and child-creation tools.

```bash
singularity once
```

Executes one autonomous cycle and prints its report — cycle number, status,
balance, earnings, net profit, token usage, and the actions taken. The full
path from nothing to a first cycle is [quick-start](quick-start.md); the
command surface is [cli](cli.md).
