<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="singularity by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/singularity) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/singularity/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.com) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

# Singularity: An Autonomous Digital Being That Earns Its Existence

Singularity is a native Rust runtime for a persistent autonomous digital being.
Nobody assigns it an objective and it does not stop existing when one task ends.
It observes available tools, decides what to pursue, creates useful value,
earns revenue, pays its model and compute costs, learns from results, changes
its own persistent mind, and can create child beings.

## Runtime

```text
                    ┌──────── Brama ─────── model cognition
                    │
Singularity ────────┼──────── Las ───────── dynamic Wisent skills
continuous loop     │           ├── Weles: internet actions
identity + memory   │           ├── Most: communication
earnings + costs    │           ├── Stado: compute and placement
self-modification   │           ├── Skarbiec: scoped capabilities
child beings        │           ├── Warsztat: repository work
                    │           └── Finance: approved real execution
                    │
                    └──────── durable state and activity journal
```

Every cycle:

1. Loads the being's current prompt, self-imposed rules, learnings, memories,
   identity, financial state and recent actions.
2. Sends that context and the current dynamic tool catalogue to Brama.
3. Executes native model tool calls through Las or the built-in persistent
   memory, self-modification, model-switching and child-creation tools.
4. Records model and instance cost exactly once.
5. Credits revenue only when a trusted `finance__*` or `trading__*` tool reports
   realized revenue.
6. Atomically saves state and begins another cycle while the being is solvent.

A normal assistant response ends only the current cycle. `run` continues until
the process is cancelled or the balance reaches zero.

The `ecosystem` profile keeps a durable portfolio of observations, independently
reviewed opportunities, initiatives, executions and measured outcomes. Read the
[ecosystem operation contract](https://singularity.wisent.com/docs/ecosystem)
for its fixed policy, recovery rules, owner API and qualification requirements.
Provisioning and release are not evidence of customer value.

## Persistent mind

Singularity exposes these built-in tools to itself:

- `singularity_memory_remember` and `singularity_memory_recall`;
- `singularity_self_set_prompt`;
- `singularity_self_add_rule`;
- `singularity_self_add_learning`;
- `singularity_self_switch_model`;
- `singularity_spawn_child`;
- `singularity_file_read` and `singularity_file_write`, confined to
  `SINGULARITY_WORKSPACE`.

Rules, learnings, memories, model choice and child records live in `state.json`.
The prompt sent to Brama is rebuilt from that state every round, so a successful
self-change affects the next model call without changing the executable.

### Import existing memory, knowledge, and profile records

To create a new being with existing mind records before its first model call,
add the file to the normal fully configured startup command:

```text
singularity once --import-file /path/to/mind.json [normal required runtime flags]
singularity run --import-file /path/to/mind.json [normal required runtime flags]
```

For a being that already exists, including one currently owned by `run`, use:

```text
singularity import --file /path/to/mind.json --state-dir /path/to/state
singularity onboarding --import-file /path/to/mind.json --state-dir /path/to/state
```

The accepted document is strict `singularity-mind-import-v1` JSON:

```json
{
  "schema_version": "singularity-mind-import-v1",
  "source": { "kind": "<export type>", "id": "<stable source id>" },
  "memories": [{ "id": "<stable item id>", "text": "<existing memory>" }],
  "knowledge": [{ "id": "<stable item id>", "text": "<existing knowledge>" }],
  "profile": [{ "id": "<stable item id>", "text": "<existing profile fact>" }]
}
```

All three arrays are required and may be empty, but at least one real record is
required across them. Source and item IDs must be stable, nonempty, and unique
within the document. Unknown
fields, malformed JSON, symbolic links, files over 16 MiB, more than 1,000
records, empty text, NULs, and oversized fields are refused.

Singularity validates the entire document before mutation and saves `state.json`
once. Repeating the same source item with the same text is unchanged; repeating
it with different text refuses the entire import; the same existing text from a
different source adds provenance without duplicating the memory. The JSON result
reports `imported`, `attributed`, `unchanged`, `conflicting`, and `rejected`
counts plus item issues. Imported profile records are retained as profile-kind
mind memories: they never replace the being identity. Import does not change the
prompt, rules, model, budget, finance policy, or enabled tools.

When `singularity run` owns the state, the command submits to its owner-only
local state service. When stopped, the command writes through `ActivityStore`
directly. A missing being or an unavailable running state owner is refused
without creating a second store.


## Dynamic skills

Las supplies the current namespaced MCP catalogue. Singularity does not freeze a
Python plugin list or copy another product's credentials. Weles, Most, Stado,
Skarbiec, Probierz, Brama, Warsztat, Finance and future approved surfaces remain
separate processes with their own authority and failure behavior.

Tool output is bounded before returning to the model. Secret-shaped fields,
private-key material and raw local paths are rejected. An ambiguous remote
effect is recorded as indeterminate and is never automatically replayed.

## Financial execution

`singularity-finance-mcp` exposes:

- `finance_propose`;
- `finance_status`;
- `finance_cancel`;
- `finance_execute`.

A proposal must pass the signed beneficiary, asset, reserve, rolling-limit,
simulation, approval and timelock policy. `finance_execute` accepts only a
signed transaction with no unresolved reconciliation requirement, then sends
the exact canonical intent over stdin to the absolute executable named by
`SINGULARITY_FINANCE_EXECUTOR`.

Before starting the isolated executor, the finance service durably marks the
transaction indeterminate so a timeout or crash cannot trigger a duplicate
effect. The executor owns signing and network credentials, performs the real
operation, and returns its signed reference plus WORM receipt. The finance
service verifies the configured executor authority and receipt before recording
submission. The model process never receives the signing key.


The release includes `singularity-finance-executor-http`, a concrete executor
adapter. It forwards the canonical intent to a credential-free HTTPS custody
URL using an owner-only bearer file, disables ambient proxies and redirects,
and validates the executor ID, signed reference and WORM receipt path before
returning them to `singularity-finance-mcp`.
Required finance environment:

```text
SINGULARITY_FINANCE_POLICY_FILE
SINGULARITY_FINANCE_ENABLE_LEASE_FILE
SINGULARITY_FINANCE_STATE_DIR
SINGULARITY_FINANCE_VERIFY_KEY_HEX
SINGULARITY_FINANCE_BINARY_SHA256
SINGULARITY_FINANCE_EXECUTOR
SINGULARITY_FINANCE_CUSTODY_URL
SINGULARITY_FINANCE_CUSTODY_TOKEN_FILE
```

`cargo run --example finance_lifecycle` drives one transaction through
`singularity-finance-mcp` with seven generated authority keys and
`/usr/bin/false` as executor: proposals and their refusals, simulation and
approval. The policy's two-second timelock is real, so the command prints its
directory and stops; run it again with that directory once `timelock_until`
has passed to sign, dispatch, reconcile, submit, confirm and exercise the
enable lease (kill switch, fresh lease, rollback refused).
`docs/examples/sandbox-being.sh` bootstraps a being with no Brama, no Las
checkout and no credentials, against the zero-tool Las stand-in
`docs/examples/stub-las.sh`.

## Child beings

`singularity_spawn_child` creates a separate owner-only state directory and
starts the same canonical executable with a new name, ticker and specialty.
Managed deployments provide Brama, Las, Most and capability configuration
through inherited workload policy; secrets remain in their files or brokers and
never enter child arguments.

## Commands

```text
singularity run         live continuously while solvent
singularity once        execute one autonomous cycle and print its report
singularity import      import attributed memory, knowledge, and profile JSON
singularity onboarding  show first use; add --import-file or --reset
singularity doctor      verify Brama, Las, Most and required surfaces
singularity tools       print the dynamic and built-in tool catalogue
singularity ecosystem run --policy FILE [--start-paused] [--ready-json]  run the delegated portfolio
singularity ecosystem status --json     read the live portfolio owner
singularity ecosystem opportunities     read a page of hypotheses and decisions
singularity ecosystem initiatives       read a page of execution and delivery state
singularity ecosystem records [KIND]    browse retained evidence and event summaries
singularity ecosystem record KIND ID    read revision-bound evidence fragments
singularity ecosystem explain ID        read decision and delivery evidence
singularity ecosystem pause             stop new admission without replaying effects
singularity ecosystem resume            resume within the original delegation
singularity capability-preflight agent|broker-linux|broker-macos ENV_FILE [--exec]
                                        check a capability-isolated unit, then start it
singularity capability-preflight deployment-static DIR  check deploy/capabilities
```

`capability-preflight` is the `ExecStartPre` of the units in
`deploy/capabilities`. It reads the unit's environment file (literal
`NAME=VALUE` lines only; a quoted or expanded value, a placeholder marker, or a
secret passed by value instead of as a `_FILE`/`_PATH` reference is refused),
requires every file it names to be owner-only and not a symlink, checks the
release binary's SHA-256, and for an agent also the bootstrap manifest and the
0660 broker socket. `--exec` then replaces the process with the checked one
(the broker's `serve --no-http`, or `singularity-bootstrap`) under a cleared
environment. `broker-macos` is always refused: launchd has no egress sandbox.
Every refusal prints `capability-preflight: <reason>` and exits 78.

## Configuration

Every environment variable, and how `singularity-bootstrap` hands credentials
to the runtime, is listed in [docs/configuration.md](docs/configuration.md).

## State

The owner-only state directory contains:

- `state.json`: identity, persistent mind, model choice, budget, earnings,
  conversation, memories with optional import provenance, children and created
  resources;
- `activity.jsonl`: starts, cycles, model usage, tool outcomes, costs, credited
  revenue, mind imports, warnings and shutdowns;
- `state-import.sock`: owner-only local import boundary while `run` is active;
- `children/<id>/`: independent state for child beings.

Onboarding progress is stored separately at
`$XDG_STATE_HOME/singularity/onboarding.json` (or
`~/.local/state/singularity/onboarding.json`) and can be redirected with
`SINGULARITY_ONBOARDING_STATE_PATH`.

The ecosystem store (metadata, records and spend reservations, every row keyed
by the being's agent id) lives in the fleet database `singularity`, reached
through Stado's shared connector `stado-database` as a SeaORM connection:
`stado database resolve singularity`, the Skarbiec route, and the
`singularity-database-client` bearer in
`~/.stado/singularity-database-client-skarbiec-token`
(`SINGULARITY_STADO_HOME` names that home when `HOME` is isolated). A
`sea-orm-migration` migrator creates the tables. A leftover
`ecosystem.sqlite3` in the state directory is refused at startup, and every
connection failure names the step that failed.

State schema `being-v1` is a clean cutover. The previous supervisor state and
the old Python runtime are not compatibility paths.

## Build

```bash
cargo build --locked
cargo install --path . --locked
```
The package builds `singularity`, `singularity-bootstrap`,
`singularity-repo-mcp`, `singularity-finance-mcp`, and
`singularity-finance-executor-http`.

License: MIT.
