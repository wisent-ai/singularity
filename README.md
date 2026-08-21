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

Required finance environment:

```text
SINGULARITY_FINANCE_POLICY_FILE
SINGULARITY_FINANCE_ENABLE_LEASE_FILE
SINGULARITY_FINANCE_STATE_DIR
SINGULARITY_FINANCE_VERIFY_KEY_HEX
SINGULARITY_FINANCE_BINARY_SHA256
SINGULARITY_FINANCE_EXECUTOR
```

## Child beings

`singularity_spawn_child` creates a separate owner-only state directory and
starts the same canonical executable with a new name, ticker and specialty.
Managed deployments provide Brama, Las, Most and capability configuration
through inherited workload policy; secrets remain in their files or brokers and
never enter child arguments.

## Commands

```text
singularity run     live continuously while solvent
singularity once    execute one autonomous cycle and print its report
singularity doctor  verify Brama, Las, Most and required surfaces
singularity tools   print the dynamic and built-in tool catalogue
```

## Configuration

Identity and accounting:

```text
SINGULARITY_AGENT_ID
SINGULARITY_AGENT_NAME
SINGULARITY_AGENT_TICKER
SINGULARITY_AGENT_TYPE
SINGULARITY_SPECIALTY
SINGULARITY_WORKSPACE
SINGULARITY_STIMULUS
SINGULARITY_STARTING_BALANCE_USD
SINGULARITY_INSTANCE_USD_PER_HOUR
SINGULARITY_STATE_DIR
SINGULARITY_RESUME
```

Brama:

```text
BRAMA_BASE_URL
BRAMA_MODEL
BRAMA_HMAC_SECRET_FILE
BRAMA_MAX_TOKENS
BRAMA_TEMPERATURE
BRAMA_INPUT_PRICE_USD_PER_MILLION
BRAMA_OUTPUT_PRICE_USD_PER_MILLION
```

Las and Most:

```text
LAS_COMMAND
LAS_MCP_ENTRYPOINT
LAS_ONLY
LAS_SKIP
LAS_RELEASE_MANIFEST_FILE
LAS_RELEASE_MANIFEST_SIGNATURE_FILE
LAS_RELEASE_TRUST_STORE_FILE
LAS_RELEASE_WATERMARK_FILE
SINGULARITY_REQUIRED_SURFACES
MOST_BASE_URL
MOST_SERVICE_TOKEN_FILE
```

The bootstrap also binds the runtime to its workload identity, host, role,
environment, executable digest, code digest and policy sequence.

## State

The owner-only state directory contains:

- `state.json`: identity, persistent mind, model choice, budget, earnings,
  conversation, memories, children and created resources;
- `activity.jsonl`: starts, cycles, model usage, tool outcomes, costs, credited
  revenue, warnings and shutdowns;
- `children/<id>/`: independent state for child beings.

State schema `being-v1` is a clean cutover. The previous supervisor state and
the old Python runtime are not compatibility paths.

## Build

```bash
cargo build --locked
cargo install --path . --locked
```

The package builds `singularity`, `singularity-bootstrap`,
`singularity-repo-mcp` and `singularity-finance-mcp`.

License: MIT.
