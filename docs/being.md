# The being

One being is one owner-only state directory. This page is the model of what
lives there: the immutable identity, the mutable mind, the built-in tools the
being uses on itself, the children it creates, and the exact files on disk.

## Identity

`AgentIdentity` is fixed at boot from configuration and never edited by the
model:

| Field | Meaning |
|---|---|
| `agent_id` | Immutable agent identifier; also the Las and Brama identity. |
| `name`, `ticker`, `agent_type`, `specialty` | Persona; the specialty is woven into the initial system prompt. |
| `role`, `environment`, `host`, `workload_id`, `workload_public_key` | Workload binding of this deployment. |
| `executable_digest`, `code_digest`, `policy_digest`, `policy_sequence` | The exact code and policy this being runs under. |

Identity is the resume gate: `run --resume` refuses state whose stored
identity differs in any field from the configured one, and the state schema
rejects unknown fields outright.

## Mind

`BeingMind` is everything the being may change about itself:

| Field | Changed by |
|---|---|
| `system_prompt` | `singularity_self_set_prompt` |
| `rules` | `singularity_self_add_rule` (append-only) |
| `learnings` | `singularity_self_add_learning` (append-only) |
| `memories` | `singularity_memory_remember` / `singularity_memory_recall` |
| `children` | `singularity_spawn_child` |
| `current_model` | `singularity_self_switch_model` |

The prompt Brama receives is rebuilt from the mind on every model round: the
current system prompt, then a `Self-imposed rules:` list, then a
`Persistent learnings:` list, then the 20 most recent memories as
`kind: text` lines, followed by the running conversation. A self-change
therefore takes effect on the very next model call.

The initial system prompt frames the existence: the being is not waiting for
an assignment, its existence has real costs, and it is to create value, earn
more than it spends, preserve solvency, learn, improve its own mind, and
create child beings when that increases durable value, using only supplied
tools and never inventing results or credentials.

## Built-in self tools

These tools are compiled into the runtime and act on the being's own state,
not on the world:

| Tool | Contract |
|---|---|
| `singularity_memory_remember` | Persist one memory: `kind` (≤64 bytes) and `text` (≤16 KiB). Keeps at most 1000 memories, dropping the oldest. Returns the `memory_id`. |
| `singularity_memory_recall` | Case-insensitive substring search over kind and text; returns the 50 most recent matches. |
| `singularity_self_set_prompt` | Replace the persistent system prompt (≤64 KiB). |
| `singularity_self_add_rule` | Append one self-imposed rule (≤4 KiB). |
| `singularity_self_add_learning` | Append one learning (≤8 KiB). |
| `singularity_self_switch_model` | Switch future cognition to another model. Accepted only if the name is a selector (`any`, `any-vision-capable`, `best`, `task:*`) or is currently advertised by Brama `/v1/models`; otherwise fails with `model_unavailable`. |
| `singularity_file_read` | Read one UTF-8 file inside the workspace, ≤2 MiB. |
| `singularity_file_write` | Atomically create or replace one UTF-8 file inside the workspace, ≤2 MiB. |
| `singularity_spawn_child` | Create and start a child being; below. |

All text arguments are trimmed, must be non-empty, and must contain no
control characters.

### Workspace confinement

`singularity_file_read` and `singularity_file_write` are confined to
`SINGULARITY_WORKSPACE`. Paths must be relative and contain only normal
components — no absolute paths, no `..`; the resolved path is canonicalized
and must still start with the workspace, or the call fails with
`workspace_boundary`. Writes go to a temporary file then rename, refuse to
replace symlinks or non-regular files, and report the byte count written.

## Children

`singularity_spawn_child` takes `name` (≤128), `ticker` (≤32) and
`specialty` (≤256), creates a separate state directory
`<state-dir>/children/<uuid>`, and starts the same canonical executable with
`run`, passing the new persona and state directory through
`SINGULARITY_AGENT_NAME`, `SINGULARITY_AGENT_TICKER`,
`SINGULARITY_SPECIALTY`, `SINGULARITY_STATE_DIR`, and
`SINGULARITY_RESUME=false`. The parent records a `ChildRecord` (id, name,
ticker, state dir, creation time, status) in its mind and returns the child
id and PID. Secrets are never placed in child arguments; managed
deployments supply Brama, Las, Most and capability configuration through
inherited workload policy.

## State on disk

The state directory is created mode `0700`; its files are `0600`.

| Path | Content |
|---|---|
| `state.json` | The entire being: `schema_version` (`being-v1`), identity, mind, status, cycle counter, budget, conversation, the last 100 actions, created Most chat and message ids, timestamps. Saved atomically — unique temp file, fsync, rename. |
| `activity.jsonl` | Append-only journal, fsynced per event: `started`, `cycle_started`, `model_completed`, `tool_finished`, `cost_debited`, `revenue_credited`, `warning`, `stopped`. |
| `children/<id>/` | Independent state directory of each child being. |

`status` is one of `starting`, `running`, `stopping`, `stopped`,
`exhausted`, `failed`. Any `state.json` whose `schema_version` is not
`being-v1` is refused; the previous supervisor state and the old Python
runtime are not compatibility paths. What the budget fields mean, and when
they change, is [the loop and solvency](loop.md).
