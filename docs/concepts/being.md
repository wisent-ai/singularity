# Being

What survives when the process dies? The being: one owner-only state
directory whose `state.json` carries an identity, a mind, a budget, a
conversation, and a status, under the schema version `being-v1`. The process
is just the current animation of that state; killing it loses nothing, and
`singularity run --resume` continues the same being.

## What it is

`AgentState` is the whole being, serialized with `deny_unknown_fields`:

| Field | Content |
|---|---|
| `schema_version` | Always `being-v1`. Any other value is refused on load with `state: unsupported state schema <version>`. |
| `identity` | The immutable [identity](identity.md), fixed at first boot. |
| `mind` | The mutable [mind](mind.md): prompt, rules, learnings, memories, children, model choice. |
| `status` | `starting`, `running`, `stopping`, `stopped`, `exhausted`, or `failed`. |
| `cycle` | Monotonic cycle counter, persisted across resumes. |
| `budget` | The [earning and cost record](earnings.md). |
| `conversation` | Every chat message so far: framing messages, assistant replies, tool outcomes. |
| `recent_actions` | The last 100 `{cycle, tool, status, at}` records; the newest 10 are summarized in the next cycle's framing message. |
| `created_resources` | Most chat and message UUIDs the being has created in the world. |
| `started_at`, `updated_at` | Creation and last-mutation timestamps. |

## Lifecycle

Boot resolves exactly four cases, three of them refusals:

| State on disk | `--resume` | Outcome |
|---|---|---|
| none | no | A fresh being is created from the configured identity, the initial system prompt, and the starting balance. |
| none | yes | `state: resume requested but no state exists` (exit 4). |
| exists | no | `state: state already exists at <path>/state.json; use --resume or a new directory` (exit 4). |
| exists | yes | Continued — only if the stored identity equals the configured identity in every field; otherwise `state: resume identity does not match configuration` (exit 4). |

A booted being is saved as `running` after a `started` journal event. Cycles
mutate it under the rules in [the loop](../loop.md). Shutdown saves
`stopping`, stops the Las child, then persists the final status: `stopped`
while still solvent, `exhausted` when the balance is not above zero.

## Where it lives

The state directory (`SINGULARITY_STATE_DIR`, default `.singularity`) is
created mode `0700`; its files are `0600`:

| Path | Content |
|---|---|
| `state.json` | The entire `AgentState`. Saved atomically: unique temp file, fsync, rename — after boot, after every framing message, after every tool call, and at shutdown. A crash loses at most the round in flight. |
| `activity.jsonl` | Append-only journal, fsynced per event: `started`, `cycle_started`, `model_completed`, `tool_finished`, `cost_debited`, `revenue_credited`, `warning`, `stopped`. |
| `children/<uuid>/` | The independent state directory of each [child](child.md). |

## Commands

```bash
singularity once             # create (or refuse) and live one cycle
singularity run --resume     # continue the same being
jq .status .singularity/state.json
```

## Not to be confused with

- **The process** — a being outlives every process that animates it; two
  processes must never animate one state directory concurrently.
- **A [child](child.md)** — a separate being with its own state directory
  under `children/`, never a thread of the parent.
- **The workload** — `workload_id` and the digests in the
  [identity](identity.md) describe the deployment that runs the being, not
  the being itself; a new deployment resumes the same being only if every
  identity field matches.
