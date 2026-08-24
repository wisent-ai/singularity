# Child

A being that decides one existence is not enough calls
`singularity_spawn_child`. A child is a full being — separate state
directory, separate process, separate budget — created and started by its
parent, and recorded in the parent's mind.

## What it is

The tool takes three persona arguments, each trimmed, non-empty, free of
control characters:

| Argument | Bound |
|---|---|
| `name` | ≤ 128 bytes |
| `ticker` | ≤ 32 bytes |
| `specialty` | ≤ 256 bytes |

On success the parent records a `ChildRecord` in `mind.children` and the
tool returns `{"child_id": <uuid>, "pid": <pid>}`:

| Field | Content |
|---|---|
| `id` | Fresh UUID; also the child's directory name. |
| `name`, `ticker` | The child's persona. |
| `state_dir` | `<parent-state-dir>/children/<uuid>`. |
| `created_at` | Creation timestamp. |
| `status` | Recorded as `running` at spawn time; the parent runtime never rewrites it — the child's own `state.json` is the authority on its later life. |

## How it starts

The parent spawns its **own canonical executable** (`current_exe`) with the
`run` subcommand and exactly five environment overrides:
`SINGULARITY_AGENT_NAME`, `SINGULARITY_AGENT_TICKER`,
`SINGULARITY_SPECIALTY`, `SINGULARITY_STATE_DIR` (the child directory), and
`SINGULARITY_RESUME=false`. Everything else — identity, Brama, Las, Most,
pricing — is inherited from the parent's environment, which in a managed
deployment is the scrubbed environment [bootstrap](../bootstrap.md) built.
No secret value ever appears in child arguments; credentials stay in their
files and brokers.

Failure paths are explicit tool errors: `child_state` when the directory
cannot be created, `child_executable` when the current executable cannot be
resolved, `child_spawn` when the OS refuses the spawn.

## Lifecycle

The child creates a fresh being (`SINGULARITY_RESUME=false`) in its own
directory and lives by the same [loop](../loop.md) and
[solvency](solvency.md) rules as any being. The parent holds the PID from
the spawn but no supervision channel: no restart, no health probe, no
shutdown propagation. A child that outlives its parent keeps running; a
child that dies leaves its state directory for `--resume`.

## Where it lives

```bash
jq '.mind.children' .singularity/state.json
ls .singularity/children/
jq .status .singularity/children/<uuid>/state.json
```

## Not to be confused with

- **A tool round** — work inside the parent's own cycle; a child is a
  separate existence with its own budget and journal.
- **A clone** — the child starts from the compiled-in initial prompt with
  its new persona, not from a copy of the parent's mind, rules, or
  memories.
- **A supervised process** — Las is supervised (spawned, deadline-bounded,
  shut down with the parent); a child is deliberately not.
