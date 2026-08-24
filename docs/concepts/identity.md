# Identity

Who is this being, and under which exact code and policy does it run?
`AgentIdentity` answers both, is fixed at boot from configuration, and is
never edited by the model. It is the resume gate: continuing a being under a
different identity is refused outright.

## Fields

| Field | Meaning | Validation |
|---|---|---|
| `agent_id` | Immutable agent identifier; also the Brama request identity and the Las/Skarbiec identity. | identifier |
| `name`, `ticker`, `agent_type`, `specialty` | Persona; `name` and `specialty` are woven into the initial system prompt. | free text (defaults `MyAgent`, `AGENT`, `general`, `general`) |
| `role`, `environment`, `host`, `workload_id` | Workload binding of this deployment. | identifier |
| `workload_public_key` | The workload's Ed25519 public key. | 64 lowercase hex |
| `executable_digest`, `code_digest`, `policy_digest` | SHA-256 of the exact executable, source, and policy. | 64 lowercase hex |
| `policy_sequence` | Monotonic policy sequence number. | `u64`, required |

An *identifier* is 1–128 bytes of ASCII alphanumerics plus `- _ . :`. The
refusal sentences are exact:

```
singularity: configuration: <label> is not a valid immutable identifier
singularity: configuration: <label> must be 64 lowercase hexadecimal characters
```

both exit 2, where `<label>` is `agent id`, `role`, `environment`, `host`,
`workload id`, `workload public key`, `executable digest`, `code digest`, or
`policy digest`.

## Where each field goes

- `agent_id` is sent as `x-agent-id` on every Brama request and signed into
  the HMAC message (see [architecture](../architecture.md)); it is passed to
  Las as the MCP `agentId` and, when the `skarbiec` surface is selected,
  exported to the Las child as `SKARBIEC_MCP_AGENT_ID`. Selecting `skarbiec`
  without an agent id is refused before spawn:
  `mcp: Skarbiec requires an explicit immutable Las agent identity`.
- The whole struct is embedded in `state.json` with `deny_unknown_fields`,
  so a state file with extra identity fields does not parse.
- In a managed deployment every identity field originates from the signed
  [bootstrap](../bootstrap.md) manifest and is injected as environment
  variables into a scrubbed process environment.

## Lifecycle

Written once when the being is created; compared on every resume:

```
singularity: state: resume identity does not match configuration
```

(exit 4) whenever any field — including persona fields and digests — differs
between the stored state and the current configuration. There is no partial
match and no migration path; a new deployment that changes `code_digest`
must present the same value it wants to be compared against.

## Commands

```bash
jq .identity .singularity/state.json
singularity once --resume     # succeeds only under the exact same identity
```

## Not to be confused with

- **The persona** — `name`/`ticker`/`agent_type`/`specialty` shape the
  initial prompt but are still identity fields: changing them also fails the
  resume gate.
- **The [mind](mind.md)** — everything the being may change about itself
  lives there; nothing in the identity is model-writable.
- **The Brama HMAC secret** — a credential, not an identity; it rotates
  freely (see [bootstrap](../bootstrap.md)) while the identity stays fixed.
