# Configuration

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
SINGULARITY_CYCLE_INTERVAL_SECS
SINGULARITY_STATE_DIR
SINGULARITY_RESUME
```

`run`, `once`, `doctor`, and `ecosystem run` require an explicitly declared
`SINGULARITY_STARTING_BALANCE_USD` (`--starting-balance`),
`SINGULARITY_INSTANCE_USD_PER_HOUR` (`--instance-price`), and
`SINGULARITY_CYCLE_INTERVAL_SECS` (`--cycle-interval-secs`), alongside the
identity and service inputs below. There is no assumed initial balance, free
host, or pause between cycles. For example, add these declared values to the
runtime invocation:

```text
--starting-balance <declared-usd> --instance-price <declared-usd-per-hour> --cycle-interval-secs <declared-seconds>
```

Omitting any of them is a parser refusal naming the missing option, with exit
status 2, before state creation or service access. A stated zero price means
the host is declared free; a missing price does not. Balances and prices must
not be negative. The initial balance is used for a new being only; resuming
one retains its saved budget. These values are accounting declarations, not
a deposit or transfer of funds.

A child receives the parent's resolved initial-budget, host-price and cycle
pause declarations explicitly, including values supplied as command-line
flags rather than environment variables. Its new budget uses that declared
initial allowance; it does not copy the parent's remaining balance.

The CLI refusal regression is `cargo test --test launch-declarations`. It
starts the real binary without inherited environment values and checks each
missing declaration before runtime startup. It does not replace a live
`once` or child-spawn run against the configured Brama and Las services.

`ecosystem run` uses the same runtime declarations. Its control, reconciliation
and dispatch timers use `SINGULARITY_CYCLE_INTERVAL_SECS`, which must be
positive for that command. The portfolio's model allocations and observation
and review intervals remain the separate values in its signed policy.
Read-only ecosystem commands do not take these runtime accounting inputs.

Brama:

```text
BRAMA_BASE_URL
BRAMA_MODEL
BRAMA_HMAC_SECRET_FILE
BRAMA_MAX_TOKENS
BRAMA_TEMPERATURE
```

Every model round is charged at the price Brama's caller-scoped catalog
(`GET /v1/models`) states for the model that served it, cache reads and writes
included; there is no configured token price. A served model the catalog does
not list, or lists without positive prices, ends the cycle with `model <id> is
absent from the caller-scoped catalog; its price is unknown` or the cost
admission refusal, instead of being charged as free. A cycle runs until the
model answers without calling a tool; no round count is configured, and the
solvency gate ends a cycle the budget can no longer pay for.

```text
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

`BRAMA_BASE_URL` (`--brama-url`) and `BRAMA_MODEL` (`--brama-model`) are
required: no Brama address or model is assumed, and a missing one is refused
by the parser before anything starts. `SINGULARITY_AGENT_NAME` and
`SINGULARITY_AGENT_TICKER` name the being; no persona is assumed.
`LAS_COMMAND`, `LAS_MCP_ENTRYPOINT` and `LAS_ONLY` declare the program that
runs Las, its MCP entrypoint and the surfaces this being is served; no
checkout location or surface set is assumed (`LAS_ONLY=` serves every surface
Las offers). `MOST_BASE_URL` (`--most-url`) is required whenever a Most
credential is configured, through `MOST_SERVICE_TOKEN_FILE` or the bootstrap
handoff; without it the runtime refuses with `a Most credential is configured
but MOST_BASE_URL (--most-url) is not`. A child being is handed the parent's
Brama address and model, Las declaration and Most address explicitly, so it
reaches the same services as its parent.

`BRAMA_MAX_TOKENS` (`--max-tokens`) and `BRAMA_TEMPERATURE` (`--temperature`)
are optional and have no default. Without `BRAMA_MAX_TOKENS` no output limit is
sent: the model's own `max_output_tokens` from Brama's catalog applies, and cost
admission quotes the call against that limit. Without `BRAMA_TEMPERATURE` no
temperature is sent and the provider's default applies. A stated temperature
must be finite and not negative (`configuration: temperature must be finite and
not negative`); its upper end is the provider's, which refuses a value outside
its range with its own message.

First-use journey and logging:

```text
STADO_INTEGRATION_API_URL
SINGULARITY_STADO_INTEGRATION_TOKEN
SINGULARITY_ONBOARDING_STATE_PATH
XDG_STATE_HOME
RUST_LOG
```

The journey runs offline when neither Stado variable is set; one of the two
without the other, either of them set to an empty value, or an endpoint the
integration transport cannot use is refused instead of quietly going offline.
The journey state needs `SINGULARITY_ONBOARDING_STATE_PATH`, `XDG_STATE_HOME`
or `HOME`, and the device identity needs `USER`; without them the command says
so rather than writing somewhere else. A `RUST_LOG` that is set but unparsable
is refused; unset means `info`.

Ecosystem store:

```text
SINGULARITY_STADO_HOME
SINGULARITY_DATABASE_URL
SINGULARITY_DATABASE_CA_FILE
```

The ecosystem store is the fleet database `singularity`, reached through
`stado-database`: `stado database resolve singularity --consumer singularity
--json` names its credential item, and `stado service directory connect
skarbiec --consumer singularity --json` names the route. The connector reads
`pooler_url` and `ca_certificate` with `stado credentials get <resolved-item>
--field <field> --route <resolved-url> --consumer
singularity-database-client --grant-file
<home>/.stado/singularity-database-client-skarbiec-token`, not the store
administrator. The home is `SINGULARITY_STADO_HOME` when set, else `HOME`,
so an isolated being still reaches the fleet as its host. A failure names
the failed resolve, route, delegated field read or connection and Stado's
status and error.

Without Stado or Skarbiec, `SINGULARITY_DATABASE_URL` (`postgres://` or
`mysql://`) and `SINGULARITY_DATABASE_CA_FILE` (the PEM bundle the server is
verified against) name the store directly and none of those steps runs; a URL
without its certificate file is refused with both variables named.

The bootstrap also binds the runtime to its workload identity, host, role,
environment, executable digest, code digest and policy sequence. `singularity
ticket launch` reads both digests from `--executable`: a compiled being runs no
code but its executable, so a service declaration names the installed binary
and never restates a digest that changes with every release.

Every other `ticket launch` input is a flag or the `SINGULARITY_LAUNCH_*`
variable of the same name (`--policy-file` is `SINGULARITY_LAUNCH_POLICY_FILE`,
`--supervisor-key` is `SINGULARITY_LAUNCH_SUPERVISOR_KEY`, and so on; `singularity
ticket launch --help` lists each). Stado's catalog service runs the same
`singularity ticket launch` on every host, and each host's unit environment
(`stado service env-set singularity --host … --key … --env-file <unit> --value-file …`) states that host's
supervisor key, trust root, policy file, Skarbiec resources, lifetimes, runtime
root and the being's own arguments: `SINGULARITY_LAUNCH_BEING_ARGS` is a JSON
array of strings, for example
`["ecosystem","run","--policy","/path/policy.json", …]`, used when nothing
follows `--`; both at once, or neither, is refused. A missing input is refused
by clap with its flag and variable named. None of them reaches the being: the
bootstrap clears the environment before it starts `singularity`, so the being
sees only its arguments and the identity the ticket binds.

`--resume` (`SINGULARITY_RESUME`) continues the being stored in the state
directory when who it is matches the configuration: agent id, name, ticker,
type, specialty, role, environment, host and workload id. What every
`singularity ticket launch` issues anew (the workload key, the executable and
code digests, the policy digest) is taken from the new start, so a restart,
an upgrade or a new policy keeps the being's mind, memory and actions. A
policy sequence older than the one the state last ran under is refused, as is
any change to who the being is; the refusal names each differing field with
its stored and configured value.

`singularity-bootstrap` is not a resident wrapper. It verifies the signed
manifest, redeems the Brama HMAC, Brama bearer and Most token from Skarbiec,
writes each into an owner-only file that it unlinks at once, and then replaces
its own process image with `singularity` (same PID, same service unit). The
credentials cross that exec only as open descriptors named by
`SINGULARITY_BRAMA_HMAC_FD`, `SINGULARITY_BRAMA_BEARER_FD` and
`SINGULARITY_MOST_TOKEN_FD`; no credential file exists on disk while the agent
runs. `singularity` refuses a descriptor that is not an unlinked, owner-only,
non-empty regular file, keeps them close-on-exec, and hands them on only to a
child agent it spawns itself. A `*_FILE` variable set explicitly still takes
precedence over the handoff.

