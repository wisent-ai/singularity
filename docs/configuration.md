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
environment, executable digest, code digest and policy sequence.

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

