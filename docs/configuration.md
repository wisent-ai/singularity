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

