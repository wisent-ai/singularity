# Bootstrap

`singularity-bootstrap` is how a managed deployment starts a being without
ever placing a long-lived secret in its environment. It verifies a signed
launch manifest, proves the workload's identity to a capability broker,
materializes short-lived Brama and Most credentials into a private runtime
directory, launches the runtime with a scrubbed environment, and removes the
credentials when the runtime exits.

## Invocation

```bash
singularity-bootstrap \
  --manifest <file>            # SINGULARITY_BOOTSTRAP_MANIFEST
  --manifest-signature <file>  # SINGULARITY_BOOTSTRAP_MANIFEST_SIG
  --trust-root <file>          # SINGULARITY_BOOTSTRAP_TRUST_ROOT
  --runtime-root <dir>         # SINGULARITY_RUNTIME_ROOT (absolute)
```

Manifest, signature and trust root must be owner-only files. The exit code
is the launched runtime's own exit status.

## The signed manifest

The manifest is verified as an Ed25519 signature over the domain-separated
manifest bytes (`SINGULARITY-BOOTSTRAP-MANIFEST\0v1\0` prefix) against the
32-byte trust root, then validated structurally:

- `version` must be `singularity.bootstrap.v1`;
- it must be fresh: not expired, issued at most 30 s in the future, and
  with a total lifetime of at most 300 s — a manifest is a launch ticket,
  not a config file;
- `workload_public_key`, `executable_digest`, `code_digest`,
  `policy_digest` and both capability ids must be 64 lowercase hex;
- the identity atoms (`agent_id`, `role`, `environment`, `host`,
  `workload_id`) must be non-empty, trimmed, NUL-free and bounded;
- the two capabilities must be distinct and correctly bound: the Brama
  capability to purpose `singularity.brama.bootstrap` and a `brama:`
  resource, the Most capability to `singularity.most.bootstrap` and a
  `most:` resource;
- `broker_socket`, `workload_private_key_file` and
  `singularity_executable` must be absolute, and the private key file
  owner-only.

Before anything launches, the executable named by the manifest is hashed
and must match `executable_digest`, and the workload private key must
derive exactly the manifest's `workload_public_key`. The key bytes are
zeroized after the signing key is constructed.

## Capability redemption

For each capability, the bootstrap connects to the broker's Unix socket and
speaks the `skarbiec.redeem.v1` wire protocol: it receives a nonce, signs a
domain-separated workload proof (`SKARBIEC-WORKLOAD-PROOF\0v1\0`) with the
workload key, and receives the secret only on an `ok` control response.
Control lines are capped at 4 KiB, secrets at 64 KiB, and every socket
operation carries a 5 s I/O deadline.

Secrets land in a fresh `singularity-<uuid>` directory (mode `0700`) under
the runtime root, as `brama.hmac` and `most.token`. The directory is
removed — files overwritten then deleted — when the runtime exits, on
success and failure alike.

## Launch

The runtime starts from `env_clear()` plus exactly: a fixed `PATH`,
`LANG`/`LC_ALL`, the identity variables (`SINGULARITY_AGENT_ID`,
`SINGULARITY_ROLE`, `SINGULARITY_ENVIRONMENT`, `SINGULARITY_HOST`,
`SINGULARITY_WORKLOAD_ID`, `SINGULARITY_WORKLOAD_PUBLIC_KEY`,
`SINGULARITY_EXECUTABLE_SHA256`, `SINGULARITY_CODE_SHA256`,
`SINGULARITY_POLICY_SHA256`, `SINGULARITY_POLICY_SEQUENCE`), and the two
materialized credential paths as `BRAMA_HMAC_SECRET_FILE` and
`MOST_SERVICE_TOKEN_FILE`. The manifest's `singularity_args` select the
subcommand. Nothing else from the bootstrap's environment leaks into the
being.

## Deployment scaffolding

The repository carries the deployment pieces under `deploy/capabilities/`:
systemd units and drop-ins for the capability broker and agent workloads, a
launchd template for macOS, example environment files, sysusers/tmpfiles
definitions, and `capability-preflight.py`, which the test suite exercises.
The runtime's own configuration contract is [configuration](configuration.md).
