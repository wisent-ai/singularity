# Walkthrough: bootstrapping a being in a sandbox

How far does a being get on a machine with no Brama gateway, no Las
checkout, and no credentials? Far enough to prove every boot gate, create
real state, and show exactly where cognition would begin. Everything below
was executed against `singularity 0.5.0` in a throwaway directory; output
is pasted verbatim, and the whole session is reproducible with
[examples/sandbox-being.sh](examples/sandbox-being.sh). Where a stand-in is
used, it is labeled — nothing here touches an operator's state or spends
anything.

## The sandbox

```bash
SBX=$(mktemp -d /tmp/singularity-walkthrough.XXXXXX)
mkdir -p "$SBX/workspace" "$SBX/las"
printf 'walkthrough-not-a-real-secret\n' > "$SBX/brama.hmac"
chmod 600 "$SBX/brama.hmac"
# Las release pinning files only need to exist for the runtime's own checks:
printf 'stub\n' > "$SBX/las/release.manifest.json"   # and .sig, .trust, .watermark
```

Two deliberate substitutions, both visible in the results:

- `BRAMA_BASE_URL=http://127.0.0.1:9` — the discard port; every Brama call
  fails with a connection error instead of reaching any real gateway.
- `LAS_COMMAND=/usr/bin/python3` with a 30-line stub entrypoint that speaks
  MCP `2024-11-05` and advertises **zero tools** (the repository does not
  ship Las). The stub exists to exercise the runtime's own spawn,
  handshake, and catalogue logic; a real deployment points
  `LAS_MCP_ENTRYPOINT` at a Las checkout. The stub is reproduced in
  [examples/stub-las.py](examples/stub-las.py).

The identity is throwaway but well-formed (64-hex digests, valid
identifiers):

```bash
export SINGULARITY_AGENT_ID=walkthrough-being SINGULARITY_ROLE=walkthrough \
  SINGULARITY_ENVIRONMENT=local-walkthrough SINGULARITY_HOST=local \
  SINGULARITY_WORKLOAD_ID=walkthrough-workload \
  SINGULARITY_WORKLOAD_PUBLIC_KEY=aaaa…aa SINGULARITY_EXECUTABLE_SHA256=bbbb…bb \
  SINGULARITY_CODE_SHA256=cccc…cc SINGULARITY_POLICY_SHA256=dddd…dd \
  SINGULARITY_POLICY_SEQUENCE=1 \
  SINGULARITY_STATE_DIR=$SBX/state SINGULARITY_WORKSPACE=$SBX/workspace \
  BRAMA_BASE_URL=http://127.0.0.1:9 BRAMA_HMAC_SECRET_FILE=$SBX/brama.hmac \
  LAS_COMMAND=/usr/bin/python3 LAS_MCP_ENTRYPOINT=$SBX/las/stub-las.py \
  LAS_RELEASE_MANIFEST_FILE=$SBX/las/release.manifest.json \
  LAS_RELEASE_MANIFEST_SIGNATURE_FILE=$SBX/las/release.manifest.sig \
  LAS_RELEASE_TRUST_STORE_FILE=$SBX/las/release.trust \
  LAS_RELEASE_WATERMARK_FILE=$SBX/las/release.watermark \
  SINGULARITY_REQUIRED_SURFACES=
```

## The boot gates, in refusal order

Each gate was triggered on purpose; every sentence below is the binary's
own stderr, with the exit code.

Missing identity (clap refuses before any code runs):

```
$ env -u SINGULARITY_AGENT_ID singularity once
error: the following required arguments were not provided:
  --agent-id <AGENT_ID>
…
exit=2
```

A malformed digest:

```
$ SINGULARITY_WORKLOAD_PUBLIC_KEY=not-hex singularity once
singularity: configuration: workload public key must be 64 lowercase hexadecimal characters
exit=2
```

A secret file that others could read:

```
$ chmod 644 $SBX/brama.hmac; singularity once
singularity: secret file: /tmp/singularity-walkthrough.iWnxIW/brama.hmac must not be group/world accessible
exit=2
```

A missing Las entrypoint, and a relative release-pinning path:

```
$ LAS_MCP_ENTRYPOINT=$SBX/las/missing.py singularity once
singularity: configuration: LAS entrypoint not found: /tmp/singularity-walkthrough.iWnxIW/las/missing.py
exit=2

$ LAS_RELEASE_MANIFEST_FILE=las/release.manifest.json singularity once
singularity: configuration: LAS release manifest must be an absolute regular file
exit=2
```

## The catalogue without a fleet

With the stub Las advertising zero tools and no Most credential, `tools`
prints exactly the compiled-in being tools — this is the floor every being
stands on:

```
$ singularity tools --format table
singularity_memory_remember	Persist a memory owned by this digital being
singularity_memory_recall	Recall persistent memories containing a query
singularity_self_set_prompt	Replace this being's persistent system prompt
singularity_self_add_rule	Add a persistent self-imposed rule
singularity_self_add_learning	Record a persistent learning that changes future decisions
singularity_self_switch_model	Switch future cognition calls to another available Brama model
singularity_spawn_child	Create and start a child digital being with separate state
singularity_file_read	Read one UTF-8 file inside the configured workspace
singularity_file_write	Atomically create or replace one UTF-8 file inside the configured workspace
```

Requiring a surface the stub cannot offer proves the existence guarantee:

```
$ SINGULARITY_REQUIRED_SURFACES=finance singularity once
singularity: mcp: required Las surface unavailable: finance
exit=3
```

## Where the credential boundary bites

`doctor` checks Brama first, and with nothing listening the answer is the
transport error, class-stable at exit 3:

```
$ singularity doctor
singularity: brama: error sending request for url (http://127.0.0.1:9/health)
exit=3
```

`once` gets further: the being is **created** — state directory, identity,
mind, journal — Las is spawned and shut down cleanly, and only the first
cognition call fails:

```
$ singularity once
singularity: brama: error sending request for url (http://127.0.0.1:9/v1/chat/completions)
exit=3

$ ls -la $SBX/state
-rw------- 1 … activity.jsonl
-rw------- 1 … state.json
```

This is the documented crash discipline visible from outside: everything up
to the failed round is durable. What is in those two files is
[walkthrough-state](walkthrough-state.md).

## The resume gates

```
$ singularity once            # state now exists
singularity: state: state already exists at /tmp/singularity-walkthrough.iWnxIW/state/state.json; use --resume or a new directory
exit=4

$ SINGULARITY_ROLE=other-role singularity once --resume
singularity: state: resume identity does not match configuration
exit=4

$ singularity once --resume   # same identity: accepted, fails again only at Brama
singularity: brama: error sending request for url (http://127.0.0.1:9/v1/chat/completions)
exit=3

$ SINGULARITY_STATE_DIR=$SBX/state-none singularity once --resume
singularity: state: resume requested but no state exists
exit=4
```

Note the second `--resume` run: the cycle counter in `state.json` advanced
to 2 — the resumed being is the same being, one failed cycle older.

## What a credentialed deployment adds

From here, a real deployment differs in exactly three ways: a reachable
Brama with a real HMAC secret (cognition and `doctor`'s model check), a Las
checkout with signed release pinning (the dynamic catalogue), and
optionally a Most token (the three direct messaging tools). The path for
obtaining the first and third without long-lived secrets is
[bootstrap](bootstrap.md); the finance surface has its own sandboxed
walkthrough in [walkthrough-finance](walkthrough-finance.md).
