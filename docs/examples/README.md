# Examples

Runnable scripts behind the executed walkthroughs. All of them are
self-contained sandboxes: throwaway `mktemp` directories, generated keys,
no network beyond the local discard port, no credentials, no operator
state, nothing spent.

| Script | What it proves | Needs |
|---|---|---|
| [`sandbox-being.sh`](sandbox-being.sh) | Every boot gate, the compiled-in tool catalogue, being creation, and the resume gates — the transcript behind [walkthrough-first-cycle](../walkthrough-first-cycle.md). | `singularity` binary (`SINGULARITY_BIN`, default: on `PATH`), `python3`. |
| [`stub-las.py`](stub-las.py) | A minimal Las stand-in: MCP `2024-11-05` over stdio, zero tools. Used by `sandbox-being.sh` to exercise the runtime's spawn, handshake, and catalogue logic without a Las checkout. | `python3` (stdlib only). |
| [`finance-lifecycle.py`](finance-lifecycle.py) | One transaction through the whole finance boundary — propose, policy refusals, simulate, approve, timelock, sign, failed dispatch, reconcile, submit, confirm, and the enable-lease kill switch and rollback — the transcript behind [walkthrough-finance](../walkthrough-finance.md). | `singularity-finance-mcp` binary (`SINGULARITY_FINANCE_MCP`, default: `target/release/`), `python3` with the `cryptography` package. |

```bash
cargo build --release
SINGULARITY_BIN=target/release/singularity docs/examples/sandbox-being.sh
python3 docs/examples/finance-lifecycle.py
```

`finance-lifecycle.py` prints its sandbox directory at the end; point the
inspection commands from [walkthrough-state](../walkthrough-state.md) at it.
