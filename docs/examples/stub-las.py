#!/usr/bin/env python3
"""Minimal stand-in for the Las MCP entrypoint: speaks protocol 2024-11-05
over line-delimited JSON-RPC on stdio and advertises zero tools.

Used by walkthrough-first-cycle.md to exercise the runtime's own spawn,
handshake, and catalogue logic without a Las checkout. Point the runtime at
it with LAS_COMMAND=/usr/bin/python3 LAS_MCP_ENTRYPOINT=<path-to-this-file>.
"""
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    if "id" not in request:
        continue  # notification
    method = request.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stub-las", "version": "0"},
        }
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
