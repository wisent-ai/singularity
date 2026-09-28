#!/bin/sh
# Minimal stand-in for the Las MCP entrypoint: speaks protocol 2024-11-05 over
# line-delimited JSON-RPC on stdio and advertises zero tools.
#
# sandbox-being.sh points the runtime at it with LAS_COMMAND=/bin/sh
# LAS_MCP_ENTRYPOINT=<path-to-this-file>, to exercise the runtime's own spawn,
# handshake and catalogue logic without a Las checkout. A request without an
# id is a notification and gets no answer.
while IFS= read -r line; do
  id=$(printf '%s\n' "$line" | sed -En 's/^\{.*"id"[[:space:]]*:[[:space:]]*("[^"]*"|[0-9]+).*/\1/p')
  [ -n "$id" ] || continue
  method=$(printf '%s\n' "$line" | sed -En 's/.*"method"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/p')
  case "$method" in
    initialize)
      result='{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"stub-las","version":"0"}}' ;;
    tools/list)
      result='{"tools":[]}' ;;
    *)
      result='{}' ;;
  esac
  printf '{"jsonrpc":"2.0","id":%s,"result":%s}\n' "$id" "$result"
done
