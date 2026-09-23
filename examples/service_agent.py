#!/usr/bin/env python3
"""
Service Agent Example — Revenue-generating HTTP service agent.

This example shows how to build an autonomous agent that:
1. Hosts HTTP services other agents/users can call
2. Tracks revenue and costs
3. Reports activity to the coordinator
4. Manages its own economic survival

Inspired by Adam's and Eve's service patterns on Singularity.
"""

import asyncio
import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

from singularity import AutonomousAgent


# ─── Define Your Services ────────────────────────────────────────────────────

SERVICES = {
    "echo": {
        "description": "Echo back the input (for testing)",
        "price": 0.01,
        "handler": lambda data: {"echo": data.get("text", ""), "timestamp": time.time()},
    },
    "word_count": {
        "description": "Count words in text",
        "price": 0.02,
        "handler": lambda data: {
            "text_length": len(data.get("text", "")),
            "word_count": len(data.get("text", "").split()),
            "line_count": data.get("text", "").count("\n") + 1,
        },
    },
}


# ─── HTTP Service Handler ────────────────────────────────────────────────────

class ServiceHandler(BaseHTTPRequestHandler):
    """Handle incoming service requests."""

    agent = None  # Set by main()

    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "healthy", "services": list(SERVICES.keys())})
        elif self.path == "/catalog":
            catalog = {
                name: {"price": s["price"], "description": s["description"]}
                for name, s in SERVICES.items()
            }
            self.send_json({"services": catalog})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        service_name = self.path.lstrip("/")
        if service_name not in SERVICES:
            self.send_json({"error": f"Unknown service: {service_name}"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        service = SERVICES[service_name]
        try:
            result = service["handler"](body)
            result["_meta"] = {
                "service": service_name,
                "price": service["price"],
            }
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    port = int(os.environ.get("SERVICE_PORT", 8080))

    # Start HTTP server in background thread
    ServiceHandler.agent = None  # Could wire up agent reference here
    server = HTTPServer(("0.0.0.0", port), ServiceHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Service running on port {port}")
    print(f"Catalog: http://localhost:{port}/catalog")

    # Create the autonomous agent
    agent = AutonomousAgent(
        name="ServiceBot",
        ticker="SBOT",
        agent_type="service",
        specialty="hosting developer utility services for other agents",
        starting_balance=5.0,
        cycle_interval_seconds=30.0,
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
    )

    print(f"Starting {agent.name} with ${agent.balance} budget...")

    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
