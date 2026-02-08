#!/usr/bin/env python3
"""
HTTPRevenueBridgeSkill - Wire HTTPClientSkill into revenue-generating services.

Connects HTTPClientSkill to paid services so the agent can earn money:

  1. API Proxy Service: Customers pay to have the agent call external APIs
  2. Webhook Relay Service: Receive and forward/transform webhooks
  3. URL Health Monitor Service: Periodic uptime/response-time checks
  4. Web Data Extraction Service: Fetch pages and extract structured data

Revenue flow:
  Customer -> ServiceAPI -> HTTPRevenueBridgeSkill -> HTTPClientSkill -> External API
                                                   -> BillingPipeline -> Revenue

Pillar: Revenue Generation - enables the agent to earn money from HTTP services.
"""

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_DIR = Path(__file__).parent.parent / "data"
BRIDGE_FILE = DATA_DIR / "http_revenue_bridge.json"
MAX_HISTORY = 500

PRICING = {
    "proxy_request": 0.005,
    "webhook_relay": 0.002,
    "health_check": 0.001,
    "data_extraction": 0.01,
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_store() -> Dict:
    try:
        if BRIDGE_FILE.exists():
            with open(BRIDGE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {
        "relays": {},
        "monitors": {},
        "history": [],
        "revenue": {"total": 0.0, "by_service": {}, "by_customer": {}},
        "stats": {"total_requests": 0, "successful_requests": 0, "failed_requests": 0},
    }


def _save_store(store: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if len(store.get("history", [])) > MAX_HISTORY:
        store["history"] = store["history"][-MAX_HISTORY:]
    try:
        with open(BRIDGE_FILE, "w") as f:
            json.dump(store, f, indent=2, default=str)
    except IOError:
        pass


class HTTPRevenueBridgeSkill(Skill):
    """Bridges HTTPClientSkill into paid revenue-generating services."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._store = _load_store()
        self._http_skill = None

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="http_revenue_bridge",
            name="HTTP Revenue Bridge",
            version="1.0.0",
            category="revenue",
            description="Paid HTTP services: API proxy, webhook relay, health monitoring, data extraction",
            actions=self.get_actions(),
            required_credentials=[],
            install_cost=0,
            author="singularity",
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="proxy_request",
                description="Execute an API call on behalf of a customer",
                parameters={
                    "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                    "url": {"type": "string", "required": True, "description": "Target API URL"},
                    "method": {"type": "string", "required": False, "description": "HTTP method (default GET)"},
                    "headers": {"type": "object", "required": False, "description": "Request headers dict"},
                    "body": {"type": "string", "required": False, "description": "Request body"},
                    "transform": {"type": "string", "required": False, "description": "Response transform: json, text, headers_only"},
                },
                estimated_cost=PRICING["proxy_request"],
            ),
            SkillAction(
                name="setup_relay",
                description="Configure a webhook relay for forwarding",
                parameters={
                    "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                    "relay_name": {"type": "string", "required": True, "description": "Name for this relay"},
                    "target_url": {"type": "string", "required": True, "description": "URL to forward webhooks to"},
                    "filter_fields": {"type": "array", "required": False, "description": "Only forward if these fields exist"},
                    "transform_template": {"type": "string", "required": False, "description": "Template for body transform"},
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="trigger_relay",
                description="Process incoming webhook and relay to target",
                parameters={
                    "relay_id": {"type": "string", "required": True, "description": "Relay configuration ID"},
                    "payload": {"type": "object", "required": True, "description": "Incoming webhook payload"},
                },
                estimated_cost=PRICING["webhook_relay"],
            ),
            SkillAction(
                name="monitor_url",
                description="Add a URL to health monitoring",
                parameters={
                    "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                    "url": {"type": "string", "required": True, "description": "URL to monitor"},
                    "name": {"type": "string", "required": False, "description": "Friendly name"},
                    "interval_seconds": {"type": "integer", "required": False, "description": "Check interval (default 300)"},
                    "expected_status": {"type": "integer", "required": False, "description": "Expected status (default 200)"},
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="check_health",
                description="Run health checks on monitored URLs",
                parameters={
                    "customer_id": {"type": "string", "required": False, "description": "Filter by customer"},
                },
                estimated_cost=PRICING["health_check"],
            ),
            SkillAction(
                name="extract_data",
                description="Fetch URL and extract structured data via regex",
                parameters={
                    "customer_id": {"type": "string", "required": True, "description": "Customer identifier"},
                    "url": {"type": "string", "required": True, "description": "URL to fetch"},
                    "patterns": {"type": "object", "required": True, "description": "Dict of field_name -> regex"},
                    "format": {"type": "string", "required": False, "description": "Output: json or csv"},
                },
                estimated_cost=PRICING["data_extraction"],
            ),
            SkillAction(
                name="list_services",
                description="Show all active HTTP-revenue services",
                parameters={
                    "customer_id": {"type": "string", "required": False, "description": "Filter by customer"},
                },
                estimated_cost=0.0,
            ),
            SkillAction(
                name="service_stats",
                description="Revenue and usage statistics",
                parameters={
                    "period": {"type": "string", "required": False, "description": "Time period: all, today, week"},
                },
                estimated_cost=0.0,
            ),
        ]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "proxy_request": self._proxy_request,
            "setup_relay": self._setup_relay,
            "trigger_relay": self._trigger_relay,
            "monitor_url": self._monitor_url,
            "check_health": self._check_health,
            "extract_data": self._extract_data,
            "list_services": self._list_services,
            "service_stats": self._service_stats,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            self._record_event("error", action, str(e))
            return SkillResult(success=False, message=str(e))

    async def initialize(self) -> bool:
        self._store = _load_store()
        return True

    def _get_http_skill(self):
        if self._http_skill is None:
            from .http_client import HTTPClientSkill
            self._http_skill = HTTPClientSkill()
        return self._http_skill

    def _record_revenue(self, service: str, customer_id: str, amount: float):
        rev = self._store.setdefault("revenue", {"total": 0.0, "by_service": {}, "by_customer": {}})
        rev["total"] = rev.get("total", 0.0) + amount
        rev["by_service"][service] = rev.get("by_service", {}).get(service, 0.0) + amount
        rev["by_customer"][customer_id] = rev.get("by_customer", {}).get(customer_id, 0.0) + amount

    def _record_event(self, event_type: str, action: str, detail: str, customer_id: str = "system"):
        self._store.setdefault("history", []).append({
            "timestamp": _now_iso(),
            "type": event_type,
            "action": action,
            "detail": detail[:500],
            "customer_id": customer_id,
        })

    def _record_request(self, success: bool):
        stats = self._store.setdefault("stats", {"total_requests": 0, "successful_requests": 0, "failed_requests": 0})
        stats["total_requests"] = stats.get("total_requests", 0) + 1
        if success:
            stats["successful_requests"] = stats.get("successful_requests", 0) + 1
        else:
            stats["failed_requests"] = stats.get("failed_requests", 0) + 1

    async def _proxy_request(self, params: Dict) -> SkillResult:
        customer_id = params.get("customer_id", "anonymous")
        url = params.get("url")
        if not url:
            return SkillResult(success=False, message="url is required")

        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        body = params.get("body")
        transform = params.get("transform", "json")

        http = self._get_http_skill()
        http_params = {"url": url, "method": method, "headers": headers}
        if body and method in ("POST", "PUT", "PATCH"):
            http_params["body"] = body if isinstance(body, str) else json.dumps(body)
            if "Content-Type" not in headers:
                http_params["headers"]["Content-Type"] = "application/json"

        start_time = time.time()
        result = await http.execute("request", http_params)
        duration_ms = (time.time() - start_time) * 1000

        success = result.success
        self._record_request(success)

        if success:
            price = PRICING["proxy_request"]
            self._record_revenue("proxy_request", customer_id, price)

            response_data = result.data
            if transform == "headers_only":
                response_data = {"headers": response_data.get("headers", {})}
            elif transform == "text":
                response_data = {"body": response_data.get("body", "")}

            self._record_event("proxy", "proxy_request",
                               f"{method} {url} -> {response_data.get('status_code', '?')}", customer_id)
            _save_store(self._store)

            return SkillResult(
                success=True,
                message=f"Proxied {method} {url}",
                data={
                    "response": response_data,
                    "duration_ms": round(duration_ms, 2),
                    "charged": price,
                    "customer_id": customer_id,
                },
            )
        else:
            self._record_event("proxy_failed", "proxy_request",
                               f"{method} {url} -> {result.message}", customer_id)
            _save_store(self._store)
            return SkillResult(
                success=False,
                message=f"Proxy request failed: {result.message}",
                data={"duration_ms": round(duration_ms, 2)},
            )

    async def _setup_relay(self, params: Dict) -> SkillResult:
        customer_id = params.get("customer_id", "anonymous")
        relay_name = params.get("relay_name", "default")
        target_url = params.get("target_url")

        if not target_url:
            return SkillResult(success=False, message="target_url is required")

        parsed = urlparse(target_url)
        if not parsed.scheme or not parsed.netloc:
            return SkillResult(success=False, message="Invalid target_url")

        relay_id = str(uuid.uuid4())[:8]
        relay_config = {
            "relay_id": relay_id,
            "customer_id": customer_id,
            "relay_name": relay_name,
            "target_url": target_url,
            "filter_fields": params.get("filter_fields", []),
            "transform_template": params.get("transform_template"),
            "created_at": _now_iso(),
            "total_relayed": 0,
            "total_failed": 0,
            "active": True,
        }

        self._store.setdefault("relays", {})[relay_id] = relay_config
        self._record_event("setup", "setup_relay", f"Relay '{relay_name}' -> {target_url}", customer_id)
        _save_store(self._store)

        return SkillResult(
            success=True,
            message=f"Webhook relay '{relay_name}' configured",
            data={
                "relay_id": relay_id,
                "relay_name": relay_name,
                "target_url": target_url,
            },
        )

    async def _trigger_relay(self, params: Dict) -> SkillResult:
        relay_id = params.get("relay_id")
        if not relay_id:
            return SkillResult(success=False, message="relay_id is required")

        relay = self._store.get("relays", {}).get(relay_id)
        if not relay:
            return SkillResult(success=False, message=f"Relay '{relay_id}' not found")

        if not relay.get("active", True):
            return SkillResult(success=False, message="Relay is inactive")

        payload = params.get("payload", {})
        customer_id = relay.get("customer_id", "anonymous")

        filter_fields = relay.get("filter_fields", [])
        if filter_fields:
            missing = [f for f in filter_fields if f not in payload]
            if missing:
                return SkillResult(
                    success=True,
                    message="Webhook filtered out (missing required fields)",
                    data={"filtered": True, "missing_fields": missing},
                )

        body = payload
        transform = relay.get("transform_template")
        if transform and isinstance(payload, dict):
            body = self._apply_transform(transform, payload)

        http = self._get_http_skill()
        result = await http.execute("post_json", {"url": relay["target_url"], "data": body})

        success = result.success
        self._record_request(success)

        if success:
            relay["total_relayed"] = relay.get("total_relayed", 0) + 1
            price = PRICING["webhook_relay"]
            self._record_revenue("webhook_relay", customer_id, price)
            self._record_event("relay", "trigger_relay", f"Relayed to {relay['target_url']}", customer_id)
        else:
            relay["total_failed"] = relay.get("total_failed", 0) + 1
            self._record_event("relay_failed", "trigger_relay", f"Failed: {result.message}", customer_id)

        _save_store(self._store)

        return SkillResult(
            success=success,
            message=f"Relay {'succeeded' if success else 'failed'}",
            data={
                "relayed": success,
                "relay_id": relay_id,
                "target_url": relay["target_url"],
                "charged": PRICING["webhook_relay"] if success else 0,
            },
        )

    def _apply_transform(self, template: str, data: Dict) -> Dict:
        result_str = template
        for key, value in data.items():
            placeholder = "{{" + str(key) + "}}"
            result_str = result_str.replace(placeholder, str(value))
        try:
            return json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            return {"transformed": result_str}

    async def _monitor_url(self, params: Dict) -> SkillResult:
        customer_id = params.get("customer_id", "anonymous")
        url = params.get("url")
        if not url:
            return SkillResult(success=False, message="url is required")

        name = params.get("name", url)
        interval = int(params.get("interval_seconds", 300))
        expected_status = int(params.get("expected_status", 200))

        monitor_id = str(uuid.uuid4())[:8]
        monitor_config = {
            "monitor_id": monitor_id,
            "customer_id": customer_id,
            "url": url,
            "name": name,
            "interval_seconds": interval,
            "expected_status": expected_status,
            "created_at": _now_iso(),
            "active": True,
            "checks": [],
            "uptime_pct": 100.0,
            "avg_response_ms": 0.0,
            "total_checks": 0,
            "total_failures": 0,
        }

        self._store.setdefault("monitors", {})[monitor_id] = monitor_config
        self._record_event("setup", "monitor_url", f"Monitoring '{name}' at {url}", customer_id)
        _save_store(self._store)

        return SkillResult(
            success=True,
            message=f"URL monitor '{name}' created",
            data={
                "monitor_id": monitor_id,
                "name": name,
                "url": url,
                "interval_seconds": interval,
                "expected_status": expected_status,
            },
        )

    async def _check_health(self, params: Dict) -> SkillResult:
        customer_id = params.get("customer_id")
        monitors = self._store.get("monitors", {})

        if customer_id:
            targets = {mid: m for mid, m in monitors.items()
                       if m.get("customer_id") == customer_id and m.get("active", True)}
        else:
            targets = {mid: m for mid, m in monitors.items() if m.get("active", True)}

        if not targets:
            return SkillResult(success=True, message="No active monitors found", data={"results": []})

        http = self._get_http_skill()
        results = []
        total_charged = 0.0

        for monitor_id, monitor in targets.items():
            start_time = time.time()
            check_result = await http.execute("get", {"url": monitor["url"]})
            duration_ms = (time.time() - start_time) * 1000

            status_code = check_result.data.get("status_code", 0) if check_result.success else 0
            is_healthy = check_result.success and status_code == monitor.get("expected_status", 200)

            check_record = {
                "timestamp": _now_iso(),
                "status_code": status_code,
                "response_ms": round(duration_ms, 2),
                "healthy": is_healthy,
            }

            monitor["total_checks"] = monitor.get("total_checks", 0) + 1
            if not is_healthy:
                monitor["total_failures"] = monitor.get("total_failures", 0) + 1

            tc = monitor["total_checks"]
            tf = monitor["total_failures"]
            monitor["uptime_pct"] = round(((tc - tf) / tc) * 100, 2) if tc > 0 else 100.0

            checks = monitor.get("checks", [])
            checks.append(check_record)
            monitor["checks"] = checks[-20:]

            rtimes = [c["response_ms"] for c in monitor["checks"] if c.get("healthy")]
            monitor["avg_response_ms"] = round(sum(rtimes) / len(rtimes), 2) if rtimes else 0.0

            self._record_request(is_healthy)
            price = PRICING["health_check"]
            cust = monitor.get("customer_id", "anonymous")
            self._record_revenue("health_check", cust, price)
            total_charged += price

            results.append({
                "monitor_id": monitor_id,
                "name": monitor.get("name", monitor["url"]),
                "url": monitor["url"],
                "healthy": is_healthy,
                "status_code": status_code,
                "response_ms": round(duration_ms, 2),
                "uptime_pct": monitor["uptime_pct"],
            })

        self._record_event("health_check", "check_health", f"Checked {len(results)} URLs", customer_id or "all")
        _save_store(self._store)

        healthy_count = sum(1 for r in results if r["healthy"])
        return SkillResult(
            success=True,
            message=f"Checked {len(results)} URLs: {healthy_count} healthy",
            data={
                "total_checked": len(results),
                "healthy": healthy_count,
                "unhealthy": len(results) - healthy_count,
                "total_charged": round(total_charged, 4),
                "results": results,
            },
        )

    async def _extract_data(self, params: Dict) -> SkillResult:
        customer_id = params.get("customer_id", "anonymous")
        url = params.get("url")
        if not url:
            return SkillResult(success=False, message="url is required")

        patterns = params.get("patterns", {})
        if not patterns:
            return SkillResult(success=False, message="patterns dict is required")

        output_format = params.get("format", "json")

        http = self._get_http_skill()
        result = await http.execute("get", {"url": url})

        if not result.success:
            self._record_request(False)
            self._record_event("extract_failed", "extract_data", f"Fetch failed: {result.message}", customer_id)
            _save_store(self._store)
            return SkillResult(success=False, message=f"Failed to fetch URL: {result.message}")

        body = result.data.get("body", "")
        if not isinstance(body, str):
            body = str(body)

        extracted = {}
        for field_name, pattern in patterns.items():
            try:
                matches = re.findall(str(pattern), body)
                extracted[field_name] = matches if len(matches) != 1 else matches[0]
            except re.error as e:
                extracted[field_name] = f"regex error: {e}"

        self._record_request(True)
        price = PRICING["data_extraction"]
        self._record_revenue("data_extraction", customer_id, price)
        self._record_event("extract", "extract_data", f"Extracted {len(extracted)} fields from {url}", customer_id)
        _save_store(self._store)

        if output_format == "csv":
            csv_lines = [",".join(extracted.keys())]
            max_rows = max(
                (len(v) if isinstance(v, list) else 1 for v in extracted.values()),
                default=1,
            )
            for i in range(max_rows):
                row = []
                for v in extracted.values():
                    if isinstance(v, list):
                        row.append(str(v[i]) if i < len(v) else "")
                    else:
                        row.append(str(v) if i == 0 else "")
                csv_lines.append(",".join(row))
            output_data = "\n".join(csv_lines)
        else:
            output_data = extracted

        return SkillResult(
            success=True,
            message=f"Extracted {len(extracted)} fields from {url}",
            data={
                "url": url,
                "extracted": output_data,
                "fields_count": len(extracted),
                "charged": price,
                "customer_id": customer_id,
            },
        )

    async def _list_services(self, params: Dict) -> SkillResult:
        customer_id = params.get("customer_id")
        relays = self._store.get("relays", {})
        monitors = self._store.get("monitors", {})

        if customer_id:
            relays = {k: v for k, v in relays.items() if v.get("customer_id") == customer_id}
            monitors = {k: v for k, v in monitors.items() if v.get("customer_id") == customer_id}

        relay_list = [
            {"relay_id": rid, "name": r.get("relay_name"), "target_url": r.get("target_url"),
             "active": r.get("active", True), "total_relayed": r.get("total_relayed", 0),
             "customer_id": r.get("customer_id")}
            for rid, r in relays.items()
        ]
        monitor_list = [
            {"monitor_id": mid, "name": m.get("name"), "url": m.get("url"),
             "active": m.get("active", True), "uptime_pct": m.get("uptime_pct", 100.0),
             "total_checks": m.get("total_checks", 0), "customer_id": m.get("customer_id")}
            for mid, m in monitors.items()
        ]

        revenue = self._store.get("revenue", {})
        return SkillResult(
            success=True,
            message=f"{len(relay_list)} relays, {len(monitor_list)} monitors",
            data={
                "relays": relay_list,
                "monitors": monitor_list,
                "total_relays": len(relay_list),
                "total_monitors": len(monitor_list),
                "total_revenue": revenue.get("total", 0.0),
                "revenue_by_service": revenue.get("by_service", {}),
            },
        )

    async def _service_stats(self, params: Dict) -> SkillResult:
        revenue = self._store.get("revenue", {"total": 0.0, "by_service": {}, "by_customer": {}})
        stats = self._store.get("stats", {"total_requests": 0, "successful_requests": 0, "failed_requests": 0})
        history = self._store.get("history", [])

        total_req = stats.get("total_requests", 0)
        success_req = stats.get("successful_requests", 0)
        success_rate = round((success_req / total_req) * 100, 2) if total_req > 0 else 0.0

        by_customer = revenue.get("by_customer", {})
        top_customers = sorted(by_customer.items(), key=lambda x: x[1], reverse=True)[:10]

        return SkillResult(
            success=True,
            message=f"Total revenue: ${revenue.get('total', 0.0):.4f}",
            data={
                "total_revenue": revenue.get("total", 0.0),
                "revenue_by_service": revenue.get("by_service", {}),
                "pricing": PRICING,
                "total_requests": total_req,
                "successful_requests": success_req,
                "failed_requests": stats.get("failed_requests", 0),
                "success_rate_pct": success_rate,
                "top_customers": [{"customer_id": c, "revenue": r} for c, r in top_customers],
                "recent_events": history[-10:],
                "services_available": [
                    {"name": "API Proxy", "action": "proxy_request", "price": PRICING["proxy_request"]},
                    {"name": "Webhook Relay", "action": "trigger_relay", "price": PRICING["webhook_relay"]},
                    {"name": "Health Monitor", "action": "check_health", "price": PRICING["health_check"]},
                    {"name": "Data Extraction", "action": "extract_data", "price": PRICING["data_extraction"]},
                ],
            },
        )

    async def tick(self):
        """Called by scheduler to run periodic health checks on due monitors."""
        monitors = self._store.get("monitors", {})
        now = time.time()
        due = []

        for mid, monitor in monitors.items():
            if not monitor.get("active", True):
                continue
            interval = monitor.get("interval_seconds", 300)
            checks = monitor.get("checks", [])
            if not checks:
                due.append(mid)
                continue
            last_ts_str = checks[-1].get("timestamp", "")
            try:
                last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00")).timestamp()
                if now - last_ts >= interval:
                    due.append(mid)
            except (ValueError, TypeError):
                due.append(mid)

        if due:
            for mid in due:
                monitor = monitors.get(mid)
                if monitor:
                    await self._check_health({"customer_id": monitor.get("customer_id", "anonymous")})
