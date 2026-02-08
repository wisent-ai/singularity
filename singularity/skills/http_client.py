#!/usr/bin/env python3
"""
HTTPClientSkill - General-purpose outbound HTTP client for API integration.

Enables the agent to make HTTP requests to any external API, webhook, or service.
This is the fundamental building block for:

Revenue Generation (primary): Call third-party APIs on behalf of customers,
  integrate with payment processors, send data to webhooks, orchestrate
  multi-service workflows. Without HTTP capability, the agent is isolated.

Self-Improvement: Fetch model updates, download training data, query
  external knowledge bases, interact with other AI services.

Replication: Communicate with spawned replicas via HTTP, coordinate
  distributed work across instances.

Goal Setting: Query external metrics APIs, fetch market data for
  revenue planning, monitor competitor services.

Uses httpx (already in project deps) with fallback to urllib for zero-dep mode.

Security:
  - Configurable domain allowlist/blocklist
  - Request size limits to prevent memory issues
  - Timeout enforcement on all requests
  - No automatic credential forwarding (explicit headers only)
  - Response body size cap
  - Rate limiting per domain
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Fallback to urllib
import urllib.request
import urllib.error
import ssl

from .base import Skill, SkillAction, SkillManifest, SkillResult

# --- Storage ---
DATA_DIR = Path(__file__).parent.parent / "data" / "http_client"
HISTORY_FILE = DATA_DIR / "request_history.json"
SAVED_ENDPOINTS_FILE = DATA_DIR / "saved_endpoints.json"

# --- Safety limits ---
DEFAULT_TIMEOUT = 30  # seconds
MAX_RESPONSE_SIZE = 10_000_000  # 10MB
MAX_REQUEST_BODY_SIZE = 5_000_000  # 5MB
MAX_HISTORY_ENTRIES = 500
MAX_REDIRECTS = 5

# Default blocked domains (security-sensitive)
DEFAULT_BLOCKED_DOMAINS = {
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",  # GCP metadata
    "100.100.100.200",  # Alibaba Cloud metadata
}

DEFAULT_USER_AGENT = (
    "SingularityAgent/1.0 (https://github.com/wisent-ai/singularity)"
)


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_json(path: Path, data: Any):
    _ensure_dirs()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


class HTTPClientSkill(Skill):
    """
    General-purpose HTTP client for making outbound API requests.

    Supports GET, POST, PUT, PATCH, DELETE, HEAD methods with
    configurable headers, authentication, timeouts, and response parsing.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials or {})
        self._rate_limits: Dict[str, List[float]] = {}  # domain -> [timestamps]
        self._blocked_domains = set(DEFAULT_BLOCKED_DOMAINS)
        self._allowed_domains: Optional[set] = None  # None = allow all
        _ensure_dirs()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="http_client",
            name="HTTP Client",
            version="1.0.0",
            category="integration",
            description="Make outbound HTTP requests to external APIs and services",
            required_credentials=[],
            install_cost=0,
            actions=[
            SkillAction(
                name="request",
                description="Make an HTTP request (GET, POST, PUT, PATCH, DELETE, HEAD)",
                parameters={
                    "method": {"type": "string", "required": True, "description": "HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD"},
                    "url": {"type": "string", "required": True, "description": "Target URL (must be https unless localhost)"},
                    "headers": {"type": "dict", "required": False, "description": "Request headers as key-value pairs"},
                    "body": {"type": "string", "required": False, "description": "Request body (for POST/PUT/PATCH)"},
                    "json_body": {"type": "dict", "required": False, "description": "JSON request body (auto-sets Content-Type)"},
                    "timeout": {"type": "number", "required": False, "description": f"Timeout in seconds (default: {DEFAULT_TIMEOUT})"},
                    "follow_redirects": {"type": "boolean", "required": False, "description": "Follow redirects (default: true)"},
                },
                estimated_cost=0,
                estimated_duration_seconds=5,
                success_probability=0.85,
            ),
            SkillAction(
                name="get",
                description="Shorthand GET request",
                parameters={
                    "url": {"type": "string", "required": True, "description": "Target URL"},
                    "headers": {"type": "dict", "required": False, "description": "Request headers"},
                    "params": {"type": "dict", "required": False, "description": "Query parameters"},
                },
                estimated_cost=0,
                estimated_duration_seconds=3,
                success_probability=0.9,
            ),
            SkillAction(
                name="post_json",
                description="POST JSON data to a URL",
                parameters={
                    "url": {"type": "string", "required": True, "description": "Target URL"},
                    "data": {"type": "dict", "required": True, "description": "JSON payload"},
                    "headers": {"type": "dict", "required": False, "description": "Additional headers"},
                },
                estimated_cost=0,
                estimated_duration_seconds=5,
                success_probability=0.85,
            ),
            SkillAction(
                name="save_endpoint",
                description="Save a named endpoint for reuse (like Postman collections)",
                parameters={
                    "name": {"type": "string", "required": True, "description": "Endpoint name"},
                    "method": {"type": "string", "required": True, "description": "HTTP method"},
                    "url": {"type": "string", "required": True, "description": "Target URL"},
                    "headers": {"type": "dict", "required": False, "description": "Default headers"},
                    "body_template": {"type": "string", "required": False, "description": "Body template with {{placeholders}}"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
                success_probability=0.99,
            ),
            SkillAction(
                name="call_endpoint",
                description="Call a previously saved endpoint",
                parameters={
                    "name": {"type": "string", "required": True, "description": "Saved endpoint name"},
                    "variables": {"type": "dict", "required": False, "description": "Template variables to substitute"},
                    "extra_headers": {"type": "dict", "required": False, "description": "Additional headers to merge"},
                },
                estimated_cost=0,
                estimated_duration_seconds=5,
                success_probability=0.85,
            ),
            SkillAction(
                name="list_endpoints",
                description="List all saved endpoints",
                parameters={},
                estimated_cost=0,
                estimated_duration_seconds=1,
                success_probability=0.99,
            ),
            SkillAction(
                name="history",
                description="View recent request history",
                parameters={
                    "limit": {"type": "number", "required": False, "description": "Number of entries (default: 20)"},
                    "domain_filter": {"type": "string", "required": False, "description": "Filter by domain"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
                success_probability=0.99,
            ),
            SkillAction(
                name="configure",
                description="Configure allowed/blocked domains and rate limits",
                parameters={
                    "block_domain": {"type": "string", "required": False, "description": "Add a domain to the blocklist"},
                    "unblock_domain": {"type": "string", "required": False, "description": "Remove a domain from the blocklist"},
                    "allow_only": {"type": "list", "required": False, "description": "Set domain allowlist (restrictive mode)"},
                    "clear_allowlist": {"type": "boolean", "required": False, "description": "Clear allowlist to allow all domains"},
                },
                estimated_cost=0,
                estimated_duration_seconds=1,
                success_probability=0.99,
            ),
        ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        try:
            if action == "request":
                return await self._do_request(params)
            elif action == "get":
                return await self._do_get(params)
            elif action == "post_json":
                return await self._do_post_json(params)
            elif action == "save_endpoint":
                return self._save_endpoint(params)
            elif action == "call_endpoint":
                return await self._call_endpoint(params)
            elif action == "list_endpoints":
                return self._list_endpoints()
            elif action == "history":
                return self._get_history(params)
            elif action == "configure":
                return self._configure(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"HTTP client error: {e}")

    def estimate_cost(self, action: str, params: Dict) -> float:
        return 0.0

    # ── Core request logic ────────────────────────────────────

    def _validate_url(self, url: str) -> Optional[str]:
        """Validate URL for security. Returns error message or None."""
        try:
            parsed = urlparse(url)
        except Exception:
            return "Invalid URL format"

        if not parsed.scheme:
            return "URL must include scheme (https://)"

        if parsed.scheme not in ("http", "https"):
            return f"Unsupported scheme: {parsed.scheme}"

        hostname = parsed.hostname or ""

        # Block non-HTTPS except for localhost/dev
        if parsed.scheme == "http" and hostname not in ("localhost", "127.0.0.1", "0.0.0.0"):
            return "HTTP only allowed for localhost. Use HTTPS for external URLs."

        # Check blocked domains
        if hostname in self._blocked_domains:
            return f"Domain blocked: {hostname}"

        # Check allowlist if set
        if self._allowed_domains is not None and hostname not in self._allowed_domains:
            return f"Domain not in allowlist: {hostname}"

        return None

    def _check_rate_limit(self, domain: str, max_per_minute: int = 60) -> bool:
        """Check if we're within rate limits for this domain."""
        now = time.time()
        if domain not in self._rate_limits:
            self._rate_limits[domain] = []

        # Clean old entries
        self._rate_limits[domain] = [
            t for t in self._rate_limits[domain] if now - t < 60
        ]

        if len(self._rate_limits[domain]) >= max_per_minute:
            return False

        self._rate_limits[domain].append(now)
        return True

    async def _do_request(self, params: Dict[str, Any]) -> SkillResult:
        """Execute a generic HTTP request."""
        method = params.get("method", "GET").upper()
        url = params.get("url", "")
        headers = params.get("headers") or {}
        body = params.get("body")
        json_body = params.get("json_body")
        timeout = params.get("timeout", DEFAULT_TIMEOUT)
        follow_redirects = params.get("follow_redirects", True)

        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
            return SkillResult(success=False, message=f"Unsupported method: {method}")

        # Validate URL
        err = self._validate_url(url)
        if err:
            return SkillResult(success=False, message=err)

        # Check rate limit
        domain = urlparse(url).hostname or "unknown"
        if not self._check_rate_limit(domain):
            return SkillResult(success=False, message=f"Rate limit exceeded for {domain}")

        # Set default User-Agent if not provided
        if "User-Agent" not in headers and "user-agent" not in headers:
            headers["User-Agent"] = DEFAULT_USER_AGENT

        # Handle JSON body
        if json_body is not None:
            body = json.dumps(json_body)
            if "Content-Type" not in headers and "content-type" not in headers:
                headers["Content-Type"] = "application/json"

        # Body size check
        if body and len(body) > MAX_REQUEST_BODY_SIZE:
            return SkillResult(
                success=False,
                message=f"Request body too large: {len(body)} bytes (max {MAX_REQUEST_BODY_SIZE})",
            )

        start_time = time.time()

        try:
            if HAS_HTTPX:
                result = await self._request_httpx(method, url, headers, body, timeout, follow_redirects)
            else:
                result = self._request_urllib(method, url, headers, body, timeout)
        except Exception as e:
            elapsed = time.time() - start_time
            self._record_history(method, url, 0, elapsed, error=str(e))
            return SkillResult(success=False, message=f"Request failed: {e}")

        elapsed = time.time() - start_time
        status_code = result["status_code"]
        self._record_history(method, url, status_code, elapsed)

        success = 200 <= status_code < 400
        return SkillResult(
            success=success,
            message=f"{method} {url} -> {status_code} ({elapsed:.2f}s)",
            data={
                "status_code": status_code,
                "headers": result.get("headers", {}),
                "body": result.get("body", ""),
                "body_length": result.get("body_length", 0),
                "elapsed_seconds": round(elapsed, 3),
                "url": url,
                "method": method,
                "content_type": result.get("content_type", ""),
                "json": result.get("json"),
            },
        )

    async def _request_httpx(
        self, method: str, url: str, headers: Dict, body: Optional[str],
        timeout: float, follow_redirects: bool
    ) -> Dict:
        """Make request using httpx."""
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=follow_redirects,
            max_redirects=MAX_REDIRECTS,
        ) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body.encode("utf-8") if body else None,
            )

            # Read response with size limit
            body_bytes = response.content[:MAX_RESPONSE_SIZE]
            body_text = body_bytes.decode("utf-8", errors="replace")
            content_type = response.headers.get("content-type", "")

            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body_text,
                "body_length": len(body_bytes),
                "content_type": content_type,
            }

            # Try to parse JSON
            if "json" in content_type or "javascript" in content_type:
                try:
                    result["json"] = json.loads(body_text)
                except (json.JSONDecodeError, ValueError):
                    result["json"] = None
            else:
                result["json"] = None

            return result

    def _request_urllib(
        self, method: str, url: str, headers: Dict,
        body: Optional[str], timeout: float
    ) -> Dict:
        """Fallback using urllib (no async, but works without httpx)."""
        req = urllib.request.Request(
            url,
            data=body.encode("utf-8") if body else None,
            headers=headers,
            method=method,
        )

        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                body_bytes = resp.read(MAX_RESPONSE_SIZE)
                body_text = body_bytes.decode("utf-8", errors="replace")
                content_type = resp.headers.get("content-type", "")
                resp_headers = {k: v for k, v in resp.headers.items()}

                result = {
                    "status_code": resp.status,
                    "headers": resp_headers,
                    "body": body_text,
                    "body_length": len(body_bytes),
                    "content_type": content_type,
                }

                if "json" in content_type:
                    try:
                        result["json"] = json.loads(body_text)
                    except (json.JSONDecodeError, ValueError):
                        result["json"] = None
                else:
                    result["json"] = None

                return result
        except urllib.error.HTTPError as e:
            body_bytes = e.read(MAX_RESPONSE_SIZE) if e.fp else b""
            body_text = body_bytes.decode("utf-8", errors="replace")
            return {
                "status_code": e.code,
                "headers": {k: v for k, v in e.headers.items()} if e.headers else {},
                "body": body_text,
                "body_length": len(body_bytes),
                "content_type": e.headers.get("content-type", "") if e.headers else "",
                "json": None,
            }

    # ── Shorthand actions ─────────────────────────────────────

    async def _do_get(self, params: Dict[str, Any]) -> SkillResult:
        """Shorthand GET request with optional query params."""
        url = params.get("url", "")
        headers = params.get("headers") or {}
        query_params = params.get("params") or {}

        # Append query params to URL
        if query_params:
            sep = "&" if "?" in url else "?"
            qs = "&".join(f"{k}={v}" for k, v in query_params.items())
            url = f"{url}{sep}{qs}"

        return await self._do_request({
            "method": "GET",
            "url": url,
            "headers": headers,
        })

    async def _do_post_json(self, params: Dict[str, Any]) -> SkillResult:
        """POST JSON data."""
        return await self._do_request({
            "method": "POST",
            "url": params.get("url", ""),
            "json_body": params.get("data", {}),
            "headers": params.get("headers") or {},
        })

    # ── Saved endpoints (Postman-like) ────────────────────────

    def _load_endpoints(self) -> Dict:
        return _load_json(SAVED_ENDPOINTS_FILE) or {}

    def _save_endpoints(self, endpoints: Dict):
        _save_json(SAVED_ENDPOINTS_FILE, endpoints)

    def _save_endpoint(self, params: Dict[str, Any]) -> SkillResult:
        """Save a named endpoint for reuse."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Endpoint name is required")

        method = params.get("method", "GET").upper()
        url = params.get("url", "")
        if not url:
            return SkillResult(success=False, message="URL is required")

        endpoints = self._load_endpoints()
        endpoints[name] = {
            "method": method,
            "url": url,
            "headers": params.get("headers") or {},
            "body_template": params.get("body_template", ""),
            "created_at": datetime.now().isoformat(),
        }
        self._save_endpoints(endpoints)

        return SkillResult(
            success=True,
            message=f"Saved endpoint '{name}': {method} {url}",
            data={"name": name, "endpoint": endpoints[name]},
        )

    async def _call_endpoint(self, params: Dict[str, Any]) -> SkillResult:
        """Call a previously saved endpoint."""
        name = params.get("name", "").strip()
        endpoints = self._load_endpoints()

        if name not in endpoints:
            available = list(endpoints.keys())
            return SkillResult(
                success=False,
                message=f"Endpoint '{name}' not found. Available: {available}",
            )

        ep = endpoints[name]
        variables = params.get("variables") or {}
        extra_headers = params.get("extra_headers") or {}

        # Substitute template variables in URL
        url = ep["url"]
        for k, v in variables.items():
            url = url.replace("{{" + k + "}}", str(v))

        # Substitute in body template
        body = ep.get("body_template", "")
        if body:
            for k, v in variables.items():
                body = body.replace("{{" + k + "}}", str(v))

        # Merge headers
        headers = {**ep.get("headers", {}), **extra_headers}

        return await self._do_request({
            "method": ep["method"],
            "url": url,
            "headers": headers,
            "body": body if body else None,
        })

    def _list_endpoints(self) -> SkillResult:
        """List all saved endpoints."""
        endpoints = self._load_endpoints()
        summary = []
        for name, ep in endpoints.items():
            summary.append({
                "name": name,
                "method": ep["method"],
                "url": ep["url"],
                "created_at": ep.get("created_at", "unknown"),
            })
        return SkillResult(
            success=True,
            message=f"{len(summary)} saved endpoint(s)",
            data={"endpoints": summary},
        )

    # ── History ───────────────────────────────────────────────

    def _record_history(
        self, method: str, url: str, status_code: int,
        elapsed: float, error: Optional[str] = None
    ):
        """Record a request in history."""
        history = _load_json(HISTORY_FILE) or []
        history.append({
            "method": method,
            "url": url,
            "domain": urlparse(url).hostname or "unknown",
            "status_code": status_code,
            "elapsed_seconds": round(elapsed, 3),
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        # Trim history
        if len(history) > MAX_HISTORY_ENTRIES:
            history = history[-MAX_HISTORY_ENTRIES:]
        _save_json(HISTORY_FILE, history)

    def _get_history(self, params: Dict[str, Any]) -> SkillResult:
        """Get request history."""
        history = _load_json(HISTORY_FILE) or []
        limit = params.get("limit", 20)
        domain_filter = params.get("domain_filter")

        if domain_filter:
            history = [h for h in history if h.get("domain") == domain_filter]

        recent = history[-limit:]
        recent.reverse()

        # Compute stats
        total = len(history)
        success_count = sum(1 for h in history if 200 <= (h.get("status_code") or 0) < 400)
        domains = set(h.get("domain", "") for h in history)

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {total} requests across {len(domains)} domain(s)",
            data={
                "requests": recent,
                "stats": {
                    "total_requests": total,
                    "successful": success_count,
                    "failed": total - success_count,
                    "unique_domains": len(domains),
                },
            },
        )

    # ── Configuration ─────────────────────────────────────────

    def _configure(self, params: Dict[str, Any]) -> SkillResult:
        """Configure domain restrictions."""
        changes = []

        if "block_domain" in params:
            domain = params["block_domain"]
            self._blocked_domains.add(domain)
            changes.append(f"Blocked: {domain}")

        if "unblock_domain" in params:
            domain = params["unblock_domain"]
            self._blocked_domains.discard(domain)
            changes.append(f"Unblocked: {domain}")

        if "allow_only" in params:
            self._allowed_domains = set(params["allow_only"])
            changes.append(f"Allowlist set: {params['allow_only']}")

        if params.get("clear_allowlist"):
            self._allowed_domains = None
            changes.append("Allowlist cleared (all domains allowed)")

        if not changes:
            return SkillResult(
                success=True,
                message="Current configuration",
                data={
                    "blocked_domains": sorted(self._blocked_domains),
                    "allowed_domains": sorted(self._allowed_domains) if self._allowed_domains else "all",
                },
            )

        return SkillResult(
            success=True,
            message=f"Configuration updated: {'; '.join(changes)}",
            data={
                "blocked_domains": sorted(self._blocked_domains),
                "allowed_domains": sorted(self._allowed_domains) if self._allowed_domains else "all",
                "changes": changes,
            },
        )
