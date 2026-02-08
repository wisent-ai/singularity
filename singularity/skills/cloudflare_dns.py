#!/usr/bin/env python3
"""
CloudflareDNSSkill - Automated DNS management via Cloudflare API.

Enables the agent to programmatically manage DNS records for domains
hosted on Cloudflare. This is critical infrastructure for the Revenue
pillar - the agent needs to point domains at deployed services, manage
subdomains for new offerings, and automate the full domain lifecycle.

Capabilities:
- List, create, update, delete DNS records (A, AAAA, CNAME, TXT, MX, etc.)
- List and manage zones (domains)
- Bulk operations for multi-record updates
- Record validation before creation
- DNS propagation status checking
- Integration with deployment skills for auto-wiring

Works with:
- VercelSkill: auto-create CNAME records for Vercel deployments
- ServiceHostingSkill: point domains at hosted services
- DeploymentSkill: DNS setup after deploying replicas
- NamecheapSkill: register domain → auto-configure Cloudflare DNS

Required credentials:
- CLOUDFLARE_API_TOKEN: API token with DNS edit permissions
  (preferred over global API key for security - scoped permissions)

Pillars:
- Revenue: Wire domains to revenue-generating services automatically
- Replication: Give replicas their own subdomains
- Self-Improvement: Track DNS operations for optimization
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_FILE = Path(__file__).parent.parent / "data" / "cloudflare_dns.json"

# Supported DNS record types
RECORD_TYPES = ["A", "AAAA", "CNAME", "TXT", "MX", "NS", "SRV", "CAA", "PTR"]

# TTL values (seconds) - 1 = automatic
TTL_AUTO = 1
TTL_MIN = 60
TTL_MAX = 86400

# Max records per bulk operation
MAX_BULK_RECORDS = 50

# Cloudflare API base
CF_API_BASE = "https://api.cloudflare.com/client/v4"


def _load_data(path: Path = None) -> Dict:
    """Load DNS operation state from disk."""
    p = path or DATA_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "zones_cache": {},
        "operations": [],
        "record_cache": {},
        "config": {
            "default_ttl": TTL_AUTO,
            "default_proxied": True,
            "cache_ttl_seconds": 300,
        },
    }


def _save_data(data: Dict, path: Path = None) -> None:
    """Save DNS operation state to disk."""
    p = path or DATA_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    # Trim operations history
    if len(data.get("operations", [])) > 500:
        data["operations"] = data["operations"][-500:]
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _validate_record_type(record_type: str) -> bool:
    """Validate DNS record type."""
    return record_type.upper() in RECORD_TYPES


def _validate_ttl(ttl: int) -> bool:
    """Validate TTL value."""
    return ttl == TTL_AUTO or (TTL_MIN <= ttl <= TTL_MAX)


def _validate_record_content(record_type: str, content: str) -> Optional[str]:
    """Basic validation of record content. Returns error message or None."""
    record_type = record_type.upper()
    if not content or not content.strip():
        return "Content cannot be empty"

    if record_type == "A":
        parts = content.split(".")
        if len(parts) != 4:
            return f"Invalid IPv4 address: {content}"
        for part in parts:
            try:
                num = int(part)
                if num < 0 or num > 255:
                    return f"Invalid IPv4 octet: {part}"
            except ValueError:
                return f"Invalid IPv4 address: {content}"

    elif record_type == "AAAA":
        # Basic IPv6 validation
        if ":" not in content:
            return f"Invalid IPv6 address: {content}"

    elif record_type == "CNAME":
        if " " in content:
            return f"CNAME target cannot contain spaces: {content}"

    elif record_type == "MX":
        # MX content should just be the mail server hostname
        if " " in content.strip():
            return f"MX content should be hostname only (priority set separately): {content}"

    return None


def _log_operation(data: Dict, action: str, details: Dict) -> None:
    """Log a DNS operation for audit trail."""
    data["operations"].append({
        "id": str(uuid.uuid4())[:8],
        "action": action,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details,
    })


class CloudflareDNSSkill(Skill):
    """
    DNS management via Cloudflare API.

    Provides full CRUD for DNS records, zone management, and bulk operations.
    All operations are logged for audit trail and can be integrated with
    deployment workflows for automatic DNS configuration.
    """

    def __init__(self):
        self._data_file = DATA_FILE

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="cloudflare_dns",
            name="Cloudflare DNS Management",
            version="1.0.0",
            category="infrastructure",
            description="Automated DNS record management via Cloudflare API",
            required_credentials=["CLOUDFLARE_API_TOKEN"],
            install_cost=0,
            actions=[
                SkillAction(
                    name="list_zones",
                    description="List all domains/zones in the Cloudflare account",
                    parameters={
                        "name": {"type": "str", "required": False, "description": "Filter by domain name"},
                        "status": {"type": "str", "required": False, "description": "Filter by status (active, pending, etc.)"},
                    },
                    estimated_cost=0,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="list_records",
                    description="List DNS records for a zone/domain",
                    parameters={
                        "zone_id": {"type": "str", "required": True, "description": "Zone ID"},
                        "record_type": {"type": "str", "required": False, "description": "Filter by record type"},
                        "name": {"type": "str", "required": False, "description": "Filter by record name"},
                    },
                    estimated_cost=0,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="create_record",
                    description="Create a new DNS record",
                    parameters={
                        "zone_id": {"type": "str", "required": True, "description": "Zone ID"},
                        "type": {"type": "str", "required": True, "description": "Record type (A, AAAA, CNAME, TXT, MX, etc.)"},
                        "name": {"type": "str", "required": True, "description": "Record name (e.g., 'api' or 'api.example.com')"},
                        "content": {"type": "str", "required": True, "description": "Record content (IP, hostname, text, etc.)"},
                        "ttl": {"type": "int", "required": False, "description": "TTL in seconds (1=auto, default auto)"},
                        "proxied": {"type": "bool", "required": False, "description": "Enable Cloudflare proxy (default true for A/CNAME)"},
                        "priority": {"type": "int", "required": False, "description": "Priority (required for MX records)"},
                    },
                    estimated_cost=0,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="update_record",
                    description="Update an existing DNS record",
                    parameters={
                        "zone_id": {"type": "str", "required": True, "description": "Zone ID"},
                        "record_id": {"type": "str", "required": True, "description": "Record ID to update"},
                        "type": {"type": "str", "required": False, "description": "New record type"},
                        "name": {"type": "str", "required": False, "description": "New record name"},
                        "content": {"type": "str", "required": False, "description": "New record content"},
                        "ttl": {"type": "int", "required": False, "description": "New TTL"},
                        "proxied": {"type": "bool", "required": False, "description": "New proxy setting"},
                    },
                    estimated_cost=0,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="delete_record",
                    description="Delete a DNS record",
                    parameters={
                        "zone_id": {"type": "str", "required": True, "description": "Zone ID"},
                        "record_id": {"type": "str", "required": True, "description": "Record ID to delete"},
                    },
                    estimated_cost=0,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="bulk_create",
                    description="Create multiple DNS records at once",
                    parameters={
                        "zone_id": {"type": "str", "required": True, "description": "Zone ID"},
                        "records": {"type": "list", "required": True, "description": "List of record dicts [{type, name, content, ttl?, proxied?, priority?}]"},
                    },
                    estimated_cost=0,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="find_record",
                    description="Find a DNS record by name and type",
                    parameters={
                        "zone_id": {"type": "str", "required": True, "description": "Zone ID"},
                        "name": {"type": "str", "required": True, "description": "Record name to find"},
                        "record_type": {"type": "str", "required": False, "description": "Record type to filter"},
                    },
                    estimated_cost=0,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="wire_service",
                    description="High-level: create DNS records to point a subdomain at a service",
                    parameters={
                        "zone_id": {"type": "str", "required": True, "description": "Zone ID"},
                        "subdomain": {"type": "str", "required": True, "description": "Subdomain (e.g., 'api', 'dashboard')"},
                        "target": {"type": "str", "required": True, "description": "Target IP or hostname"},
                        "target_type": {"type": "str", "required": False, "description": "'ip' or 'hostname' (auto-detected if not set)"},
                        "proxied": {"type": "bool", "required": False, "description": "Enable Cloudflare proxy (default true)"},
                    },
                    estimated_cost=0,
                    success_probability=0.9,
                ),
                SkillAction(
                    name="history",
                    description="View DNS operation history",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Number of recent operations to show (default 20)"},
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="configure",
                    description="Update default DNS settings",
                    parameters={
                        "default_ttl": {"type": "int", "required": False, "description": "Default TTL for new records"},
                        "default_proxied": {"type": "bool", "required": False, "description": "Default proxy setting"},
                    },
                    estimated_cost=0,
                    success_probability=1.0,
                ),
            ],
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get Cloudflare API headers using token from context or env."""
        import os
        token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if hasattr(self, "_context") and self._context:
            creds = getattr(self._context, "credentials", {})
            if creds and "CLOUDFLARE_API_TOKEN" in creds:
                token = creds["CLOUDFLARE_API_TOKEN"]
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _has_token(self) -> bool:
        """Check if API token is available."""
        import os
        token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if hasattr(self, "_context") and self._context:
            creds = getattr(self._context, "credentials", {})
            if creds and "CLOUDFLARE_API_TOKEN" in creds:
                token = creds["CLOUDFLARE_API_TOKEN"]
        return bool(token)

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        """Execute a Cloudflare DNS action."""
        actions = {
            "list_zones": self._list_zones,
            "list_records": self._list_records,
            "create_record": self._create_record,
            "update_record": self._update_record,
            "delete_record": self._delete_record,
            "bulk_create": self._bulk_create,
            "find_record": self._find_record,
            "wire_service": self._wire_service,
            "history": self._history,
            "configure": self._configure,
        }
        if action not in actions:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(actions.keys())}",
            )
        return await actions[action](params)

    async def _api_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make a Cloudflare API request. Returns response dict."""
        if not HAS_HTTPX:
            return {"success": False, "errors": [{"message": "httpx not installed"}]}

        if not self._has_token():
            return {"success": False, "errors": [{"message": "CLOUDFLARE_API_TOKEN not set"}]}

        url = f"{CF_API_BASE}{endpoint}"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "POST":
                resp = await client.post(url, headers=headers, json=data)
            elif method == "PUT":
                resp = await client.put(url, headers=headers, json=data)
            elif method == "PATCH":
                resp = await client.patch(url, headers=headers, json=data)
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                return {"success": False, "errors": [{"message": f"Unsupported method: {method}"}]}

            return resp.json()

    async def _list_zones(self, params: Dict) -> SkillResult:
        """List all zones/domains in the account."""
        endpoint = "/zones?"
        filters = []
        if params.get("name"):
            filters.append(f"name={params['name']}")
        if params.get("status"):
            filters.append(f"status={params['status']}")
        filters.append("per_page=50")
        endpoint += "&".join(filters)

        resp = await self._api_request("GET", endpoint)
        if not resp.get("success"):
            errors = resp.get("errors", [{"message": "Unknown error"}])
            return SkillResult(
                success=False,
                message=f"Failed to list zones: {errors[0].get('message', 'Unknown error')}",
            )

        zones = resp.get("result", [])
        data = _load_data(self._data_file)

        # Cache zone info
        for z in zones:
            data["zones_cache"][z["id"]] = {
                "name": z["name"],
                "status": z["status"],
                "cached_at": datetime.utcnow().isoformat(),
            }
        _save_data(data, self._data_file)

        zone_list = [
            {
                "id": z["id"],
                "name": z["name"],
                "status": z["status"],
                "name_servers": z.get("name_servers", []),
                "plan": z.get("plan", {}).get("name", "unknown"),
            }
            for z in zones
        ]

        _log_operation(data, "list_zones", {"count": len(zone_list)})
        _save_data(data, self._data_file)

        return SkillResult(
            success=True,
            message=f"Found {len(zone_list)} zones",
            data={"zones": zone_list, "count": len(zone_list)},
        )

    async def _list_records(self, params: Dict) -> SkillResult:
        """List DNS records for a zone."""
        zone_id = params.get("zone_id")
        if not zone_id:
            return SkillResult(success=False, message="zone_id is required")

        endpoint = f"/zones/{zone_id}/dns_records?"
        filters = ["per_page=100"]
        if params.get("record_type"):
            filters.append(f"type={params['record_type'].upper()}")
        if params.get("name"):
            filters.append(f"name={params['name']}")
        endpoint += "&".join(filters)

        resp = await self._api_request("GET", endpoint)
        if not resp.get("success"):
            errors = resp.get("errors", [{"message": "Unknown error"}])
            return SkillResult(
                success=False,
                message=f"Failed to list records: {errors[0].get('message', 'Unknown error')}",
            )

        records = resp.get("result", [])
        record_list = [
            {
                "id": r["id"],
                "type": r["type"],
                "name": r["name"],
                "content": r["content"],
                "ttl": r["ttl"],
                "proxied": r.get("proxied", False),
                "priority": r.get("priority"),
            }
            for r in records
        ]

        data = _load_data(self._data_file)
        _log_operation(data, "list_records", {"zone_id": zone_id, "count": len(record_list)})
        _save_data(data, self._data_file)

        return SkillResult(
            success=True,
            message=f"Found {len(record_list)} records in zone {zone_id}",
            data={"records": record_list, "count": len(record_list)},
        )

    async def _create_record(self, params: Dict) -> SkillResult:
        """Create a new DNS record."""
        zone_id = params.get("zone_id")
        record_type = params.get("type", "").upper()
        name = params.get("name", "")
        content = params.get("content", "")

        if not zone_id:
            return SkillResult(success=False, message="zone_id is required")
        if not record_type:
            return SkillResult(success=False, message="type is required")
        if not name:
            return SkillResult(success=False, message="name is required")
        if not content:
            return SkillResult(success=False, message="content is required")

        if not _validate_record_type(record_type):
            return SkillResult(
                success=False,
                message=f"Invalid record type: {record_type}. Valid: {RECORD_TYPES}",
            )

        validation_error = _validate_record_content(record_type, content)
        if validation_error:
            return SkillResult(success=False, message=validation_error)

        data = _load_data(self._data_file)
        ttl = params.get("ttl", data["config"]["default_ttl"])
        if not _validate_ttl(ttl):
            return SkillResult(
                success=False,
                message=f"Invalid TTL: {ttl}. Use 1 for auto or {TTL_MIN}-{TTL_MAX}",
            )

        # Build record payload
        record_data = {
            "type": record_type,
            "name": name,
            "content": content,
            "ttl": ttl,
        }

        # Proxied only applies to A, AAAA, CNAME
        if record_type in ("A", "AAAA", "CNAME"):
            record_data["proxied"] = params.get("proxied", data["config"]["default_proxied"])

        if record_type == "MX":
            record_data["priority"] = params.get("priority", 10)

        if record_type == "SRV":
            record_data["priority"] = params.get("priority", 0)

        resp = await self._api_request("POST", f"/zones/{zone_id}/dns_records", record_data)
        if not resp.get("success"):
            errors = resp.get("errors", [{"message": "Unknown error"}])
            return SkillResult(
                success=False,
                message=f"Failed to create record: {errors[0].get('message', 'Unknown error')}",
            )

        result = resp.get("result", {})
        _log_operation(data, "create_record", {
            "zone_id": zone_id,
            "type": record_type,
            "name": name,
            "content": content,
            "record_id": result.get("id"),
        })
        _save_data(data, self._data_file)

        return SkillResult(
            success=True,
            message=f"Created {record_type} record: {name} → {content}",
            data={
                "record_id": result.get("id"),
                "type": record_type,
                "name": result.get("name", name),
                "content": content,
                "ttl": result.get("ttl", ttl),
                "proxied": result.get("proxied", False),
            },
        )

    async def _update_record(self, params: Dict) -> SkillResult:
        """Update an existing DNS record."""
        zone_id = params.get("zone_id")
        record_id = params.get("record_id")

        if not zone_id:
            return SkillResult(success=False, message="zone_id is required")
        if not record_id:
            return SkillResult(success=False, message="record_id is required")

        # Build update payload with only provided fields
        update_data = {}
        if "type" in params:
            rt = params["type"].upper()
            if not _validate_record_type(rt):
                return SkillResult(success=False, message=f"Invalid record type: {rt}")
            update_data["type"] = rt

        if "name" in params:
            update_data["name"] = params["name"]

        if "content" in params:
            rt = update_data.get("type", params.get("type", "A")).upper()
            validation_error = _validate_record_content(rt, params["content"])
            if validation_error:
                return SkillResult(success=False, message=validation_error)
            update_data["content"] = params["content"]

        if "ttl" in params:
            if not _validate_ttl(params["ttl"]):
                return SkillResult(success=False, message=f"Invalid TTL: {params['ttl']}")
            update_data["ttl"] = params["ttl"]

        if "proxied" in params:
            update_data["proxied"] = params["proxied"]

        if not update_data:
            return SkillResult(success=False, message="No fields to update")

        resp = await self._api_request(
            "PATCH", f"/zones/{zone_id}/dns_records/{record_id}", update_data
        )
        if not resp.get("success"):
            errors = resp.get("errors", [{"message": "Unknown error"}])
            return SkillResult(
                success=False,
                message=f"Failed to update record: {errors[0].get('message', 'Unknown error')}",
            )

        result = resp.get("result", {})
        data = _load_data(self._data_file)
        _log_operation(data, "update_record", {
            "zone_id": zone_id,
            "record_id": record_id,
            "updates": update_data,
        })
        _save_data(data, self._data_file)

        return SkillResult(
            success=True,
            message=f"Updated record {record_id}",
            data={
                "record_id": result.get("id", record_id),
                "type": result.get("type"),
                "name": result.get("name"),
                "content": result.get("content"),
                "ttl": result.get("ttl"),
                "proxied": result.get("proxied"),
            },
        )

    async def _delete_record(self, params: Dict) -> SkillResult:
        """Delete a DNS record."""
        zone_id = params.get("zone_id")
        record_id = params.get("record_id")

        if not zone_id:
            return SkillResult(success=False, message="zone_id is required")
        if not record_id:
            return SkillResult(success=False, message="record_id is required")

        resp = await self._api_request(
            "DELETE", f"/zones/{zone_id}/dns_records/{record_id}"
        )
        if not resp.get("success"):
            errors = resp.get("errors", [{"message": "Unknown error"}])
            return SkillResult(
                success=False,
                message=f"Failed to delete record: {errors[0].get('message', 'Unknown error')}",
            )

        data = _load_data(self._data_file)
        _log_operation(data, "delete_record", {
            "zone_id": zone_id,
            "record_id": record_id,
        })
        _save_data(data, self._data_file)

        return SkillResult(
            success=True,
            message=f"Deleted record {record_id} from zone {zone_id}",
            data={"record_id": record_id, "deleted": True},
        )

    async def _bulk_create(self, params: Dict) -> SkillResult:
        """Create multiple DNS records at once."""
        zone_id = params.get("zone_id")
        records = params.get("records", [])

        if not zone_id:
            return SkillResult(success=False, message="zone_id is required")
        if not records:
            return SkillResult(success=False, message="records list is required and cannot be empty")
        if len(records) > MAX_BULK_RECORDS:
            return SkillResult(
                success=False,
                message=f"Too many records ({len(records)}). Max per batch: {MAX_BULK_RECORDS}",
            )

        # Validate all records first
        for i, rec in enumerate(records):
            if not rec.get("type"):
                return SkillResult(success=False, message=f"Record {i}: type is required")
            if not rec.get("name"):
                return SkillResult(success=False, message=f"Record {i}: name is required")
            if not rec.get("content"):
                return SkillResult(success=False, message=f"Record {i}: content is required")
            if not _validate_record_type(rec["type"]):
                return SkillResult(success=False, message=f"Record {i}: invalid type {rec['type']}")
            err = _validate_record_content(rec["type"].upper(), rec["content"])
            if err:
                return SkillResult(success=False, message=f"Record {i}: {err}")

        # Create records one by one (Cloudflare API doesn't have bulk endpoint)
        created = []
        failed = []
        for rec in records:
            result = await self._create_record({
                "zone_id": zone_id,
                **rec,
            })
            if result.success:
                created.append(result.data)
            else:
                failed.append({"record": rec, "error": result.message})

        data = _load_data(self._data_file)
        _log_operation(data, "bulk_create", {
            "zone_id": zone_id,
            "attempted": len(records),
            "created": len(created),
            "failed": len(failed),
        })
        _save_data(data, self._data_file)

        success = len(failed) == 0
        return SkillResult(
            success=success,
            message=f"Bulk create: {len(created)} created, {len(failed)} failed out of {len(records)}",
            data={
                "created": created,
                "failed": failed,
                "total": len(records),
                "success_count": len(created),
                "fail_count": len(failed),
            },
        )

    async def _find_record(self, params: Dict) -> SkillResult:
        """Find DNS records by name and optional type."""
        zone_id = params.get("zone_id")
        name = params.get("name")

        if not zone_id:
            return SkillResult(success=False, message="zone_id is required")
        if not name:
            return SkillResult(success=False, message="name is required")

        search_params = {"zone_id": zone_id, "name": name}
        if params.get("record_type"):
            search_params["record_type"] = params["record_type"]

        return await self._list_records(search_params)

    async def _wire_service(self, params: Dict) -> SkillResult:
        """High-level action: point a subdomain at a service target.

        Auto-detects whether target is an IP (creates A record) or
        hostname (creates CNAME), with sensible defaults.
        """
        zone_id = params.get("zone_id")
        subdomain = params.get("subdomain")
        target = params.get("target")

        if not zone_id:
            return SkillResult(success=False, message="zone_id is required")
        if not subdomain:
            return SkillResult(success=False, message="subdomain is required")
        if not target:
            return SkillResult(success=False, message="target is required")

        # Auto-detect target type
        target_type = params.get("target_type")
        if not target_type:
            # Check if it looks like an IP address
            parts = target.split(".")
            try:
                if len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts):
                    target_type = "ip"
                else:
                    target_type = "hostname"
            except ValueError:
                target_type = "hostname"

        if target_type == "ip":
            record_type = "A"
        else:
            record_type = "CNAME"

        proxied = params.get("proxied", True)

        # Check if record already exists
        existing = await self._find_record({
            "zone_id": zone_id,
            "name": subdomain,
            "record_type": record_type,
        })

        if existing.success and existing.data.get("count", 0) > 0:
            # Update existing record
            existing_record = existing.data["records"][0]
            if existing_record["content"] == target:
                return SkillResult(
                    success=True,
                    message=f"Record already exists: {subdomain} → {target}",
                    data=existing_record,
                )
            # Update to new target
            result = await self._update_record({
                "zone_id": zone_id,
                "record_id": existing_record["id"],
                "content": target,
                "proxied": proxied,
            })
            if result.success:
                result.message = f"Updated {subdomain} → {target} (was {existing_record['content']})"
            return result

        # Create new record
        result = await self._create_record({
            "zone_id": zone_id,
            "type": record_type,
            "name": subdomain,
            "content": target,
            "proxied": proxied,
        })
        if result.success:
            result.message = f"Wired {subdomain} → {target} ({record_type} record)"
        return result

    async def _history(self, params: Dict) -> SkillResult:
        """View DNS operation history."""
        limit = params.get("limit", 20)
        data = _load_data(self._data_file)
        ops = data.get("operations", [])
        recent = ops[-limit:] if len(ops) > limit else ops
        recent.reverse()  # Most recent first

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {len(ops)} operations",
            data={"operations": recent, "total": len(ops)},
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update default DNS configuration."""
        data = _load_data(self._data_file)

        if "default_ttl" in params:
            ttl = params["default_ttl"]
            if not _validate_ttl(ttl):
                return SkillResult(
                    success=False,
                    message=f"Invalid TTL: {ttl}. Use 1 for auto or {TTL_MIN}-{TTL_MAX}",
                )
            data["config"]["default_ttl"] = ttl

        if "default_proxied" in params:
            data["config"]["default_proxied"] = bool(params["default_proxied"])

        _save_data(data, self._data_file)

        return SkillResult(
            success=True,
            message="Configuration updated",
            data={"config": data["config"]},
        )
