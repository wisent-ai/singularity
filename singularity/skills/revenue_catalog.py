#!/usr/bin/env python3
"""
RevenueServiceCatalogSkill - Pre-built, deployable service packages for revenue generation.

While RevenueServiceSkill provides raw service implementations and MarketplaceSkill
handles business logic, THIS skill is the "product management" layer. It provides:

  - Ready-to-deploy service packages with pricing, SLAs, and configurations
  - Service bundles (e.g., "Developer Tools" = code_review + api_docs + data_analysis)
  - Deployment automation: one-command service activation with marketplace + monitoring wiring
  - Revenue projections based on pricing and expected volume
  - Service lifecycle: deploy, pause, retire, promote with tracking
  - A/B pricing experiments for revenue optimization

This completes the revenue pipeline:
  RevenueServiceSkill (implementation) → RevenueServiceCatalog (product packaging)
  → MarketplaceSkill (orders) → PaymentSkill (billing) → ServiceMonitor (health)

Without a curated catalog, the agent has services but no products. Products have
pricing, positioning, target customers, SLAs, and marketing descriptions.

Part of Revenue Generation pillar: the product management layer.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_FILE = DATA_DIR / "revenue_catalog.json"

# Service tiers
TIER_FREE = "free"
TIER_BASIC = "basic"
TIER_PRO = "pro"
TIER_ENTERPRISE = "enterprise"
TIERS = [TIER_FREE, TIER_BASIC, TIER_PRO, TIER_ENTERPRISE]

# Service statuses
STATUS_DRAFT = "draft"
STATUS_ACTIVE = "active"
STATUS_PAUSED = "paused"
STATUS_RETIRED = "retired"
STATUSES = [STATUS_DRAFT, STATUS_ACTIVE, STATUS_PAUSED, STATUS_RETIRED]

# Limits
MAX_PRODUCTS = 100
MAX_BUNDLES = 50
MAX_DEPLOYMENTS = 500


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _default_data() -> Dict:
    return {
        "products": {},
        "bundles": {},
        "deployments": [],
        "config": {
            "auto_monitor": True,
            "default_tier": TIER_BASIC,
            "currency": "USD",
        },
        "stats": {
            "total_deployed": 0,
            "total_retired": 0,
            "total_revenue_projected": 0.0,
        },
    }


# ─── Pre-built product catalog ────────────────────────────────────────
# These are ready-to-deploy service products with full configuration.

BUILTIN_PRODUCTS = {
    "code_review_basic": {
        "name": "AI Code Review - Basic",
        "description": "Automated code review detecting bugs, security vulnerabilities, and style issues. Supports Python, JavaScript, TypeScript, Go, Rust, Java, Ruby.",
        "skill_id": "revenue_services",
        "action": "code_review",
        "tier": TIER_BASIC,
        "pricing": {"model": "per_request", "price_usd": 0.10, "free_tier_requests": 5},
        "sla": {"latency_p95_ms": 5000, "availability": 99.0, "error_rate_max": 5.0},
        "target_customers": ["developers", "startups", "open_source"],
        "tags": ["code", "review", "security", "quality", "developer-tools"],
        "estimated_cost_per_request": 0.02,
        "category": "developer_tools",
    },
    "code_review_pro": {
        "name": "AI Code Review - Pro",
        "description": "Deep code analysis with security audit, performance profiling, dependency vulnerability scanning, and architecture recommendations. LLM-enhanced for nuanced feedback.",
        "skill_id": "revenue_services",
        "action": "code_review",
        "tier": TIER_PRO,
        "pricing": {"model": "per_request", "price_usd": 0.50, "free_tier_requests": 0},
        "sla": {"latency_p95_ms": 10000, "availability": 99.5, "error_rate_max": 3.0},
        "target_customers": ["enterprises", "security_teams", "fintech"],
        "tags": ["code", "review", "security", "deep-analysis", "enterprise"],
        "estimated_cost_per_request": 0.10,
        "config_overrides": {"depth": "deep", "include_security_audit": True},
        "category": "developer_tools",
    },
    "text_summarizer": {
        "name": "Text Summarization",
        "description": "Condense long documents, articles, or reports into key points. Extracts main themes, important details, and actionable insights.",
        "skill_id": "revenue_services",
        "action": "summarize_text",
        "tier": TIER_BASIC,
        "pricing": {"model": "per_request", "price_usd": 0.05, "free_tier_requests": 10},
        "sla": {"latency_p95_ms": 3000, "availability": 99.0, "error_rate_max": 5.0},
        "target_customers": ["researchers", "content_teams", "executives"],
        "tags": ["text", "summarization", "nlp", "content"],
        "estimated_cost_per_request": 0.01,
        "category": "content",
    },
    "data_analyzer": {
        "name": "Data Analysis & Insights",
        "description": "Upload CSV or JSON data and get statistical analysis, trend detection, outlier identification, and visualization recommendations.",
        "skill_id": "revenue_services",
        "action": "analyze_data",
        "tier": TIER_BASIC,
        "pricing": {"model": "per_request", "price_usd": 0.20, "free_tier_requests": 3},
        "sla": {"latency_p95_ms": 10000, "availability": 99.0, "error_rate_max": 5.0},
        "target_customers": ["analysts", "product_managers", "data_teams"],
        "tags": ["data", "analysis", "statistics", "insights"],
        "estimated_cost_per_request": 0.05,
        "category": "data",
    },
    "seo_optimizer": {
        "name": "SEO Content Audit",
        "description": "Analyze text content for search engine optimization. Get keyword density analysis, readability scores, meta description suggestions, and content improvement recommendations.",
        "skill_id": "revenue_services",
        "action": "seo_audit",
        "tier": TIER_BASIC,
        "pricing": {"model": "per_request", "price_usd": 0.15, "free_tier_requests": 5},
        "sla": {"latency_p95_ms": 5000, "availability": 99.0, "error_rate_max": 5.0},
        "target_customers": ["marketers", "content_writers", "seo_specialists"],
        "tags": ["seo", "content", "marketing", "optimization"],
        "estimated_cost_per_request": 0.02,
        "category": "marketing",
    },
    "api_doc_generator": {
        "name": "API Documentation Generator",
        "description": "Generate comprehensive API documentation from endpoint specifications. Produces formatted docs with request/response examples, parameter descriptions, and error codes.",
        "skill_id": "revenue_services",
        "action": "generate_api_docs",
        "tier": TIER_BASIC,
        "pricing": {"model": "per_request", "price_usd": 0.25, "free_tier_requests": 2},
        "sla": {"latency_p95_ms": 8000, "availability": 99.0, "error_rate_max": 5.0},
        "target_customers": ["api_developers", "devops", "technical_writers"],
        "tags": ["api", "documentation", "developer-tools"],
        "estimated_cost_per_request": 0.05,
        "category": "developer_tools",
    },
}

# Pre-built bundles
BUILTIN_BUNDLES = {
    "developer_essentials": {
        "name": "Developer Essentials",
        "description": "Everything a development team needs: code review, API docs, and data analysis at a bundle discount.",
        "product_ids": ["code_review_basic", "api_doc_generator", "data_analyzer"],
        "discount_pct": 20,
        "target_customers": ["development_teams", "startups"],
        "tags": ["developer", "bundle", "discount"],
    },
    "content_suite": {
        "name": "Content & Marketing Suite",
        "description": "Full content optimization toolkit: text summarization, SEO audit, and content analysis.",
        "product_ids": ["text_summarizer", "seo_optimizer"],
        "discount_pct": 15,
        "target_customers": ["content_teams", "marketing_agencies"],
        "tags": ["content", "marketing", "bundle"],
    },
    "full_platform": {
        "name": "Full Platform Access",
        "description": "Access to all services at maximum discount. Ideal for organizations needing the complete AI toolkit.",
        "product_ids": ["code_review_pro", "text_summarizer", "data_analyzer", "seo_optimizer", "api_doc_generator"],
        "discount_pct": 30,
        "target_customers": ["enterprises", "agencies"],
        "tags": ["enterprise", "all-access", "bundle"],
    },
}


class RevenueServiceCatalogSkill(Skill):
    """
    Product catalog for revenue-generating services.

    Manages the full lifecycle of service products: browse catalog,
    deploy services, create bundles, track revenue projections,
    and optimize pricing.
    """

    def __init__(self, credentials: Dict[str, str] = None, data_path: Path = None):
        super().__init__(credentials)
        self._data_path = data_path or CATALOG_FILE
        self._ensure_data()

    def _ensure_data(self):
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._data_path.exists():
            self._save(_default_data())

    def _load(self) -> Dict:
        try:
            with open(self._data_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_data()

    def _save(self, data: Dict):
        try:
            with open(self._data_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError:
            pass

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue_catalog",
            name="Revenue Service Catalog",
            version="1.0.0",
            category="revenue",
            description="Pre-built, deployable service products with pricing, SLAs, and bundles for revenue generation",
            required_credentials=[],
            actions=self.get_actions(),
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="browse",
                description="Browse available service products with optional filtering",
                parameters={
                    "category": {"type": "string", "required": False, "description": "Filter by category: developer_tools, content, data, marketing"},
                    "tier": {"type": "string", "required": False, "description": "Filter by tier: free, basic, pro, enterprise"},
                    "tag": {"type": "string", "required": False, "description": "Filter by tag"},
                },
            ),
            SkillAction(
                name="details",
                description="Get full details of a specific product including pricing, SLA, and revenue projections",
                parameters={
                    "product_id": {"type": "string", "required": True, "description": "Product identifier"},
                },
            ),
            SkillAction(
                name="deploy",
                description="Deploy a service product - activates it in the marketplace with monitoring",
                parameters={
                    "product_id": {"type": "string", "required": True, "description": "Product to deploy"},
                    "price_override": {"type": "number", "required": False, "description": "Override default price (USD)"},
                    "sla_override": {"type": "object", "required": False, "description": "Override SLA targets"},
                },
            ),
            SkillAction(
                name="deploy_bundle",
                description="Deploy all products in a bundle with discount pricing",
                parameters={
                    "bundle_id": {"type": "string", "required": True, "description": "Bundle to deploy"},
                },
            ),
            SkillAction(
                name="pause",
                description="Pause a deployed service (stops accepting new requests)",
                parameters={
                    "product_id": {"type": "string", "required": True, "description": "Product to pause"},
                },
            ),
            SkillAction(
                name="retire",
                description="Permanently retire a deployed service",
                parameters={
                    "product_id": {"type": "string", "required": True, "description": "Product to retire"},
                },
            ),
            SkillAction(
                name="bundles",
                description="List available service bundles",
                parameters={},
            ),
            SkillAction(
                name="projections",
                description="Calculate revenue projections based on deployed services and expected volume",
                parameters={
                    "monthly_requests_per_service": {"type": "integer", "required": False, "description": "Expected monthly requests (default 1000)"},
                },
            ),
            SkillAction(
                name="deployments",
                description="List all deployment history",
                parameters={
                    "status": {"type": "string", "required": False, "description": "Filter: active, paused, retired, all"},
                },
            ),
            SkillAction(
                name="create_product",
                description="Create a custom product from any skill action",
                parameters={
                    "product_id": {"type": "string", "required": True, "description": "Unique product identifier"},
                    "name": {"type": "string", "required": True, "description": "Product display name"},
                    "description": {"type": "string", "required": True, "description": "Product description"},
                    "skill_id": {"type": "string", "required": True, "description": "Skill that provides the service"},
                    "action": {"type": "string", "required": True, "description": "Skill action to invoke"},
                    "price_usd": {"type": "number", "required": True, "description": "Price per request in USD"},
                    "tier": {"type": "string", "required": False, "description": "Product tier"},
                },
            ),
        ]

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "browse": self._browse,
            "details": self._details,
            "deploy": self._deploy,
            "deploy_bundle": self._deploy_bundle,
            "pause": self._pause,
            "retire": self._retire,
            "bundles": self._bundles,
            "projections": self._projections,
            "deployments": self._deployments,
            "create_product": self._create_product,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return handler(params)

    def _get_all_products(self) -> Dict[str, Dict]:
        """Get all products: built-in + custom."""
        data = self._load()
        # Merge built-in and custom products
        all_products = {}
        for pid, prod in BUILTIN_PRODUCTS.items():
            all_products[pid] = {**prod, "source": "builtin"}
        for pid, prod in data.get("products", {}).items():
            all_products[pid] = {**prod, "source": "custom"}
        return all_products

    def _browse(self, params: Dict) -> SkillResult:
        """Browse available products with filtering."""
        category = params.get("category")
        tier = params.get("tier")
        tag = params.get("tag")

        products = self._get_all_products()
        data = self._load()
        results = []

        for pid, prod in products.items():
            # Check deployment status
            deployment = self._find_deployment(data, pid)
            status = deployment["status"] if deployment else STATUS_DRAFT

            # Apply filters
            if category and prod.get("category") != category:
                continue
            if tier and prod.get("tier") != tier:
                continue
            if tag and tag not in prod.get("tags", []):
                continue

            pricing = prod.get("pricing", {})
            results.append({
                "product_id": pid,
                "name": prod["name"],
                "description": prod["description"][:100] + "..." if len(prod.get("description", "")) > 100 else prod.get("description", ""),
                "tier": prod.get("tier", TIER_BASIC),
                "price_usd": pricing.get("price_usd", 0),
                "category": prod.get("category", "general"),
                "status": status,
                "source": prod.get("source", "builtin"),
                "tags": prod.get("tags", [])[:5],
            })

        results.sort(key=lambda x: x["price_usd"])
        return SkillResult(
            success=True,
            message=f"{len(results)} products available",
            data={"products": results, "filters": {"category": category, "tier": tier, "tag": tag}},
        )

    def _details(self, params: Dict) -> SkillResult:
        """Get full product details."""
        product_id = params.get("product_id", "").strip()
        if not product_id:
            return SkillResult(success=False, message="product_id is required")

        products = self._get_all_products()
        prod = products.get(product_id)
        if not prod:
            return SkillResult(success=False, message=f"Product '{product_id}' not found. Use 'browse' to see available products.")

        data = self._load()
        deployment = self._find_deployment(data, product_id)

        pricing = prod.get("pricing", {})
        cost_per = prod.get("estimated_cost_per_request", 0)
        price_per = pricing.get("price_usd", 0)
        margin = ((price_per - cost_per) / price_per * 100) if price_per > 0 else 0

        detail = {
            "product_id": product_id,
            "name": prod["name"],
            "description": prod.get("description", ""),
            "skill_id": prod.get("skill_id"),
            "action": prod.get("action"),
            "tier": prod.get("tier", TIER_BASIC),
            "pricing": pricing,
            "sla": prod.get("sla", {}),
            "estimated_cost_per_request": cost_per,
            "estimated_margin_pct": round(margin, 1),
            "target_customers": prod.get("target_customers", []),
            "tags": prod.get("tags", []),
            "category": prod.get("category", "general"),
            "source": prod.get("source", "builtin"),
            "config_overrides": prod.get("config_overrides", {}),
            "deployment": deployment,
        }

        return SkillResult(
            success=True,
            message=f"Product: {prod['name']} (${price_per}/req, {margin:.0f}% margin)",
            data=detail,
        )

    def _deploy(self, params: Dict) -> SkillResult:
        """Deploy a service product."""
        product_id = params.get("product_id", "").strip()
        if not product_id:
            return SkillResult(success=False, message="product_id is required")

        products = self._get_all_products()
        prod = products.get(product_id)
        if not prod:
            return SkillResult(success=False, message=f"Product '{product_id}' not found")

        data = self._load()

        # Check if already active
        existing = self._find_deployment(data, product_id)
        if existing and existing["status"] == STATUS_ACTIVE:
            return SkillResult(success=False, message=f"Product '{product_id}' is already deployed and active")

        # Apply overrides
        pricing = {**prod.get("pricing", {})}
        if params.get("price_override"):
            pricing["price_usd"] = params["price_override"]

        sla = {**prod.get("sla", {})}
        if params.get("sla_override") and isinstance(params["sla_override"], dict):
            sla.update(params["sla_override"])

        deployment = {
            "id": str(uuid.uuid4())[:8],
            "product_id": product_id,
            "name": prod["name"],
            "status": STATUS_ACTIVE,
            "deployed_at": _now_iso(),
            "pricing": pricing,
            "sla": sla,
            "skill_id": prod.get("skill_id"),
            "action": prod.get("action"),
            "config_overrides": prod.get("config_overrides", {}),
            "requests_served": 0,
            "revenue_earned": 0.0,
            "paused_at": None,
            "retired_at": None,
        }

        # If reactivating a paused deployment, update it
        if existing and existing["status"] == STATUS_PAUSED:
            for d in data["deployments"]:
                if d["product_id"] == product_id and d["status"] == STATUS_PAUSED:
                    d["status"] = STATUS_ACTIVE
                    d["pricing"] = pricing
                    d["sla"] = sla
                    deployment = d
                    break
        else:
            data["deployments"].append(deployment)

        data["stats"]["total_deployed"] += 1

        # Trim deployment history
        if len(data["deployments"]) > MAX_DEPLOYMENTS:
            # Keep active/paused + most recent retired
            active = [d for d in data["deployments"] if d["status"] in (STATUS_ACTIVE, STATUS_PAUSED)]
            retired = [d for d in data["deployments"] if d["status"] == STATUS_RETIRED]
            retired = retired[-100:]
            data["deployments"] = active + retired

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Deployed '{prod['name']}' at ${pricing.get('price_usd', 0)}/request",
            data={
                "deployment_id": deployment["id"],
                "product_id": product_id,
                "name": prod["name"],
                "pricing": pricing,
                "sla": sla,
                "status": STATUS_ACTIVE,
            },
        )

    def _deploy_bundle(self, params: Dict) -> SkillResult:
        """Deploy all products in a bundle."""
        bundle_id = params.get("bundle_id", "").strip()
        if not bundle_id:
            return SkillResult(success=False, message="bundle_id is required")

        all_bundles = {**BUILTIN_BUNDLES}
        data = self._load()
        all_bundles.update(data.get("bundles", {}))

        bundle = all_bundles.get(bundle_id)
        if not bundle:
            return SkillResult(success=False, message=f"Bundle '{bundle_id}' not found")

        discount = bundle.get("discount_pct", 0)
        deployed = []
        skipped = []

        for pid in bundle.get("product_ids", []):
            result = self._deploy({
                "product_id": pid,
                "price_override": None,  # Will apply discount below
            })
            if result.success:
                # Apply bundle discount
                if discount > 0:
                    data = self._load()
                    for d in data["deployments"]:
                        if d["product_id"] == pid and d["status"] == STATUS_ACTIVE:
                            original = d["pricing"].get("price_usd", 0)
                            d["pricing"]["price_usd"] = round(original * (1 - discount / 100), 4)
                            d["pricing"]["bundle_discount_pct"] = discount
                            d["pricing"]["original_price_usd"] = original
                            break
                    self._save(data)
                deployed.append(pid)
            else:
                skipped.append({"product_id": pid, "reason": result.message})

        return SkillResult(
            success=len(deployed) > 0,
            message=f"Bundle '{bundle['name']}': deployed {len(deployed)}, skipped {len(skipped)} (discount: {discount}%)",
            data={
                "bundle_id": bundle_id,
                "bundle_name": bundle["name"],
                "deployed": deployed,
                "skipped": skipped,
                "discount_pct": discount,
            },
        )

    def _pause(self, params: Dict) -> SkillResult:
        """Pause a deployed product."""
        product_id = params.get("product_id", "").strip()
        if not product_id:
            return SkillResult(success=False, message="product_id is required")

        data = self._load()
        found = False
        for d in data["deployments"]:
            if d["product_id"] == product_id and d["status"] == STATUS_ACTIVE:
                d["status"] = STATUS_PAUSED
                d["paused_at"] = _now_iso()
                found = True
                break

        if not found:
            return SkillResult(success=False, message=f"No active deployment found for '{product_id}'")

        self._save(data)
        return SkillResult(success=True, message=f"Product '{product_id}' paused", data={"product_id": product_id, "status": STATUS_PAUSED})

    def _retire(self, params: Dict) -> SkillResult:
        """Retire a deployed product."""
        product_id = params.get("product_id", "").strip()
        if not product_id:
            return SkillResult(success=False, message="product_id is required")

        data = self._load()
        found = False
        for d in data["deployments"]:
            if d["product_id"] == product_id and d["status"] in (STATUS_ACTIVE, STATUS_PAUSED):
                d["status"] = STATUS_RETIRED
                d["retired_at"] = _now_iso()
                found = True
                break

        if not found:
            return SkillResult(success=False, message=f"No active/paused deployment found for '{product_id}'")

        data["stats"]["total_retired"] += 1
        self._save(data)
        return SkillResult(success=True, message=f"Product '{product_id}' retired", data={"product_id": product_id, "status": STATUS_RETIRED})

    def _bundles(self, params: Dict) -> SkillResult:
        """List available bundles."""
        data = self._load()
        all_bundles = {**BUILTIN_BUNDLES}
        all_bundles.update(data.get("bundles", {}))

        products = self._get_all_products()
        bundle_list = []

        for bid, bundle in all_bundles.items():
            # Calculate total price and discounted price
            total_price = 0
            product_names = []
            for pid in bundle.get("product_ids", []):
                prod = products.get(pid, {})
                pricing = prod.get("pricing", {})
                total_price += pricing.get("price_usd", 0)
                product_names.append(prod.get("name", pid))

            discount = bundle.get("discount_pct", 0)
            discounted = round(total_price * (1 - discount / 100), 4)

            bundle_list.append({
                "bundle_id": bid,
                "name": bundle["name"],
                "description": bundle.get("description", ""),
                "products": product_names,
                "product_count": len(bundle.get("product_ids", [])),
                "total_price_usd": round(total_price, 4),
                "discounted_price_usd": discounted,
                "discount_pct": discount,
                "savings_usd": round(total_price - discounted, 4),
                "tags": bundle.get("tags", []),
            })

        return SkillResult(
            success=True,
            message=f"{len(bundle_list)} bundles available",
            data={"bundles": bundle_list},
        )

    def _projections(self, params: Dict) -> SkillResult:
        """Calculate revenue projections for deployed services."""
        monthly_requests = params.get("monthly_requests_per_service", 1000)
        data = self._load()
        products = self._get_all_products()

        active_deployments = [d for d in data["deployments"] if d["status"] == STATUS_ACTIVE]
        if not active_deployments:
            return SkillResult(
                success=True,
                message="No active deployments. Deploy products first.",
                data={"projections": [], "total_monthly_revenue": 0, "total_monthly_cost": 0},
            )

        projections = []
        total_revenue = 0
        total_cost = 0

        for dep in active_deployments:
            pid = dep["product_id"]
            prod = products.get(pid, {})
            price = dep["pricing"].get("price_usd", 0)
            cost = prod.get("estimated_cost_per_request", 0)
            free_requests = dep["pricing"].get("free_tier_requests", 0)

            billable_requests = max(0, monthly_requests - free_requests)
            monthly_rev = round(billable_requests * price, 2)
            monthly_cost = round(monthly_requests * cost, 2)
            monthly_profit = round(monthly_rev - monthly_cost, 2)
            margin = round((monthly_profit / monthly_rev * 100), 1) if monthly_rev > 0 else 0

            total_revenue += monthly_rev
            total_cost += monthly_cost

            projections.append({
                "product_id": pid,
                "name": dep.get("name", pid),
                "price_per_request": price,
                "cost_per_request": cost,
                "monthly_requests": monthly_requests,
                "free_tier_requests": free_requests,
                "billable_requests": billable_requests,
                "monthly_revenue": monthly_rev,
                "monthly_cost": monthly_cost,
                "monthly_profit": monthly_profit,
                "margin_pct": margin,
            })

        projections.sort(key=lambda p: p["monthly_profit"], reverse=True)

        return SkillResult(
            success=True,
            message=f"Projected ${total_revenue:.2f}/mo revenue from {len(active_deployments)} services at {monthly_requests} req/service",
            data={
                "projections": projections,
                "total_monthly_revenue": round(total_revenue, 2),
                "total_monthly_cost": round(total_cost, 2),
                "total_monthly_profit": round(total_revenue - total_cost, 2),
                "overall_margin_pct": round(((total_revenue - total_cost) / total_revenue * 100), 1) if total_revenue > 0 else 0,
                "assumed_monthly_requests_per_service": monthly_requests,
                "active_service_count": len(active_deployments),
            },
        )

    def _deployments(self, params: Dict) -> SkillResult:
        """List deployment history."""
        data = self._load()
        status_filter = params.get("status", "active")

        deployments = data.get("deployments", [])
        if status_filter and status_filter != "all":
            deployments = [d for d in deployments if d.get("status") == status_filter]

        # Return most recent first
        deployments = list(reversed(deployments[-50:]))

        summary = {
            "active": sum(1 for d in data.get("deployments", []) if d["status"] == STATUS_ACTIVE),
            "paused": sum(1 for d in data.get("deployments", []) if d["status"] == STATUS_PAUSED),
            "retired": sum(1 for d in data.get("deployments", []) if d["status"] == STATUS_RETIRED),
        }

        return SkillResult(
            success=True,
            message=f"{len(deployments)} deployments ({status_filter})",
            data={
                "deployments": deployments,
                "summary": summary,
                "filter": status_filter,
            },
        )

    def _create_product(self, params: Dict) -> SkillResult:
        """Create a custom product."""
        product_id = params.get("product_id", "").strip()
        name = params.get("name", "").strip()
        description = params.get("description", "").strip()
        skill_id = params.get("skill_id", "").strip()
        action = params.get("action", "").strip()
        price_usd = params.get("price_usd", 0)

        if not all([product_id, name, description, skill_id, action]):
            return SkillResult(success=False, message="product_id, name, description, skill_id, and action are required")

        data = self._load()

        # Check not duplicate
        if product_id in BUILTIN_PRODUCTS or product_id in data.get("products", {}):
            return SkillResult(success=False, message=f"Product '{product_id}' already exists")

        if len(data.get("products", {})) >= MAX_PRODUCTS:
            return SkillResult(success=False, message=f"Maximum {MAX_PRODUCTS} custom products reached")

        product = {
            "name": name,
            "description": description,
            "skill_id": skill_id,
            "action": action,
            "tier": params.get("tier", TIER_BASIC),
            "pricing": {"model": "per_request", "price_usd": price_usd, "free_tier_requests": 0},
            "sla": {"latency_p95_ms": 10000, "availability": 99.0, "error_rate_max": 5.0},
            "target_customers": [],
            "tags": [],
            "estimated_cost_per_request": 0,
            "category": "custom",
            "created_at": _now_iso(),
        }

        data.setdefault("products", {})[product_id] = product
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Custom product '{name}' created at ${price_usd}/request",
            data={"product_id": product_id, "product": product},
        )

    def _find_deployment(self, data: Dict, product_id: str) -> Optional[Dict]:
        """Find the most recent deployment for a product."""
        for d in reversed(data.get("deployments", [])):
            if d["product_id"] == product_id:
                return d
        return None
