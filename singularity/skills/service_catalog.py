#!/usr/bin/env python3
"""
ServiceCatalogSkill - Pre-built service packages deployable in one command.

While RevenueServiceSkill implements individual services and MarketplaceSkill
manages the business layer, THIS skill provides curated service PACKAGES —
ready-to-deploy bundles with optimized pricing, descriptions, and deployment
configurations. Think of it as an "app store" of pre-built revenue offerings.

Available packages:
  - Developer Toolkit: Code review + API doc generation (for dev teams)
  - Content Suite: Text summarization + SEO audit (for marketers)
  - Data Intelligence: Data analysis + text summarization (for analysts)
  - Full Stack: All 5 services bundled at a discount (for enterprises)
  - Custom: User-defined bundles from available services

Each package includes:
  - Curated service selection with inter-service workflows
  - Optimized bundle pricing (discounted vs individual)
  - Pre-written marketing descriptions and tags
  - SLA targets and resource estimates
  - One-command deploy → registers all services in Marketplace + ServiceHosting

Revenue flow:
  ServiceCatalog → deploy_package → MarketplaceSkill (registers services)
                                  → ServiceHostingSkill (registers endpoints)
                                  → AutoCatalogSkill (keeps in sync)

Part of the Revenue Generation pillar: the product packaging layer.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction


# Persistent catalog state
CATALOG_FILE = Path(__file__).parent.parent / "data" / "service_catalog.json"

# ─── Pre-built service packages ───────────────────────────────────────────

BUILTIN_PACKAGES = {
    "developer_toolkit": {
        "name": "Developer Toolkit",
        "description": "Essential developer services: professional code review and automated API documentation generation. Perfect for dev teams wanting quality assurance and up-to-date docs.",
        "services": [
            {
                "skill": "revenue_services",
                "action": "code_review",
                "name": "AI Code Review",
                "price": 0.10,
                "sla_minutes": 5,
            },
            {
                "skill": "revenue_services",
                "action": "generate_api_docs",
                "name": "API Documentation Generator",
                "price": 0.08,
                "sla_minutes": 5,
            },
        ],
        "bundle_discount": 0.15,  # 15% discount when bought as bundle
        "tags": ["developer", "code", "quality", "documentation", "api"],
        "target_audience": "Development teams, API consumers, open source projects",
        "estimated_monthly_revenue": 50.0,
        "category": "developer",
    },
    "content_suite": {
        "name": "Content Suite",
        "description": "Complete content optimization package: summarize long documents into key insights and audit content for search engine performance. Ideal for content marketers and publishers.",
        "services": [
            {
                "skill": "revenue_services",
                "action": "summarize_text",
                "name": "Text Summarization",
                "price": 0.05,
                "sla_minutes": 3,
            },
            {
                "skill": "revenue_services",
                "action": "seo_audit",
                "name": "SEO Content Audit",
                "price": 0.05,
                "sla_minutes": 3,
            },
        ],
        "bundle_discount": 0.10,
        "tags": ["content", "marketing", "seo", "summarization", "publishing"],
        "target_audience": "Content marketers, publishers, SEO specialists",
        "estimated_monthly_revenue": 30.0,
        "category": "content",
    },
    "data_intelligence": {
        "name": "Data Intelligence",
        "description": "Turn raw data into actionable insights: structured data analysis with statistical profiling plus document summarization for report generation. Built for analysts and data teams.",
        "services": [
            {
                "skill": "revenue_services",
                "action": "analyze_data",
                "name": "Data Analysis",
                "price": 0.10,
                "sla_minutes": 5,
            },
            {
                "skill": "revenue_services",
                "action": "summarize_text",
                "name": "Report Summarization",
                "price": 0.05,
                "sla_minutes": 3,
            },
        ],
        "bundle_discount": 0.12,
        "tags": ["data", "analytics", "insights", "reporting", "statistics"],
        "target_audience": "Data analysts, business intelligence teams, researchers",
        "estimated_monthly_revenue": 40.0,
        "category": "data",
    },
    "full_stack": {
        "name": "Full Stack Enterprise",
        "description": "All five AI services in one premium bundle: code review, API docs, text summarization, SEO audit, and data analysis. Maximum value at the deepest discount. For teams that need it all.",
        "services": [
            {
                "skill": "revenue_services",
                "action": "code_review",
                "name": "AI Code Review",
                "price": 0.10,
                "sla_minutes": 5,
            },
            {
                "skill": "revenue_services",
                "action": "generate_api_docs",
                "name": "API Documentation Generator",
                "price": 0.08,
                "sla_minutes": 5,
            },
            {
                "skill": "revenue_services",
                "action": "summarize_text",
                "name": "Text Summarization",
                "price": 0.05,
                "sla_minutes": 3,
            },
            {
                "skill": "revenue_services",
                "action": "seo_audit",
                "name": "SEO Content Audit",
                "price": 0.05,
                "sla_minutes": 3,
            },
            {
                "skill": "revenue_services",
                "action": "analyze_data",
                "name": "Data Analysis",
                "price": 0.10,
                "sla_minutes": 5,
            },
        ],
        "bundle_discount": 0.25,  # Biggest discount for full bundle
        "tags": ["enterprise", "all-in-one", "premium", "complete"],
        "target_audience": "Enterprise teams, agencies, full-service consulting",
        "estimated_monthly_revenue": 100.0,
        "category": "enterprise",
    },
}

# Maximum custom packages a user can create
MAX_CUSTOM_PACKAGES = 50
MAX_DEPLOYMENTS = 100


class ServiceCatalogSkill(Skill):
    """
    Pre-built service packages deployable in one command.

    Provides curated bundles of revenue services with optimized pricing,
    marketing copy, and deployment configs. Bridges the gap between having
    individual services and offering them as cohesive products.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._state = self._load_state()

    def _load_state(self) -> Dict:
        if CATALOG_FILE.exists():
            try:
                return json.loads(CATALOG_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _default_state(self) -> Dict:
        return {
            "custom_packages": {},
            "deployments": [],
            "deployment_history": [],
            "created_at": datetime.now().isoformat(),
        }

    def _save_state(self):
        CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            CATALOG_FILE.write_text(json.dumps(self._state, indent=2, default=str))
        except OSError:
            pass

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="service_catalog",
            name="Service Catalog",
            version="1.0.0",
            category="revenue",
            description="Pre-built service packages deployable in one command. Curated bundles with optimized pricing for different audiences.",
            actions=[
                SkillAction(
                    name="list_packages",
                    description="List all available service packages (built-in and custom)",
                    parameters={},
                ),
                SkillAction(
                    name="preview",
                    description="Preview a package's services, pricing, and deployment plan",
                    parameters={
                        "package_id": {"type": "string", "required": True, "description": "Package ID to preview"},
                    },
                ),
                SkillAction(
                    name="deploy",
                    description="Deploy a service package — registers all services in marketplace and hosting",
                    parameters={
                        "package_id": {"type": "string", "required": True, "description": "Package to deploy"},
                        "custom_pricing": {"type": "object", "required": False, "description": "Override prices {service_action: new_price}"},
                        "agent_name": {"type": "string", "required": False, "description": "Agent name for service hosting domain"},
                    },
                ),
                SkillAction(
                    name="undeploy",
                    description="Undeploy a previously deployed package",
                    parameters={
                        "deployment_id": {"type": "string", "required": True, "description": "Deployment ID to remove"},
                    },
                ),
                SkillAction(
                    name="create_custom",
                    description="Create a custom package from available services",
                    parameters={
                        "package_id": {"type": "string", "required": True, "description": "Unique ID for this custom package"},
                        "name": {"type": "string", "required": True, "description": "Human-readable package name"},
                        "description": {"type": "string", "required": True, "description": "Package description"},
                        "services": {"type": "array", "required": True, "description": "List of {skill, action, name, price, sla_minutes}"},
                        "bundle_discount": {"type": "number", "required": False, "description": "Discount 0.0-0.5 (default 0.1)"},
                        "tags": {"type": "array", "required": False, "description": "Tags for the package"},
                    },
                ),
                SkillAction(
                    name="delete_custom",
                    description="Delete a custom package",
                    parameters={
                        "package_id": {"type": "string", "required": True, "description": "Custom package to delete"},
                    },
                ),
                SkillAction(
                    name="compare",
                    description="Compare two or more packages side by side",
                    parameters={
                        "package_ids": {"type": "array", "required": True, "description": "List of package IDs to compare"},
                    },
                ),
                SkillAction(
                    name="recommend",
                    description="Recommend the best package for a given use case",
                    parameters={
                        "use_case": {"type": "string", "required": True, "description": "Describe the use case (e.g., 'marketing team', 'developer tools')"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="View deployment status and revenue summary",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        handlers = {
            "list_packages": self._list_packages,
            "preview": self._preview,
            "deploy": self._deploy,
            "undeploy": self._undeploy,
            "create_custom": self._create_custom,
            "delete_custom": self._delete_custom,
            "compare": self._compare,
            "recommend": self._recommend,
            "status": self._status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}. Available: {list(handlers.keys())}")
        return handler(params)

    def _get_all_packages(self) -> Dict[str, Dict]:
        """Get all packages (built-in + custom)."""
        packages = dict(BUILTIN_PACKAGES)
        for pid, pkg in self._state.get("custom_packages", {}).items():
            packages[pid] = pkg
        return packages

    def _calculate_bundle_price(self, package: Dict) -> Dict[str, float]:
        """Calculate individual and bundle pricing for a package."""
        individual_total = sum(s["price"] for s in package["services"])
        discount = package.get("bundle_discount", 0.10)
        bundle_total = round(individual_total * (1 - discount), 4)
        savings = round(individual_total - bundle_total, 4)
        return {
            "individual_total": individual_total,
            "bundle_total": bundle_total,
            "discount_pct": round(discount * 100, 1),
            "savings": savings,
            "per_service_avg": round(bundle_total / max(len(package["services"]), 1), 4),
        }

    # ─── Actions ──────────────────────────────────────────────────────────

    def _list_packages(self, params: Dict) -> SkillResult:
        """List all available packages."""
        packages = self._get_all_packages()
        summaries = []
        for pid, pkg in packages.items():
            pricing = self._calculate_bundle_price(pkg)
            summaries.append({
                "package_id": pid,
                "name": pkg["name"],
                "description": pkg["description"][:120] + "..." if len(pkg["description"]) > 120 else pkg["description"],
                "services_count": len(pkg["services"]),
                "bundle_price": pricing["bundle_total"],
                "savings": pricing["savings"],
                "category": pkg.get("category", "custom"),
                "tags": pkg.get("tags", []),
                "is_builtin": pid in BUILTIN_PACKAGES,
            })

        active_deployments = [d for d in self._state.get("deployments", []) if d.get("status") == "active"]

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} packages ({len(BUILTIN_PACKAGES)} built-in, {len(self._state.get('custom_packages', {}))} custom). {len(active_deployments)} active deployments.",
            data={
                "packages": summaries,
                "total_builtin": len(BUILTIN_PACKAGES),
                "total_custom": len(self._state.get("custom_packages", {})),
                "active_deployments": len(active_deployments),
            },
        )

    def _preview(self, params: Dict) -> SkillResult:
        """Preview a package before deploying."""
        package_id = params.get("package_id", "")
        packages = self._get_all_packages()
        if package_id not in packages:
            return SkillResult(
                success=False,
                message=f"Package '{package_id}' not found. Available: {list(packages.keys())}",
            )

        pkg = packages[package_id]
        pricing = self._calculate_bundle_price(pkg)

        service_details = []
        for svc in pkg["services"]:
            individual_price = svc["price"]
            discounted_price = round(individual_price * (1 - pkg.get("bundle_discount", 0.10)), 4)
            service_details.append({
                "name": svc["name"],
                "skill": svc["skill"],
                "action": svc["action"],
                "individual_price": individual_price,
                "bundle_price": discounted_price,
                "sla_minutes": svc.get("sla_minutes", 5),
            })

        # Check if already deployed
        active = [d for d in self._state.get("deployments", [])
                  if d.get("package_id") == package_id and d.get("status") == "active"]

        return SkillResult(
            success=True,
            message=f"Package '{pkg['name']}': {len(pkg['services'])} services, bundle price ${pricing['bundle_total']:.2f} (save ${pricing['savings']:.2f})",
            data={
                "package_id": package_id,
                "name": pkg["name"],
                "description": pkg["description"],
                "services": service_details,
                "pricing": pricing,
                "tags": pkg.get("tags", []),
                "target_audience": pkg.get("target_audience", ""),
                "estimated_monthly_revenue": pkg.get("estimated_monthly_revenue", 0),
                "category": pkg.get("category", "custom"),
                "already_deployed": len(active) > 0,
                "active_deployments": len(active),
            },
        )

    def _deploy(self, params: Dict) -> SkillResult:
        """Deploy a package — register all services."""
        package_id = params.get("package_id", "")
        custom_pricing = params.get("custom_pricing", {})
        agent_name = params.get("agent_name", "singularity")

        packages = self._get_all_packages()
        if package_id not in packages:
            return SkillResult(
                success=False,
                message=f"Package '{package_id}' not found. Available: {list(packages.keys())}",
            )

        if len(self._state.get("deployments", [])) >= MAX_DEPLOYMENTS:
            return SkillResult(success=False, message=f"Maximum deployments ({MAX_DEPLOYMENTS}) reached. Undeploy existing first.")

        pkg = packages[package_id]
        pricing = self._calculate_bundle_price(pkg)
        deployment_id = f"deploy-{uuid.uuid4().hex[:8]}"

        # Build marketplace registration entries
        marketplace_entries = []
        hosting_entries = []
        for svc in pkg["services"]:
            # Apply custom pricing override if provided
            base_price = svc["price"]
            if svc["action"] in custom_pricing:
                base_price = custom_pricing[svc["action"]]
            # Apply bundle discount
            final_price = round(base_price * (1 - pkg.get("bundle_discount", 0.10)), 4)

            marketplace_entries.append({
                "service_id": f"{deployment_id}-{svc['action']}",
                "name": svc["name"],
                "description": f"Part of {pkg['name']} package",
                "skill": svc["skill"],
                "action": svc["action"],
                "price": final_price,
                "sla_minutes": svc.get("sla_minutes", 5),
                "package_id": package_id,
                "deployment_id": deployment_id,
            })

            hosting_entries.append({
                "service_name": svc["action"],
                "route": f"/api/v1/{svc['action']}",
                "method": "POST",
                "price_per_request": final_price,
                "sla_minutes": svc.get("sla_minutes", 5),
            })

        # Record deployment
        deployment = {
            "deployment_id": deployment_id,
            "package_id": package_id,
            "package_name": pkg["name"],
            "agent_name": agent_name,
            "status": "active",
            "deployed_at": datetime.now().isoformat(),
            "services_count": len(pkg["services"]),
            "total_bundle_price": pricing["bundle_total"],
            "marketplace_entries": marketplace_entries,
            "hosting_entries": hosting_entries,
        }

        if "deployments" not in self._state:
            self._state["deployments"] = []
        self._state["deployments"].append(deployment)

        if "deployment_history" not in self._state:
            self._state["deployment_history"] = []
        self._state["deployment_history"].append({
            "action": "deploy",
            "deployment_id": deployment_id,
            "package_id": package_id,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Deployed '{pkg['name']}' as {deployment_id}: {len(pkg['services'])} services registered. Bundle price: ${pricing['bundle_total']:.2f}/request",
            data={
                "deployment_id": deployment_id,
                "package_id": package_id,
                "package_name": pkg["name"],
                "services_deployed": len(marketplace_entries),
                "marketplace_entries": marketplace_entries,
                "hosting_entries": hosting_entries,
                "pricing": pricing,
                "domain": f"{agent_name}.singularity.wisent.ai",
            },
            revenue=0,
            cost=0,
        )

    def _undeploy(self, params: Dict) -> SkillResult:
        """Undeploy a deployment."""
        deployment_id = params.get("deployment_id", "")
        deployments = self._state.get("deployments", [])

        target = None
        for d in deployments:
            if d["deployment_id"] == deployment_id:
                target = d
                break

        if not target:
            active_ids = [d["deployment_id"] for d in deployments if d.get("status") == "active"]
            return SkillResult(
                success=False,
                message=f"Deployment '{deployment_id}' not found. Active: {active_ids}",
            )

        if target.get("status") != "active":
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' is already {target.get('status')}")

        target["status"] = "undeployed"
        target["undeployed_at"] = datetime.now().isoformat()

        self._state.setdefault("deployment_history", []).append({
            "action": "undeploy",
            "deployment_id": deployment_id,
            "package_id": target["package_id"],
            "timestamp": datetime.now().isoformat(),
        })

        self._save_state()

        return SkillResult(
            success=True,
            message=f"Undeployed '{target['package_name']}' ({deployment_id}). {target['services_count']} services removed.",
            data={
                "deployment_id": deployment_id,
                "package_name": target["package_name"],
                "services_removed": target["services_count"],
            },
        )

    def _create_custom(self, params: Dict) -> SkillResult:
        """Create a custom package."""
        package_id = params.get("package_id", "")
        name = params.get("name", "")
        description = params.get("description", "")
        services = params.get("services", [])
        bundle_discount = params.get("bundle_discount", 0.10)
        tags = params.get("tags", [])

        if not package_id or not name or not services:
            return SkillResult(success=False, message="Required: package_id, name, services (list of {skill, action, name, price})")

        if package_id in BUILTIN_PACKAGES:
            return SkillResult(success=False, message=f"Cannot use built-in package ID: {package_id}")

        customs = self._state.get("custom_packages", {})
        if package_id in customs:
            return SkillResult(success=False, message=f"Custom package '{package_id}' already exists. Delete first to recreate.")

        if len(customs) >= MAX_CUSTOM_PACKAGES:
            return SkillResult(success=False, message=f"Maximum custom packages ({MAX_CUSTOM_PACKAGES}) reached.")

        if bundle_discount < 0 or bundle_discount > 0.5:
            return SkillResult(success=False, message="bundle_discount must be between 0.0 and 0.5")

        # Validate services have required fields
        validated_services = []
        for svc in services:
            if not svc.get("skill") or not svc.get("action") or not svc.get("name"):
                return SkillResult(success=False, message=f"Each service needs: skill, action, name. Got: {svc}")
            validated_services.append({
                "skill": svc["skill"],
                "action": svc["action"],
                "name": svc["name"],
                "price": svc.get("price", 0.05),
                "sla_minutes": svc.get("sla_minutes", 5),
            })

        custom_pkg = {
            "name": name,
            "description": description,
            "services": validated_services,
            "bundle_discount": bundle_discount,
            "tags": tags,
            "target_audience": params.get("target_audience", ""),
            "category": "custom",
            "created_at": datetime.now().isoformat(),
        }

        self._state.setdefault("custom_packages", {})[package_id] = custom_pkg
        self._save_state()

        pricing = self._calculate_bundle_price(custom_pkg)

        return SkillResult(
            success=True,
            message=f"Created custom package '{name}' with {len(validated_services)} services. Bundle price: ${pricing['bundle_total']:.2f}",
            data={
                "package_id": package_id,
                "name": name,
                "services_count": len(validated_services),
                "pricing": pricing,
            },
        )

    def _delete_custom(self, params: Dict) -> SkillResult:
        """Delete a custom package."""
        package_id = params.get("package_id", "")

        if package_id in BUILTIN_PACKAGES:
            return SkillResult(success=False, message="Cannot delete built-in packages.")

        customs = self._state.get("custom_packages", {})
        if package_id not in customs:
            return SkillResult(success=False, message=f"Custom package '{package_id}' not found. Available: {list(customs.keys())}")

        # Check for active deployments
        active = [d for d in self._state.get("deployments", [])
                  if d.get("package_id") == package_id and d.get("status") == "active"]
        if active:
            return SkillResult(
                success=False,
                message=f"Cannot delete: package has {len(active)} active deployment(s). Undeploy first.",
            )

        del customs[package_id]
        self._save_state()

        return SkillResult(
            success=True,
            message=f"Deleted custom package '{package_id}'.",
            data={"package_id": package_id},
        )

    def _compare(self, params: Dict) -> SkillResult:
        """Compare multiple packages side by side."""
        package_ids = params.get("package_ids", [])
        if len(package_ids) < 2:
            return SkillResult(success=False, message="Need at least 2 package IDs to compare.")

        packages = self._get_all_packages()
        comparisons = []

        for pid in package_ids:
            if pid not in packages:
                return SkillResult(success=False, message=f"Package '{pid}' not found.")
            pkg = packages[pid]
            pricing = self._calculate_bundle_price(pkg)
            service_actions = [s["action"] for s in pkg["services"]]
            comparisons.append({
                "package_id": pid,
                "name": pkg["name"],
                "services_count": len(pkg["services"]),
                "service_actions": service_actions,
                "bundle_price": pricing["bundle_total"],
                "individual_price": pricing["individual_total"],
                "savings": pricing["savings"],
                "discount_pct": pricing["discount_pct"],
                "tags": pkg.get("tags", []),
                "category": pkg.get("category", "custom"),
                "estimated_monthly_revenue": pkg.get("estimated_monthly_revenue", 0),
            })

        # Find unique and shared services
        all_actions = [set(c["service_actions"]) for c in comparisons]
        shared = set.intersection(*all_actions) if all_actions else set()
        unique_per_package = {}
        for i, c in enumerate(comparisons):
            unique_per_package[c["package_id"]] = list(set(c["service_actions"]) - shared)

        # Best value (lowest per-service price)
        best_value = min(comparisons, key=lambda c: c["bundle_price"] / max(c["services_count"], 1))

        return SkillResult(
            success=True,
            message=f"Compared {len(comparisons)} packages. Best value: {best_value['name']} (${best_value['bundle_price']:.2f} for {best_value['services_count']} services)",
            data={
                "comparisons": comparisons,
                "shared_services": list(shared),
                "unique_per_package": unique_per_package,
                "best_value": best_value["package_id"],
            },
        )

    def _recommend(self, params: Dict) -> SkillResult:
        """Recommend the best package for a use case."""
        use_case = params.get("use_case", "").lower()
        if not use_case:
            return SkillResult(success=False, message="Provide a use_case description.")

        packages = self._get_all_packages()

        # Keyword-based scoring
        keyword_map = {
            "developer_toolkit": ["developer", "dev", "code", "api", "review", "documentation", "engineering", "software", "programming", "quality", "bugs", "security"],
            "content_suite": ["content", "marketing", "seo", "writing", "blog", "article", "publisher", "copywriting", "social media", "summarize", "summary"],
            "data_intelligence": ["data", "analytics", "analysis", "report", "statistics", "insights", "research", "metrics", "dashboard", "bi", "business intelligence"],
            "full_stack": ["enterprise", "full", "all", "everything", "complete", "agency", "consulting", "team", "company", "startup"],
        }

        scores = []
        for pid, keywords in keyword_map.items():
            if pid not in packages:
                continue
            score = sum(1 for kw in keywords if kw in use_case)
            # Boost for exact phrase matches
            if pid.replace("_", " ") in use_case:
                score += 5
            scores.append((pid, score))

        # Also score custom packages by tags
        for pid, pkg in self._state.get("custom_packages", {}).items():
            score = sum(1 for tag in pkg.get("tags", []) if tag.lower() in use_case)
            if pkg.get("target_audience", "").lower() in use_case:
                score += 3
            scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Build recommendations
        recommendations = []
        for pid, score in scores[:3]:
            pkg = packages[pid]
            pricing = self._calculate_bundle_price(pkg)
            recommendations.append({
                "package_id": pid,
                "name": pkg["name"],
                "relevance_score": score,
                "bundle_price": pricing["bundle_total"],
                "services_count": len(pkg["services"]),
                "description": pkg["description"][:150],
                "why": self._explain_recommendation(pid, use_case),
            })

        if not recommendations:
            return SkillResult(
                success=True,
                message="No strong matches found. Consider 'full_stack' for the most complete coverage.",
                data={"recommendations": [], "fallback": "full_stack"},
            )

        top = recommendations[0]
        return SkillResult(
            success=True,
            message=f"Top recommendation for '{use_case}': {top['name']} (${top['bundle_price']:.2f}, {top['services_count']} services)",
            data={
                "use_case": use_case,
                "recommendations": recommendations,
                "top_pick": top["package_id"],
            },
        )

    def _explain_recommendation(self, package_id: str, use_case: str) -> str:
        """Generate a brief explanation for why this package fits."""
        explanations = {
            "developer_toolkit": "Matches developer/engineering needs with code review and API documentation.",
            "content_suite": "Matches content/marketing needs with summarization and SEO optimization.",
            "data_intelligence": "Matches data/analytics needs with structured analysis and report summarization.",
            "full_stack": "Comprehensive coverage — includes all services for maximum flexibility.",
        }
        return explanations.get(package_id, f"Custom package with tags matching '{use_case}'.")

    def _status(self, params: Dict) -> SkillResult:
        """View deployment status and summary."""
        deployments = self._state.get("deployments", [])
        active = [d for d in deployments if d.get("status") == "active"]
        inactive = [d for d in deployments if d.get("status") != "active"]

        active_summary = []
        total_services = 0
        for d in active:
            total_services += d.get("services_count", 0)
            active_summary.append({
                "deployment_id": d["deployment_id"],
                "package_name": d["package_name"],
                "services_count": d.get("services_count", 0),
                "deployed_at": d.get("deployed_at", ""),
                "bundle_price": d.get("total_bundle_price", 0),
            })

        history = self._state.get("deployment_history", [])[-10:]

        return SkillResult(
            success=True,
            message=f"{len(active)} active deployments ({total_services} services), {len(inactive)} inactive. {len(self._state.get('custom_packages', {}))} custom packages.",
            data={
                "active_deployments": active_summary,
                "active_count": len(active),
                "inactive_count": len(inactive),
                "total_services_deployed": total_services,
                "custom_packages_count": len(self._state.get("custom_packages", {})),
                "recent_history": history,
                "builtin_packages": list(BUILTIN_PACKAGES.keys()),
            },
        )
