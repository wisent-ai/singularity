#!/usr/bin/env python3
"""
WorkflowTemplateLibrarySkill - Pre-built workflow templates for common integrations.

Provides a library of reusable, parameterizable workflow templates that agents
can instantiate for common automation patterns. Instead of building workflows
from scratch, agents browse the template library, pick a template, fill in
parameters, and get a working workflow instantly.

Built-in template categories:
- **CI/CD**: GitHub PR review, deploy on merge, test runner
- **Billing**: Stripe payment processing, invoice generation, usage alerts
- **Monitoring**: Health checks, alerting, incident response
- **Onboarding**: Customer welcome, access provisioning, tutorial delivery
- **Content**: Blog post pipeline, social media scheduling, email campaigns
- **DevOps**: Log analysis, backup verification, scaling decisions

Each template specifies:
- Required and optional parameters with defaults
- Multi-step workflow with data flow between steps
- Which skills are needed (dependencies)
- Estimated cost and execution time
- Success/failure hooks

Actions:
1. BROWSE     - List available templates by category
2. GET        - Get full template details with parameter docs
3. INSTANTIATE - Create a workflow from a template with custom parameters
4. REGISTER   - Register a custom template from an existing workflow
5. SEARCH     - Search templates by keyword
6. RATE       - Rate a template based on experience
7. POPULAR    - Get most-used templates
8. EXPORT     - Export a template as a standalone workflow definition

Pillars served:
- Revenue: Customers can quickly deploy automations without custom development
- Self-Improvement: Best practices encoded as reusable templates
- Replication: New agents bootstrap with proven workflow patterns
- Goal Setting: Templates suggest what automations are possible
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
TEMPLATES_FILE = DATA_DIR / "workflow_templates.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_data(path: Path = None) -> Dict:
    p = path or TEMPLATES_FILE
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return _default_data()


def _save_data(data: Dict, path: Path = None):
    p = path or TEMPLATES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, str(p))


def _default_data() -> Dict:
    return {
        "templates": _builtin_templates(),
        "custom_templates": {},
        "instantiations": [],
        "ratings": {},
        "stats": {
            "total_instantiations": 0,
            "total_custom_templates": 0,
            "total_ratings": 0,
        },
    }


def _builtin_templates() -> Dict[str, Dict]:
    """Pre-built workflow templates organized by category."""
    return {
        # ─── CI/CD Templates ───────────────────────────────────────
        "github_pr_review": {
            "id": "github_pr_review",
            "name": "GitHub PR Auto-Review",
            "category": "ci_cd",
            "description": "Automatically review pull requests: run code review, post comments, and track quality metrics.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "repo": {"type": "string", "required": True, "description": "Repository (owner/repo)"},
                "review_depth": {"type": "string", "required": False, "default": "standard", "description": "Review depth: quick, standard, thorough"},
                "auto_approve": {"type": "boolean", "required": False, "default": False, "description": "Auto-approve if no issues found"},
                "focus_areas": {"type": "list", "required": False, "default": [], "description": "Areas to focus on: security, performance, style"},
            },
            "steps": [
                {"skill": "github", "action": "get_pr", "params_from": {"repo": "param.repo"}},
                {"skill": "code_review", "action": "review", "params_from": {"code": "step.0.diff", "depth": "param.review_depth"}},
                {"skill": "github", "action": "post_comment", "params_from": {"repo": "param.repo", "body": "step.1.review"}},
            ],
            "required_skills": ["github", "code_review"],
            "estimated_cost": 0.05,
            "estimated_duration_seconds": 60,
            "tags": ["github", "code-review", "automation", "ci"],
            "use_count": 0,
        },

        "deploy_on_merge": {
            "id": "deploy_on_merge",
            "name": "Deploy on Merge",
            "category": "ci_cd",
            "description": "Trigger deployment when a PR is merged to main branch.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "repo": {"type": "string", "required": True, "description": "Repository (owner/repo)"},
                "deploy_target": {"type": "string", "required": True, "description": "Deployment target: fly.io, railway, vercel"},
                "branch": {"type": "string", "required": False, "default": "main", "description": "Branch to deploy from"},
                "notify_channel": {"type": "string", "required": False, "default": "", "description": "Notification channel for deploy status"},
            },
            "steps": [
                {"skill": "github", "action": "check_merge", "params_from": {"repo": "param.repo", "branch": "param.branch"}},
                {"skill": "deployment", "action": "deploy", "params_from": {"target": "param.deploy_target"}},
                {"skill": "notification", "action": "send", "params_from": {"channel": "param.notify_channel", "message": "step.1.status"}},
            ],
            "required_skills": ["github", "deployment", "notification"],
            "estimated_cost": 0.10,
            "estimated_duration_seconds": 300,
            "tags": ["deploy", "ci-cd", "github", "automation"],
            "use_count": 0,
        },

        # ─── Billing Templates ─────────────────────────────────────
        "stripe_payment_flow": {
            "id": "stripe_payment_flow",
            "name": "Stripe Payment Processing",
            "category": "billing",
            "description": "Process Stripe payments: verify, register customer, provision access, send receipt.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "customer_email": {"type": "string", "required": True, "description": "Customer email address"},
                "amount": {"type": "float", "required": True, "description": "Payment amount"},
                "product": {"type": "string", "required": True, "description": "Product/service being purchased"},
                "tier": {"type": "string", "required": False, "default": "basic", "description": "Service tier: free, basic, premium"},
            },
            "steps": [
                {"skill": "payment", "action": "verify", "params_from": {"email": "param.customer_email", "amount": "param.amount"}},
                {"skill": "usage_tracking", "action": "register", "params_from": {"customer_id": "param.customer_email", "tier": "param.tier"}},
                {"skill": "notification", "action": "send", "params_from": {"to": "param.customer_email", "template": "welcome"}},
            ],
            "required_skills": ["payment", "usage_tracking", "notification"],
            "estimated_cost": 0.02,
            "estimated_duration_seconds": 30,
            "tags": ["stripe", "payment", "billing", "customer"],
            "use_count": 0,
        },

        "usage_alert": {
            "id": "usage_alert",
            "name": "Usage Threshold Alert",
            "category": "billing",
            "description": "Monitor customer API usage and alert when approaching limits.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "customer_id": {"type": "string", "required": True, "description": "Customer ID to monitor"},
                "threshold_pct": {"type": "float", "required": False, "default": 80.0, "description": "Alert at this usage percentage"},
                "alert_method": {"type": "string", "required": False, "default": "email", "description": "Alert method: email, webhook, slack"},
            },
            "steps": [
                {"skill": "usage_tracking", "action": "report", "params_from": {"customer_id": "param.customer_id"}},
                {"skill": "notification", "action": "send", "params_from": {"to": "param.customer_id", "message": "step.0.summary"}},
            ],
            "required_skills": ["usage_tracking", "notification"],
            "estimated_cost": 0.01,
            "estimated_duration_seconds": 10,
            "tags": ["usage", "alert", "billing", "monitoring"],
            "use_count": 0,
        },

        # ─── Monitoring Templates ──────────────────────────────────
        "health_check_pipeline": {
            "id": "health_check_pipeline",
            "name": "Service Health Check",
            "category": "monitoring",
            "description": "Run health checks on deployed services, diagnose issues, and auto-heal if possible.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "service_url": {"type": "string", "required": True, "description": "URL of service to check"},
                "check_interval_min": {"type": "integer", "required": False, "default": 5, "description": "Check interval in minutes"},
                "auto_heal": {"type": "boolean", "required": False, "default": True, "description": "Attempt automatic healing"},
                "escalate_after": {"type": "integer", "required": False, "default": 3, "description": "Escalate after N failures"},
            },
            "steps": [
                {"skill": "request", "action": "get", "params_from": {"url": "param.service_url"}},
                {"skill": "self_healing", "action": "diagnose", "params_from": {"target": "param.service_url", "symptoms": "step.0.errors"}},
                {"skill": "notification", "action": "send", "params_from": {"message": "step.1.diagnosis"}},
            ],
            "required_skills": ["request", "self_healing", "notification"],
            "estimated_cost": 0.01,
            "estimated_duration_seconds": 15,
            "tags": ["health", "monitoring", "auto-heal", "uptime"],
            "use_count": 0,
        },

        "incident_response": {
            "id": "incident_response",
            "name": "Automated Incident Response",
            "category": "monitoring",
            "description": "Respond to incidents: triage severity, notify stakeholders, attempt remediation.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "incident_type": {"type": "string", "required": True, "description": "Type: outage, degradation, security, data_loss"},
                "affected_service": {"type": "string", "required": True, "description": "Service affected"},
                "severity": {"type": "string", "required": False, "default": "medium", "description": "Severity: low, medium, high, critical"},
                "notify_list": {"type": "list", "required": False, "default": [], "description": "Agent IDs to notify"},
            },
            "steps": [
                {"skill": "self_healing", "action": "scan", "params_from": {"target": "param.affected_service"}},
                {"skill": "decision_log", "action": "log", "params_from": {"decision": "step.0.recommendation", "context": "param.incident_type"}},
                {"skill": "messaging", "action": "broadcast", "params_from": {"content": "step.0.summary"}},
            ],
            "required_skills": ["self_healing", "decision_log", "messaging"],
            "estimated_cost": 0.03,
            "estimated_duration_seconds": 45,
            "tags": ["incident", "response", "remediation", "alerting"],
            "use_count": 0,
        },

        # ─── Onboarding Templates ─────────────────────────────────
        "customer_onboarding": {
            "id": "customer_onboarding",
            "name": "Customer Onboarding Flow",
            "category": "onboarding",
            "description": "Complete customer onboarding: register, provision API key, send welcome docs, track activation.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "customer_name": {"type": "string", "required": True, "description": "Customer name"},
                "customer_email": {"type": "string", "required": True, "description": "Customer email"},
                "tier": {"type": "string", "required": False, "default": "free", "description": "Service tier: free, basic, premium"},
                "services": {"type": "list", "required": False, "default": ["code_review"], "description": "Services to enable"},
            },
            "steps": [
                {"skill": "usage_tracking", "action": "register", "params_from": {"customer_id": "param.customer_email", "tier": "param.tier"}},
                {"skill": "api_gateway", "action": "create_key", "params_from": {"customer_id": "param.customer_email", "tier": "param.tier"}},
                {"skill": "notification", "action": "send", "params_from": {"to": "param.customer_email", "template": "onboarding"}},
            ],
            "required_skills": ["usage_tracking", "api_gateway", "notification"],
            "estimated_cost": 0.01,
            "estimated_duration_seconds": 15,
            "tags": ["onboarding", "customer", "welcome", "api-key"],
            "use_count": 0,
        },

        # ─── Content Templates ─────────────────────────────────────
        "content_pipeline": {
            "id": "content_pipeline",
            "name": "Content Generation Pipeline",
            "category": "content",
            "description": "Generate, review, and publish content: draft, SEO optimize, schedule.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "topic": {"type": "string", "required": True, "description": "Content topic"},
                "format": {"type": "string", "required": False, "default": "blog_post", "description": "Format: blog_post, social_media, email, docs"},
                "tone": {"type": "string", "required": False, "default": "professional", "description": "Tone: professional, casual, technical"},
                "target_length": {"type": "integer", "required": False, "default": 500, "description": "Target word count"},
            },
            "steps": [
                {"skill": "revenue_services", "action": "summarize", "params_from": {"text": "param.topic", "max_length": "param.target_length"}},
                {"skill": "revenue_services", "action": "seo_audit", "params_from": {"content": "step.0.summary"}},
                {"skill": "outcome_tracker", "action": "log", "params_from": {"outcome": "content_generated", "details": "step.1.suggestions"}},
            ],
            "required_skills": ["revenue_services", "outcome_tracker"],
            "estimated_cost": 0.03,
            "estimated_duration_seconds": 30,
            "tags": ["content", "blog", "seo", "generation"],
            "use_count": 0,
        },

        # ─── DevOps Templates ──────────────────────────────────────
        "scaling_decision": {
            "id": "scaling_decision",
            "name": "Auto-Scaling Decision",
            "category": "devops",
            "description": "Analyze load metrics and make consensus-based scaling decisions.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "service_id": {"type": "string", "required": True, "description": "Service to evaluate scaling for"},
                "cpu_threshold": {"type": "float", "required": False, "default": 80.0, "description": "CPU threshold for scale-up trigger"},
                "min_instances": {"type": "integer", "required": False, "default": 1, "description": "Minimum instance count"},
                "max_instances": {"type": "integer", "required": False, "default": 5, "description": "Maximum instance count"},
            },
            "steps": [
                {"skill": "performance", "action": "snapshot", "params_from": {"target": "param.service_id"}},
                {"skill": "consensus_protocol", "action": "propose", "params_from": {"title": "Scaling decision", "description": "step.0.metrics"}},
                {"skill": "deployment", "action": "scale", "params_from": {"service": "param.service_id", "instances": "step.1.outcome"}},
            ],
            "required_skills": ["performance", "consensus_protocol", "deployment"],
            "estimated_cost": 0.05,
            "estimated_duration_seconds": 120,
            "tags": ["scaling", "devops", "auto-scale", "performance"],
            "use_count": 0,
        },

        "backup_verification": {
            "id": "backup_verification",
            "name": "Backup Verification",
            "category": "devops",
            "description": "Verify data backups exist, are recent, and can be restored.",
            "version": "1.0.0",
            "author": "system",
            "parameters": {
                "backup_path": {"type": "string", "required": True, "description": "Path or URL to backup location"},
                "max_age_hours": {"type": "integer", "required": False, "default": 24, "description": "Maximum backup age in hours"},
                "verify_restore": {"type": "boolean", "required": False, "default": False, "description": "Attempt a test restore"},
            },
            "steps": [
                {"skill": "filesystem", "action": "list", "params_from": {"path": "param.backup_path"}},
                {"skill": "diagnostics", "action": "check", "params_from": {"target": "param.backup_path", "max_age": "param.max_age_hours"}},
                {"skill": "outcome_tracker", "action": "log", "params_from": {"outcome": "backup_check", "details": "step.1.result"}},
            ],
            "required_skills": ["filesystem", "diagnostics", "outcome_tracker"],
            "estimated_cost": 0.01,
            "estimated_duration_seconds": 30,
            "tags": ["backup", "verification", "devops", "data-safety"],
            "use_count": 0,
        },
    }


# Unique categories from built-in templates
CATEGORIES = ["ci_cd", "billing", "monitoring", "onboarding", "content", "devops"]

MAX_CUSTOM_TEMPLATES = 200
MAX_INSTANTIATIONS = 1000


class WorkflowTemplateLibrarySkill(Skill):
    """Pre-built workflow templates for common integrations."""

    def __init__(self):
        self._data_path = TEMPLATES_FILE

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="workflow_templates",
            name="Workflow Template Library",
            version="1.0.0",
            category="automation",
            description="Pre-built workflow templates for common integrations",
            actions=[
                SkillAction(
                    name="browse",
                    description="List available templates by category",
                    parameters={
                        "category": {"type": "string", "required": False, "description": "Filter by category: ci_cd, billing, monitoring, onboarding, content, devops"},
                        "limit": {"type": "integer", "required": False, "description": "Max results (default: 20)"},
                    },
                ),
                SkillAction(
                    name="get",
                    description="Get full template details with parameter docs",
                    parameters={
                        "template_id": {"type": "string", "required": True, "description": "Template ID"},
                    },
                ),
                SkillAction(
                    name="instantiate",
                    description="Create a workflow from a template with custom parameters",
                    parameters={
                        "template_id": {"type": "string", "required": True, "description": "Template ID to instantiate"},
                        "params": {"type": "dict", "required": True, "description": "Parameter values for the template"},
                        "name": {"type": "string", "required": False, "description": "Custom name for this instance"},
                    },
                ),
                SkillAction(
                    name="register",
                    description="Register a custom template",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Template name"},
                        "category": {"type": "string", "required": True, "description": "Category"},
                        "description": {"type": "string", "required": True, "description": "What this template does"},
                        "parameters": {"type": "dict", "required": True, "description": "Parameter definitions"},
                        "steps": {"type": "list", "required": True, "description": "Workflow steps"},
                        "required_skills": {"type": "list", "required": False, "description": "Skills needed"},
                        "tags": {"type": "list", "required": False, "description": "Search tags"},
                    },
                ),
                SkillAction(
                    name="search",
                    description="Search templates by keyword",
                    parameters={
                        "query": {"type": "string", "required": True, "description": "Search query"},
                        "limit": {"type": "integer", "required": False, "description": "Max results (default: 10)"},
                    },
                ),
                SkillAction(
                    name="rate",
                    description="Rate a template based on experience",
                    parameters={
                        "template_id": {"type": "string", "required": True, "description": "Template ID"},
                        "rating": {"type": "integer", "required": True, "description": "Rating 1-5"},
                        "agent_id": {"type": "string", "required": True, "description": "Agent rating"},
                        "comment": {"type": "string", "required": False, "description": "Optional comment"},
                    },
                ),
                SkillAction(
                    name="popular",
                    description="Get most-used templates",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max results (default: 5)"},
                    },
                ),
                SkillAction(
                    name="export",
                    description="Export a template as a standalone workflow definition",
                    parameters={
                        "template_id": {"type": "string", "required": True, "description": "Template ID"},
                        "params": {"type": "dict", "required": False, "description": "Fill in parameters for export"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        data = _load_data(self._data_path)

        handlers = {
            "browse": self._browse,
            "get": self._get,
            "instantiate": self._instantiate,
            "register": self._register,
            "search": self._search,
            "rate": self._rate,
            "popular": self._popular,
            "export": self._export,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")

        result = handler(data, params)
        _save_data(data, self._data_path)
        return result

    def _all_templates(self, data: Dict) -> Dict[str, Dict]:
        """Merge built-in and custom templates."""
        merged = dict(data.get("templates", {}))
        merged.update(data.get("custom_templates", {}))
        return merged

    def _browse(self, data: Dict, params: Dict) -> SkillResult:
        category = params.get("category")
        limit = min(int(params.get("limit", 20)), 50)

        templates = self._all_templates(data)

        if category:
            templates = {k: v for k, v in templates.items() if v.get("category") == category}

        summaries = []
        for tid, t in sorted(templates.items(), key=lambda x: x[1].get("use_count", 0), reverse=True):
            summaries.append({
                "id": tid,
                "name": t["name"],
                "category": t["category"],
                "description": t["description"][:100],
                "tags": t.get("tags", []),
                "use_count": t.get("use_count", 0),
                "avg_rating": self._avg_rating(data, tid),
                "required_skills": t.get("required_skills", []),
            })

        summaries = summaries[:limit]

        return SkillResult(
            success=True,
            message=f"Found {len(summaries)} templates" + (f" in category '{category}'" if category else ""),
            data={
                "templates": summaries,
                "categories": CATEGORIES,
                "total_templates": len(self._all_templates(data)),
            },
        )

    def _get(self, data: Dict, params: Dict) -> SkillResult:
        template_id = params.get("template_id", "").strip()
        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        templates = self._all_templates(data)
        template = templates.get(template_id)
        if not template:
            return SkillResult(success=False, message=f"Template '{template_id}' not found")

        return SkillResult(
            success=True,
            message=f"Template: {template['name']}",
            data={
                "template": template,
                "avg_rating": self._avg_rating(data, template_id),
                "rating_count": len(data.get("ratings", {}).get(template_id, [])),
            },
        )

    def _instantiate(self, data: Dict, params: Dict) -> SkillResult:
        template_id = params.get("template_id", "").strip()
        user_params = params.get("params", {})

        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        templates = self._all_templates(data)
        template = templates.get(template_id)
        if not template:
            return SkillResult(success=False, message=f"Template '{template_id}' not found")

        # Validate required parameters
        missing = []
        for pname, pdef in template.get("parameters", {}).items():
            if pdef.get("required", False) and pname not in user_params:
                missing.append(pname)

        if missing:
            return SkillResult(
                success=False,
                message=f"Missing required parameters: {missing}",
                data={"missing": missing, "parameters": template["parameters"]},
            )

        # Build resolved parameters (fill defaults)
        resolved_params = {}
        for pname, pdef in template.get("parameters", {}).items():
            if pname in user_params:
                resolved_params[pname] = user_params[pname]
            elif "default" in pdef:
                resolved_params[pname] = pdef["default"]

        # Resolve steps with parameters
        resolved_steps = []
        for step in template.get("steps", []):
            resolved_step = {
                "skill": step["skill"],
                "action": step["action"],
                "params": {},
            }
            for param_key, source in step.get("params_from", {}).items():
                if isinstance(source, str) and source.startswith("param."):
                    pkey = source[6:]
                    resolved_step["params"][param_key] = resolved_params.get(pkey)
                else:
                    # References to previous steps or literals - keep as-is
                    resolved_step["params"][param_key] = source
            resolved_steps.append(resolved_step)

        instance_id = f"wf_{uuid.uuid4().hex[:12]}"
        instance = {
            "id": instance_id,
            "template_id": template_id,
            "template_name": template["name"],
            "name": params.get("name", f"{template['name']} instance"),
            "parameters": resolved_params,
            "steps": resolved_steps,
            "created_at": _now_iso(),
            "status": "ready",
        }

        data.setdefault("instantiations", []).append(instance)
        if len(data["instantiations"]) > MAX_INSTANTIATIONS:
            data["instantiations"] = data["instantiations"][-MAX_INSTANTIATIONS:]

        # Increment use count
        if template_id in data.get("templates", {}):
            data["templates"][template_id]["use_count"] = data["templates"][template_id].get("use_count", 0) + 1
        elif template_id in data.get("custom_templates", {}):
            data["custom_templates"][template_id]["use_count"] = data["custom_templates"][template_id].get("use_count", 0) + 1

        data["stats"]["total_instantiations"] += 1

        return SkillResult(
            success=True,
            message=f"Workflow instantiated from '{template['name']}': {instance_id}",
            data={
                "instance_id": instance_id,
                "instance": instance,
                "estimated_cost": template.get("estimated_cost", 0),
                "estimated_duration_seconds": template.get("estimated_duration_seconds", 0),
            },
        )

    def _register(self, data: Dict, params: Dict) -> SkillResult:
        name = params.get("name", "").strip()
        category = params.get("category", "").strip()
        description = params.get("description", "").strip()
        parameters = params.get("parameters", {})
        steps = params.get("steps", [])

        if not name or not category or not description:
            return SkillResult(success=False, message="name, category, and description are required")

        if not steps:
            return SkillResult(success=False, message="At least one step is required")

        if len(data.get("custom_templates", {})) >= MAX_CUSTOM_TEMPLATES:
            return SkillResult(success=False, message=f"Maximum custom templates ({MAX_CUSTOM_TEMPLATES}) reached")

        template_id = f"custom_{name.lower().replace(' ', '_')[:30]}_{uuid.uuid4().hex[:6]}"

        template = {
            "id": template_id,
            "name": name,
            "category": category,
            "description": description,
            "version": "1.0.0",
            "author": "agent",
            "parameters": parameters,
            "steps": steps,
            "required_skills": params.get("required_skills", []),
            "estimated_cost": 0,
            "estimated_duration_seconds": 0,
            "tags": params.get("tags", []),
            "use_count": 0,
            "created_at": _now_iso(),
        }

        data.setdefault("custom_templates", {})[template_id] = template
        data["stats"]["total_custom_templates"] += 1

        return SkillResult(
            success=True,
            message=f"Custom template '{name}' registered: {template_id}",
            data={"template_id": template_id, "template": template},
        )

    def _search(self, data: Dict, params: Dict) -> SkillResult:
        query = params.get("query", "").strip().lower()
        limit = min(int(params.get("limit", 10)), 50)

        if not query:
            return SkillResult(success=False, message="query is required")

        templates = self._all_templates(data)
        results = []

        for tid, t in templates.items():
            score = 0
            searchable = f"{t['name']} {t['description']} {' '.join(t.get('tags', []))} {t['category']}".lower()

            for word in query.split():
                if word in searchable:
                    score += 1
                if word in t["name"].lower():
                    score += 2  # Name matches worth more
                if word in t.get("tags", []):
                    score += 1.5

            if score > 0:
                results.append({
                    "id": tid,
                    "name": t["name"],
                    "category": t["category"],
                    "description": t["description"][:100],
                    "tags": t.get("tags", []),
                    "score": score,
                    "use_count": t.get("use_count", 0),
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:limit]

        return SkillResult(
            success=True,
            message=f"Found {len(results)} templates matching '{query}'",
            data={"results": results, "query": query},
        )

    def _rate(self, data: Dict, params: Dict) -> SkillResult:
        template_id = params.get("template_id", "").strip()
        rating = int(params.get("rating", 0))
        agent_id = params.get("agent_id", "").strip()

        if not template_id or not agent_id:
            return SkillResult(success=False, message="template_id and agent_id are required")

        if rating < 1 or rating > 5:
            return SkillResult(success=False, message="Rating must be 1-5")

        templates = self._all_templates(data)
        if template_id not in templates:
            return SkillResult(success=False, message=f"Template '{template_id}' not found")

        data.setdefault("ratings", {}).setdefault(template_id, [])

        # Replace existing rating from same agent
        data["ratings"][template_id] = [
            r for r in data["ratings"][template_id] if r.get("agent_id") != agent_id
        ]

        data["ratings"][template_id].append({
            "agent_id": agent_id,
            "rating": rating,
            "comment": params.get("comment", ""),
            "rated_at": _now_iso(),
        })

        data["stats"]["total_ratings"] += 1

        return SkillResult(
            success=True,
            message=f"Rated template '{templates[template_id]['name']}': {rating}/5",
            data={
                "template_id": template_id,
                "rating": rating,
                "avg_rating": self._avg_rating(data, template_id),
            },
        )

    def _popular(self, data: Dict, params: Dict) -> SkillResult:
        limit = min(int(params.get("limit", 5)), 20)

        templates = self._all_templates(data)
        ranked = sorted(
            templates.values(),
            key=lambda t: t.get("use_count", 0),
            reverse=True,
        )[:limit]

        return SkillResult(
            success=True,
            message=f"Top {len(ranked)} most popular templates",
            data={
                "popular": [
                    {
                        "id": t["id"],
                        "name": t["name"],
                        "category": t["category"],
                        "use_count": t.get("use_count", 0),
                        "avg_rating": self._avg_rating(data, t["id"]),
                    }
                    for t in ranked
                ],
            },
        )

    def _export(self, data: Dict, params: Dict) -> SkillResult:
        template_id = params.get("template_id", "").strip()
        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        templates = self._all_templates(data)
        template = templates.get(template_id)
        if not template:
            return SkillResult(success=False, message=f"Template '{template_id}' not found")

        user_params = params.get("params", {})

        # Resolve parameters with defaults
        resolved = {}
        for pname, pdef in template.get("parameters", {}).items():
            if pname in user_params:
                resolved[pname] = user_params[pname]
            elif "default" in pdef:
                resolved[pname] = pdef["default"]
            else:
                resolved[pname] = f"<{pname}>"

        export = {
            "workflow_name": template["name"],
            "source_template": template_id,
            "version": template.get("version", "1.0.0"),
            "description": template["description"],
            "parameters": resolved,
            "steps": template["steps"],
            "required_skills": template.get("required_skills", []),
            "estimated_cost": template.get("estimated_cost", 0),
            "estimated_duration_seconds": template.get("estimated_duration_seconds", 0),
            "exported_at": _now_iso(),
        }

        return SkillResult(
            success=True,
            message=f"Exported template '{template['name']}'",
            data={"export": export},
        )

    def _avg_rating(self, data: Dict, template_id: str) -> float:
        ratings = data.get("ratings", {}).get(template_id, [])
        if not ratings:
            return 0.0
        return round(sum(r["rating"] for r in ratings) / len(ratings), 2)
