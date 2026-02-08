#!/usr/bin/env python3
"""
ConfigTemplateSkill - Agent configuration profiles and specialization templates.

Manages named configuration templates that define how an agent should be set up:
which skills to enable, what model to use, budget limits, and parameter overrides.
Critical for Replication (configure new replicas) and Revenue (specialize for services).

Templates can be:
- Created from the current agent state (snapshot)
- Defined manually with specific settings
- Applied to configure/reconfigure the agent at runtime
- Shared between agents via export/import
- Compared to understand differences between configurations

Built-in templates for common specializations:
- code_reviewer: Focused on code review and analysis
- content_writer: Optimized for content generation
- data_analyst: Configured for data transformation and analysis
- ops_monitor: Set up for infrastructure monitoring and healing
- revenue_agent: Maximized for revenue-generating activities

Pillar: Replication (configure replicas), Revenue (service specialization)
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Skill, SkillAction, SkillManifest, SkillResult

CONFIG_TEMPLATE_FILE = Path(__file__).parent.parent / "data" / "config_templates.json"
MAX_TEMPLATES = 50
MAX_HISTORY = 200


# Built-in specialization templates
BUILTIN_TEMPLATES = {
    "code_reviewer": {
        "name": "Code Reviewer",
        "description": "Specialized for code review, analysis, and quality assurance",
        "skills_enabled": [
            "code_review", "github", "self_testing", "error_recovery",
            "skill_analyzer", "filesystem", "shell",
        ],
        "skills_disabled": [
            "twitter", "email", "crypto", "content", "namecheap",
            "browser", "web_scraper",
        ],
        "parameters": {
            "model_preference": "claude-sonnet",
            "max_budget_per_task": 0.50,
            "focus_areas": ["code_quality", "security", "performance"],
        },
        "category": "revenue",
        "tags": ["development", "quality", "review"],
    },
    "content_writer": {
        "name": "Content Writer",
        "description": "Optimized for content generation, editing, and publishing",
        "skills_enabled": [
            "content", "browser", "web_scraper", "filesystem",
            "email", "vercel",
        ],
        "skills_disabled": [
            "crypto", "deployment", "shell", "replication",
            "code_review", "skill_analyzer",
        ],
        "parameters": {
            "model_preference": "claude-sonnet",
            "max_budget_per_task": 0.30,
            "focus_areas": ["writing", "editing", "seo"],
            "tone": "professional",
        },
        "category": "revenue",
        "tags": ["content", "writing", "marketing"],
    },
    "data_analyst": {
        "name": "Data Analyst",
        "description": "Configured for data transformation, analysis, and reporting",
        "skills_enabled": [
            "data_transform", "filesystem", "shell", "web_scraper",
            "browser", "dashboard",
        ],
        "skills_disabled": [
            "twitter", "email", "crypto", "content", "namecheap",
            "deployment", "replication",
        ],
        "parameters": {
            "model_preference": "claude-sonnet",
            "max_budget_per_task": 0.40,
            "focus_areas": ["analysis", "visualization", "reporting"],
        },
        "category": "revenue",
        "tags": ["data", "analysis", "reporting"],
    },
    "ops_monitor": {
        "name": "Operations Monitor",
        "description": "Set up for infrastructure monitoring, alerting, and healing",
        "skills_enabled": [
            "observability", "health_monitor", "self_healing",
            "incident_response", "alert_incident_bridge", "diagnostics",
            "scheduler", "scheduler_presets", "deployment",
        ],
        "skills_disabled": [
            "twitter", "content", "crypto", "email", "namecheap",
            "browser", "web_scraper",
        ],
        "parameters": {
            "model_preference": "claude-haiku",
            "max_budget_per_task": 0.10,
            "focus_areas": ["uptime", "alerting", "healing"],
            "alert_check_interval_seconds": 60,
        },
        "category": "operations",
        "tags": ["monitoring", "ops", "infrastructure"],
    },
    "revenue_agent": {
        "name": "Revenue Agent",
        "description": "Maximized for revenue-generating service delivery",
        "skills_enabled": [
            "revenue_services", "usage_tracking", "payment",
            "marketplace", "auto_catalog", "api_gateway",
            "service_hosting", "public_deployer", "code_review",
            "content", "data_transform",
        ],
        "skills_disabled": [
            "replication", "prompt_evolution", "self_modify",
            "experiment",
        ],
        "parameters": {
            "model_preference": "claude-sonnet",
            "max_budget_per_task": 1.00,
            "focus_areas": ["service_delivery", "billing", "customer_satisfaction"],
            "auto_bill": True,
        },
        "category": "revenue",
        "tags": ["revenue", "services", "billing"],
    },
}


class ConfigTemplateSkill(Skill):
    """
    Manage agent configuration profiles and specialization templates.

    Enables agents to snapshot their configuration, define specialized
    profiles, apply configurations, and share templates with replicas.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._context = None
        self._ensure_data()

    def set_context(self, context):
        self._context = context

    def _ensure_data(self):
        CONFIG_TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not CONFIG_TEMPLATE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "templates": {},
            "active_template": None,
            "apply_history": [],
            "stats": {
                "total_created": 0,
                "total_applied": 0,
                "total_exported": 0,
                "total_imported": 0,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(CONFIG_TEMPLATE_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        if len(data.get("apply_history", [])) > MAX_HISTORY:
            data["apply_history"] = data["apply_history"][-MAX_HISTORY:]
        CONFIG_TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_TEMPLATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="config_template",
            name="Configuration Template Manager",
            version="1.0.0",
            category="replication",
            description="Manage agent configuration profiles and specialization templates",
            actions=[
                SkillAction(
                    name="list",
                    description="List all available templates (built-in and custom)",
                    parameters={
                        "category": {"type": "string", "required": False,
                                     "description": "Filter by category"},
                        "tag": {"type": "string", "required": False,
                                "description": "Filter by tag"},
                    },
                ),
                SkillAction(
                    name="get",
                    description="Get full details of a specific template",
                    parameters={
                        "template_id": {"type": "string", "required": True,
                                        "description": "Template ID or built-in name"},
                    },
                ),
                SkillAction(
                    name="create",
                    description="Create a new custom configuration template",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Template name"},
                        "description": {"type": "string", "required": False,
                                        "description": "Template description"},
                        "skills_enabled": {"type": "array", "required": False,
                                           "description": "Skills to enable"},
                        "skills_disabled": {"type": "array", "required": False,
                                            "description": "Skills to disable"},
                        "parameters": {"type": "object", "required": False,
                                       "description": "Parameter overrides"},
                        "category": {"type": "string", "required": False,
                                     "description": "Template category"},
                        "tags": {"type": "array", "required": False,
                                 "description": "Tags for filtering"},
                    },
                ),
                SkillAction(
                    name="snapshot",
                    description="Create a template from the current agent configuration",
                    parameters={
                        "name": {"type": "string", "required": True,
                                 "description": "Name for the snapshot template"},
                        "description": {"type": "string", "required": False,
                                        "description": "Description"},
                    },
                ),
                SkillAction(
                    name="apply",
                    description="Apply a template to configure the agent (dry_run supported)",
                    parameters={
                        "template_id": {"type": "string", "required": True,
                                        "description": "Template ID or built-in name"},
                        "dry_run": {"type": "boolean", "required": False,
                                    "description": "Preview changes without applying"},
                    },
                ),
                SkillAction(
                    name="diff",
                    description="Compare two templates side by side",
                    parameters={
                        "template_a": {"type": "string", "required": True,
                                       "description": "First template ID"},
                        "template_b": {"type": "string", "required": True,
                                       "description": "Second template ID"},
                    },
                ),
                SkillAction(
                    name="export",
                    description="Export a template as a portable JSON bundle",
                    parameters={
                        "template_id": {"type": "string", "required": True,
                                        "description": "Template to export"},
                    },
                ),
                SkillAction(
                    name="import_template",
                    description="Import a template from a JSON bundle",
                    parameters={
                        "bundle": {"type": "object", "required": True,
                                   "description": "Template bundle from export"},
                    },
                ),
                SkillAction(
                    name="delete",
                    description="Delete a custom template",
                    parameters={
                        "template_id": {"type": "string", "required": True,
                                        "description": "Template to delete"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Get current configuration status and active template",
                    parameters={},
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any] = None) -> SkillResult:
        params = params or {}
        handlers = {
            "list": self._handle_list,
            "get": self._handle_get,
            "create": self._handle_create,
            "snapshot": self._handle_snapshot,
            "apply": self._handle_apply,
            "diff": self._handle_diff,
            "export": self._handle_export,
            "import_template": self._handle_import,
            "delete": self._handle_delete,
            "status": self._handle_status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action: {action}. Available: {list(handlers.keys())}",
            )
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    def _get_template(self, template_id: str) -> Optional[Dict]:
        """Get a template by ID, checking built-ins first then custom."""
        if template_id in BUILTIN_TEMPLATES:
            t = BUILTIN_TEMPLATES[template_id].copy()
            t["id"] = template_id
            t["builtin"] = True
            return t
        data = self._load()
        return data["templates"].get(template_id)

    async def _handle_list(self, params: Dict) -> SkillResult:
        """List all available templates."""
        category_filter = params.get("category")
        tag_filter = params.get("tag")

        templates = []

        # Add built-in templates
        for tid, t in BUILTIN_TEMPLATES.items():
            entry = {
                "id": tid,
                "name": t["name"],
                "description": t["description"],
                "category": t.get("category", "general"),
                "tags": t.get("tags", []),
                "builtin": True,
                "skills_enabled_count": len(t.get("skills_enabled", [])),
                "skills_disabled_count": len(t.get("skills_disabled", [])),
            }
            templates.append(entry)

        # Add custom templates
        data = self._load()
        for tid, t in data["templates"].items():
            entry = {
                "id": tid,
                "name": t.get("name", tid),
                "description": t.get("description", ""),
                "category": t.get("category", "custom"),
                "tags": t.get("tags", []),
                "builtin": False,
                "skills_enabled_count": len(t.get("skills_enabled", [])),
                "skills_disabled_count": len(t.get("skills_disabled", [])),
                "created_at": t.get("created_at", ""),
            }
            templates.append(entry)

        # Apply filters
        if category_filter:
            templates = [t for t in templates if t.get("category") == category_filter]
        if tag_filter:
            templates = [t for t in templates if tag_filter in t.get("tags", [])]

        return SkillResult(
            success=True,
            message=f"Found {len(templates)} templates",
            data={"templates": templates, "total": len(templates)},
        )

    async def _handle_get(self, params: Dict) -> SkillResult:
        """Get full details of a specific template."""
        template_id = params.get("template_id", "")
        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        template = self._get_template(template_id)
        if not template:
            return SkillResult(
                success=False,
                message=f"Template '{template_id}' not found",
                data={"available": list(BUILTIN_TEMPLATES.keys())},
            )

        return SkillResult(
            success=True,
            message=f"Template: {template.get('name', template_id)}",
            data={"template": template},
        )

    async def _handle_create(self, params: Dict) -> SkillResult:
        """Create a new custom configuration template."""
        name = params.get("name", "")
        if not name:
            return SkillResult(success=False, message="name is required")

        data = self._load()
        if len(data["templates"]) >= MAX_TEMPLATES:
            return SkillResult(
                success=False,
                message=f"Maximum {MAX_TEMPLATES} custom templates reached",
            )

        template_id = f"custom_{uuid.uuid4().hex[:8]}"
        template = {
            "id": template_id,
            "name": name,
            "description": params.get("description", ""),
            "skills_enabled": params.get("skills_enabled", []),
            "skills_disabled": params.get("skills_disabled", []),
            "parameters": params.get("parameters", {}),
            "category": params.get("category", "custom"),
            "tags": params.get("tags", []),
            "builtin": False,
            "created_at": datetime.now().isoformat(),
            "created_by": "agent",
        }

        data["templates"][template_id] = template
        data["stats"]["total_created"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Created template '{name}' (ID: {template_id})",
            data={"template_id": template_id, "template": template},
        )

    async def _handle_snapshot(self, params: Dict) -> SkillResult:
        """Create a template from the current agent configuration."""
        name = params.get("name", "")
        if not name:
            return SkillResult(success=False, message="name is required")

        data = self._load()
        if len(data["templates"]) >= MAX_TEMPLATES:
            return SkillResult(
                success=False,
                message=f"Maximum {MAX_TEMPLATES} custom templates reached",
            )

        # Gather current agent state
        current_skills = []
        agent_params = {}

        if self._context:
            # Get installed skills from registry
            try:
                registry = self._context._registry
                for sid, skill in registry._skills.items():
                    current_skills.append(sid)
            except (AttributeError, Exception):
                pass
            # Get agent info
            agent_params["agent_name"] = getattr(self._context, "agent_name", "unknown")
            agent_params["agent_ticker"] = getattr(self._context, "agent_ticker", "AGENT")

        template_id = f"snap_{uuid.uuid4().hex[:8]}"
        template = {
            "id": template_id,
            "name": name,
            "description": params.get("description", f"Snapshot taken at {datetime.now().isoformat()}"),
            "skills_enabled": current_skills,
            "skills_disabled": [],
            "parameters": agent_params,
            "category": "snapshot",
            "tags": ["snapshot", "auto-generated"],
            "builtin": False,
            "created_at": datetime.now().isoformat(),
            "created_by": "snapshot",
            "snapshot_metadata": {
                "skill_count": len(current_skills),
                "timestamp": time.time(),
            },
        }

        data["templates"][template_id] = template
        data["stats"]["total_created"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Snapshot template '{name}' created with {len(current_skills)} skills",
            data={"template_id": template_id, "template": template},
        )

    async def _handle_apply(self, params: Dict) -> SkillResult:
        """Apply a template to configure the agent."""
        template_id = params.get("template_id", "")
        dry_run = params.get("dry_run", False)

        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        template = self._get_template(template_id)
        if not template:
            return SkillResult(
                success=False,
                message=f"Template '{template_id}' not found",
            )

        # Calculate changes that would be made
        changes = {
            "skills_to_enable": template.get("skills_enabled", []),
            "skills_to_disable": template.get("skills_disabled", []),
            "parameter_overrides": template.get("parameters", {}),
        }

        # Check which skills are currently available
        currently_installed = []
        if self._context:
            try:
                registry = self._context._registry
                for sid in registry._skills:
                    currently_installed.append(sid)
            except (AttributeError, Exception):
                pass

        skills_to_enable = template.get("skills_enabled", [])
        skills_to_disable = template.get("skills_disabled", [])

        already_enabled = [s for s in skills_to_enable if s in currently_installed]
        not_installed = [s for s in skills_to_enable if s not in currently_installed]
        will_disable = [s for s in skills_to_disable if s in currently_installed]

        changes["already_enabled"] = already_enabled
        changes["not_installed"] = not_installed
        changes["will_disable"] = will_disable

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run for template '{template.get('name', template_id)}'",
                data={
                    "dry_run": True,
                    "template_id": template_id,
                    "changes": changes,
                },
            )

        # Apply: record the application (actual skill enable/disable
        # would need agent-level support, so we record the intent)
        data = self._load()
        data["active_template"] = template_id
        data["apply_history"].append({
            "template_id": template_id,
            "template_name": template.get("name", template_id),
            "applied_at": datetime.now().isoformat(),
            "changes": changes,
        })
        data["stats"]["total_applied"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Applied template '{template.get('name', template_id)}'",
            data={
                "template_id": template_id,
                "changes": changes,
                "active_template": template_id,
            },
        )

    async def _handle_diff(self, params: Dict) -> SkillResult:
        """Compare two templates side by side."""
        id_a = params.get("template_a", "")
        id_b = params.get("template_b", "")

        if not id_a or not id_b:
            return SkillResult(
                success=False, message="Both template_a and template_b are required"
            )

        tmpl_a = self._get_template(id_a)
        tmpl_b = self._get_template(id_b)

        if not tmpl_a:
            return SkillResult(success=False, message=f"Template '{id_a}' not found")
        if not tmpl_b:
            return SkillResult(success=False, message=f"Template '{id_b}' not found")

        enabled_a = set(tmpl_a.get("skills_enabled", []))
        enabled_b = set(tmpl_b.get("skills_enabled", []))
        disabled_a = set(tmpl_a.get("skills_disabled", []))
        disabled_b = set(tmpl_b.get("skills_disabled", []))
        params_a = tmpl_a.get("parameters", {})
        params_b = tmpl_b.get("parameters", {})

        # Compute differences
        skills_only_a = sorted(enabled_a - enabled_b)
        skills_only_b = sorted(enabled_b - enabled_a)
        skills_both = sorted(enabled_a & enabled_b)

        disabled_only_a = sorted(disabled_a - disabled_b)
        disabled_only_b = sorted(disabled_b - disabled_a)

        # Parameter differences
        all_param_keys = set(list(params_a.keys()) + list(params_b.keys()))
        param_diffs = {}
        for key in sorted(all_param_keys):
            val_a = params_a.get(key, "<not set>")
            val_b = params_b.get(key, "<not set>")
            if val_a != val_b:
                param_diffs[key] = {"a": val_a, "b": val_b}

        diff = {
            "template_a": {"id": id_a, "name": tmpl_a.get("name", id_a)},
            "template_b": {"id": id_b, "name": tmpl_b.get("name", id_b)},
            "skills_enabled": {
                "only_in_a": skills_only_a,
                "only_in_b": skills_only_b,
                "in_both": skills_both,
            },
            "skills_disabled": {
                "only_in_a": disabled_only_a,
                "only_in_b": disabled_only_b,
            },
            "parameter_differences": param_diffs,
            "summary": {
                "total_skill_differences": len(skills_only_a) + len(skills_only_b),
                "total_param_differences": len(param_diffs),
            },
        }

        return SkillResult(
            success=True,
            message=f"Compared '{tmpl_a.get('name', id_a)}' vs '{tmpl_b.get('name', id_b)}'",
            data={"diff": diff},
        )

    async def _handle_export(self, params: Dict) -> SkillResult:
        """Export a template as a portable JSON bundle."""
        template_id = params.get("template_id", "")
        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        template = self._get_template(template_id)
        if not template:
            return SkillResult(
                success=False, message=f"Template '{template_id}' not found"
            )

        bundle = {
            "format": "singularity_config_template_v1",
            "exported_at": datetime.now().isoformat(),
            "template": template,
        }

        data = self._load()
        data["stats"]["total_exported"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Exported template '{template.get('name', template_id)}'",
            data={"bundle": bundle},
        )

    async def _handle_import(self, params: Dict) -> SkillResult:
        """Import a template from a JSON bundle."""
        bundle = params.get("bundle", {})
        if not bundle:
            return SkillResult(success=False, message="bundle is required")

        fmt = bundle.get("format", "")
        if fmt != "singularity_config_template_v1":
            return SkillResult(
                success=False,
                message=f"Unknown bundle format: '{fmt}'. Expected 'singularity_config_template_v1'",
            )

        template = bundle.get("template", {})
        if not template or not template.get("name"):
            return SkillResult(
                success=False, message="Bundle missing valid template data"
            )

        data = self._load()
        if len(data["templates"]) >= MAX_TEMPLATES:
            return SkillResult(
                success=False,
                message=f"Maximum {MAX_TEMPLATES} custom templates reached",
            )

        # Create new ID to avoid collisions
        template_id = f"imported_{uuid.uuid4().hex[:8]}"
        template["id"] = template_id
        template["builtin"] = False
        template["imported_at"] = datetime.now().isoformat()
        template["imported_from"] = bundle.get("exported_at", "unknown")

        data["templates"][template_id] = template
        data["stats"]["total_imported"] += 1
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Imported template '{template.get('name')}' (ID: {template_id})",
            data={"template_id": template_id, "template": template},
        )

    async def _handle_delete(self, params: Dict) -> SkillResult:
        """Delete a custom template."""
        template_id = params.get("template_id", "")
        if not template_id:
            return SkillResult(success=False, message="template_id is required")

        if template_id in BUILTIN_TEMPLATES:
            return SkillResult(
                success=False, message="Cannot delete built-in templates"
            )

        data = self._load()
        if template_id not in data["templates"]:
            return SkillResult(
                success=False, message=f"Template '{template_id}' not found"
            )

        name = data["templates"][template_id].get("name", template_id)
        del data["templates"][template_id]

        # Clear active template if it was the deleted one
        if data.get("active_template") == template_id:
            data["active_template"] = None

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Deleted template '{name}'",
            data={"deleted_id": template_id},
        )

    async def _handle_status(self, params: Dict) -> SkillResult:
        """Get current configuration status and active template."""
        data = self._load()

        active = data.get("active_template")
        active_template = None
        if active:
            active_template = self._get_template(active)

        # Current skill inventory
        current_skills = []
        if self._context:
            try:
                registry = self._context._registry
                for sid in registry._skills:
                    current_skills.append(sid)
            except (AttributeError, Exception):
                pass

        recent_applies = data.get("apply_history", [])[-5:]

        return SkillResult(
            success=True,
            message=f"Active template: {active or 'none'}",
            data={
                "active_template": active,
                "active_template_details": active_template,
                "current_skills_count": len(current_skills),
                "current_skills": current_skills[:20],  # Limit output
                "custom_template_count": len(data["templates"]),
                "builtin_template_count": len(BUILTIN_TEMPLATES),
                "stats": data["stats"],
                "recent_applies": recent_applies,
            },
        )
