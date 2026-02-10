#!/usr/bin/env python3
"""
Supabase Skill - Core Class

SupabaseSkill with manifest, execute routing, and helper function.
"""

import asyncio
import httpx
from typing import Dict, List, Optional, Any
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


class SupabaseSkill(Skill):
    """
    Supabase Management Skill.

    Create and configure Supabase projects programmatically.
    Provides auth, database, storage, and edge function management.
    """

    API_BASE = "https://api.supabase.com/v1"

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()
        self._projects_cache: Dict[str, dict] = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="supabase",
            name="Supabase",
            version="1.0.0",
            category="infrastructure",
            description="Create and manage Supabase projects with auth, database, and storage",
            required_credentials=["SUPABASE_ACCESS_TOKEN"],
            install_cost=0,
            actions=[
                SkillAction(name="list_projects", description="List all Supabase projects",
                    parameters={}, estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="create_project", description="Create a new Supabase project",
                    parameters={
                        "name": {"type": "string", "required": True, "description": "Project name"},
                        "organization_id": {"type": "string", "required": True, "description": "Organization ID"},
                        "region": {"type": "string", "required": False, "description": "Region (default: us-east-1)"},
                        "db_password": {"type": "string", "required": True, "description": "Database password"},
                        "plan": {"type": "string", "required": False, "description": "Plan: free, pro (default: free)"}
                    },
                    estimated_cost=0, estimated_duration_seconds=60, success_probability=0.9),
                SkillAction(name="get_project", description="Get project details including API keys",
                    parameters={"project_id": {"type": "string", "required": True, "description": "Project ID (ref)"}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="delete_project", description="Delete a Supabase project",
                    parameters={"project_id": {"type": "string", "required": True, "description": "Project ID to delete"}},
                    estimated_cost=0, estimated_duration_seconds=5, success_probability=0.9),
                SkillAction(name="get_api_keys", description="Get project API keys (anon, service_role)",
                    parameters={"project_id": {"type": "string", "required": True, "description": "Project ID"}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="run_sql", description="Run SQL query on project database",
                    parameters={
                        "project_id": {"type": "string", "required": True, "description": "Project ID"},
                        "query": {"type": "string", "required": True, "description": "SQL query to execute"}
                    },
                    estimated_cost=0, estimated_duration_seconds=5, success_probability=0.85),
                SkillAction(name="list_tables", description="List all tables in the database",
                    parameters={
                        "project_id": {"type": "string", "required": True, "description": "Project ID"},
                        "schema": {"type": "string", "required": False, "description": "Schema name (default: public)"}
                    },
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.9),
                SkillAction(name="create_table", description="Create a new table with columns",
                    parameters={
                        "project_id": {"type": "string", "required": True, "description": "Project ID"},
                        "table_name": {"type": "string", "required": True, "description": "Table name"},
                        "columns": {"type": "array", "required": True, "description": "Column definitions"},
                        "enable_rls": {"type": "boolean", "required": False, "description": "Enable RLS (default: true)"}
                    },
                    estimated_cost=0, estimated_duration_seconds=5, success_probability=0.85),
                SkillAction(name="get_auth_config", description="Get current auth configuration",
                    parameters={"project_id": {"type": "string", "required": True, "description": "Project ID"}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="update_auth_config", description="Update auth settings",
                    parameters={
                        "project_id": {"type": "string", "required": True, "description": "Project ID"},
                        "site_url": {"type": "string", "required": False},
                        "jwt_expiry": {"type": "integer", "required": False},
                        "disable_signup": {"type": "boolean", "required": False},
                        "external_email_enabled": {"type": "boolean", "required": False},
                        "external_phone_enabled": {"type": "boolean", "required": False}
                    },
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.9),
                SkillAction(name="configure_oauth_provider", description="Configure OAuth provider",
                    parameters={
                        "project_id": {"type": "string", "required": True, "description": "Project ID"},
                        "provider": {"type": "string", "required": True, "description": "Provider name"},
                        "enabled": {"type": "boolean", "required": True},
                        "client_id": {"type": "string", "required": False},
                        "client_secret": {"type": "string", "required": False}
                    },
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.9),
                SkillAction(name="list_buckets", description="List storage buckets",
                    parameters={"project_id": {"type": "string", "required": True, "description": "Project ID"}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="create_bucket", description="Create a storage bucket",
                    parameters={
                        "project_id": {"type": "string", "required": True, "description": "Project ID"},
                        "name": {"type": "string", "required": True, "description": "Bucket name"},
                        "public": {"type": "boolean", "required": False},
                        "file_size_limit": {"type": "integer", "required": False},
                        "allowed_mime_types": {"type": "array", "required": False}
                    },
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.9),
                SkillAction(name="list_organizations", description="List all organizations",
                    parameters={}, estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="list_secrets", description="List project secrets (names only)",
                    parameters={"project_id": {"type": "string", "required": True, "description": "Project ID"}},
                    estimated_cost=0, estimated_duration_seconds=2, success_probability=0.95),
                SkillAction(name="create_secret", description="Create or update a project secret",
                    parameters={
                        "project_id": {"type": "string", "required": True, "description": "Project ID"},
                        "name": {"type": "string", "required": True, "description": "Secret name"},
                        "value": {"type": "string", "required": True, "description": "Secret value"}
                    },
                    estimated_cost=0, estimated_duration_seconds=3, success_probability=0.9),
            ]
        )

    def _get_headers(self) -> Dict[str, str]:
        token = self.credentials.get("SUPABASE_ACCESS_TOKEN", "")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not self.check_credentials():
            return SkillResult(success=False, message="Missing SUPABASE_ACCESS_TOKEN credential")

        try:
            from . import projects, database, config

            actions_map = {
                "list_projects": lambda: projects.list_projects(self),
                "create_project": lambda: projects.create_project(
                    self, params.get("name"), params.get("organization_id"),
                    params.get("region", "us-east-1"), params.get("db_password"),
                    params.get("plan", "free")),
                "get_project": lambda: projects.get_project(self, params.get("project_id")),
                "delete_project": lambda: projects.delete_project(self, params.get("project_id")),
                "get_api_keys": lambda: projects.get_api_keys(self, params.get("project_id")),
                "run_sql": lambda: database.run_sql(self, params.get("project_id"), params.get("query")),
                "list_tables": lambda: database.list_tables(self, params.get("project_id"), params.get("schema", "public")),
                "create_table": lambda: database.create_table(
                    self, params.get("project_id"), params.get("table_name"),
                    params.get("columns", []), params.get("enable_rls", True)),
                "get_auth_config": lambda: config.get_auth_config(self, params.get("project_id")),
                "update_auth_config": lambda: config.update_auth_config(self, params),
                "configure_oauth_provider": lambda: config.configure_oauth_provider(self, params),
                "list_buckets": lambda: config.list_buckets(self, params.get("project_id")),
                "create_bucket": lambda: config.create_bucket(self, params),
                "list_organizations": lambda: config.list_organizations(self),
                "list_secrets": lambda: config.list_secrets(self, params.get("project_id")),
                "create_secret": lambda: config.create_secret(
                    self, params.get("project_id"), params.get("name"), params.get("value")),
            }

            handler = actions_map.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()

        except Exception as e:
            return SkillResult(success=False, message=f"Supabase error: {str(e)}")

    async def close(self):
        """Cleanup"""
        await self.http.aclose()


async def setup_supabase_auth_project(
    skill: SupabaseSkill,
    name: str,
    organization_id: str,
    db_password: str,
    enable_google: bool = False,
    enable_github: bool = False,
    site_url: str = None
) -> Dict:
    """
    Helper to create a new Supabase project with auth configured.

    Returns dict with project_id, api_url, anon_key, service_role_key.
    """
    result = await skill.execute("create_project", {
        "name": name,
        "organization_id": organization_id,
        "db_password": db_password,
        "plan": "free"
    })

    if not result.success:
        return {"error": result.message}

    project_id = result.data["project_id"]

    for _ in range(30):
        await asyncio.sleep(10)
        status_result = await skill.execute("get_project", {"project_id": project_id})
        if status_result.success:
            status = status_result.data.get("project", {}).get("status")
            if status == "ACTIVE_HEALTHY":
                break

    keys_result = await skill.execute("get_api_keys", {"project_id": project_id})
    if not keys_result.success:
        return {"error": f"Project created but failed to get keys: {keys_result.message}"}

    if site_url:
        await skill.execute("update_auth_config", {
            "project_id": project_id,
            "site_url": site_url
        })

    return {
        "project_id": project_id,
        "api_url": keys_result.data["api_url"],
        "anon_key": keys_result.data["anon_key"],
        "service_role_key": keys_result.data["service_role_key"]
    }
