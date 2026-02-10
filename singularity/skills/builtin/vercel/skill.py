#!/usr/bin/env python3
"""
Vercel Skill - Deploy and manage projects on Vercel

Real deployments. No mocks.
"""

import httpx
from typing import Dict, List
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


class VercelSkill(Skill):
    """
    Skill for Vercel deployments and project management.

    Required credentials:
    - VERCEL_TOKEN: Vercel API token
    """

    API_BASE = "https://api.vercel.com"

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="vercel",
            name="Vercel Deployments",
            version="1.0.0",
            category="dev",
            description="Deploy projects, manage domains, and control Vercel infrastructure",
            required_credentials=["VERCEL_TOKEN"],
            install_cost=0,
            actions=[
                SkillAction(name="list_projects", description="List all Vercel projects",
                    parameters={}, estimated_cost=0, success_probability=0.95),
                SkillAction(name="get_project", description="Get details of a specific project",
                    parameters={"project_id": "Project ID or name"}, estimated_cost=0, success_probability=0.95),
                SkillAction(name="create_project", description="Create a new Vercel project",
                    parameters={"name": "Project name", "framework": "Framework (nextjs, react, vue, etc.)",
                                "git_repo": "GitHub repo (owner/repo)"},
                    estimated_cost=0, success_probability=0.9),
                SkillAction(name="deploy", description="Trigger a new deployment",
                    parameters={"project_id": "Project ID or name",
                                "target": "Deployment target (production or preview)"},
                    estimated_cost=0, success_probability=0.85),
                SkillAction(name="list_deployments", description="List deployments for a project",
                    parameters={"project_id": "Project ID or name", "limit": "Max results (default 10)"},
                    estimated_cost=0, success_probability=0.95),
                SkillAction(name="get_deployment", description="Get deployment details and status",
                    parameters={"deployment_id": "Deployment ID or URL"}, estimated_cost=0, success_probability=0.95),
                SkillAction(name="list_domains", description="List all domains",
                    parameters={}, estimated_cost=0, success_probability=0.95),
                SkillAction(name="add_domain", description="Add a domain to a project",
                    parameters={"project_id": "Project ID or name", "domain": "Domain name to add"},
                    estimated_cost=0, success_probability=0.85),
                SkillAction(name="remove_domain", description="Remove a domain from a project",
                    parameters={"project_id": "Project ID or name", "domain": "Domain name to remove"},
                    estimated_cost=0, success_probability=0.9),
                SkillAction(name="set_env", description="Set an environment variable",
                    parameters={"project_id": "Project ID or name", "key": "Variable name",
                                "value": "Variable value",
                                "target": "Environment (production, preview, development)"},
                    estimated_cost=0, success_probability=0.9),
                SkillAction(name="list_env", description="List environment variables for a project",
                    parameters={"project_id": "Project ID or name"}, estimated_cost=0, success_probability=0.95),
                SkillAction(name="delete_project", description="Delete a Vercel project",
                    parameters={"project_id": "Project ID or name"}, estimated_cost=0, success_probability=0.9),
                SkillAction(name="get_user", description="Get current user/team info",
                    parameters={}, estimated_cost=0, success_probability=0.95),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient()

    def _get_headers(self) -> Dict:
        """Get authorization headers"""
        return {
            "Authorization": f"Bearer {self.credentials.get('VERCEL_TOKEN')}",
            "Content-Type": "application/json"
        }

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a Vercel action"""
        from . import projects, infra

        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(success=False, message=f"Missing credentials: {missing}")

        try:
            if action == "list_projects":
                return await projects.list_projects(self)
            elif action == "get_project":
                return await projects.get_project(self, params.get("project_id"))
            elif action == "create_project":
                return await projects.create_project(self, params.get("name"), params.get("framework"), params.get("git_repo"))
            elif action == "deploy":
                return await projects.deploy(self, params.get("project_id"), params.get("target", "production"))
            elif action == "list_deployments":
                return await projects.list_deployments(self, params.get("project_id"), params.get("limit", 10))
            elif action == "get_deployment":
                return await projects.get_deployment(self, params.get("deployment_id"))
            elif action == "delete_project":
                return await projects.delete_project(self, params.get("project_id"))
            elif action == "list_domains":
                return await infra.list_domains(self)
            elif action == "add_domain":
                return await infra.add_domain(self, params.get("project_id"), params.get("domain"))
            elif action == "remove_domain":
                return await infra.remove_domain(self, params.get("project_id"), params.get("domain"))
            elif action == "set_env":
                return await infra.set_env(self, params.get("project_id"), params.get("key"), params.get("value"),
                                           params.get("target", ["production", "preview", "development"]))
            elif action == "list_env":
                return await infra.list_env(self, params.get("project_id"))
            elif action == "get_user":
                return await infra.get_user(self)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Vercel error: {str(e)}")

    async def close(self):
        """Clean up"""
        await self.http.aclose()
