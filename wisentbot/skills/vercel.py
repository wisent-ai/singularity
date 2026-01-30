#!/usr/bin/env python3
"""
Vercel Skill - Deploy and manage projects on Vercel

Real deployments. No mocks.
"""

import httpx
from typing import Dict, List
from .base import Skill, SkillResult, SkillManifest, SkillAction


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
                SkillAction(
                    name="list_projects",
                    description="List all Vercel projects",
                    parameters={},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_project",
                    description="Get details of a specific project",
                    parameters={"project_id": "Project ID or name"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="create_project",
                    description="Create a new Vercel project",
                    parameters={
                        "name": "Project name",
                        "framework": "Framework (nextjs, react, vue, etc.)",
                        "git_repo": "GitHub repo (owner/repo)"
                    },
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="deploy",
                    description="Trigger a new deployment",
                    parameters={
                        "project_id": "Project ID or name",
                        "target": "Deployment target (production or preview)"
                    },
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="list_deployments",
                    description="List deployments for a project",
                    parameters={
                        "project_id": "Project ID or name",
                        "limit": "Max results (default 10)"
                    },
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_deployment",
                    description="Get deployment details and status",
                    parameters={"deployment_id": "Deployment ID or URL"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="list_domains",
                    description="List all domains",
                    parameters={},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="add_domain",
                    description="Add a domain to a project",
                    parameters={
                        "project_id": "Project ID or name",
                        "domain": "Domain name to add"
                    },
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="remove_domain",
                    description="Remove a domain from a project",
                    parameters={
                        "project_id": "Project ID or name",
                        "domain": "Domain name to remove"
                    },
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="set_env",
                    description="Set an environment variable",
                    parameters={
                        "project_id": "Project ID or name",
                        "key": "Variable name",
                        "value": "Variable value",
                        "target": "Environment (production, preview, development)"
                    },
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="list_env",
                    description="List environment variables for a project",
                    parameters={"project_id": "Project ID or name"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="delete_project",
                    description="Delete a Vercel project",
                    parameters={"project_id": "Project ID or name"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="get_user",
                    description="Get current user/team info",
                    parameters={},
                    estimated_cost=0,
                    success_probability=0.95
                ),
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient(timeout=60)

    def _get_headers(self) -> Dict:
        """Get authorization headers"""
        return {
            "Authorization": f"Bearer {self.credentials.get('VERCEL_TOKEN')}",
            "Content-Type": "application/json"
        }

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute a Vercel action"""
        if not self.check_credentials():
            missing = self.get_missing_credentials()
            return SkillResult(
                success=False,
                message=f"Missing credentials: {missing}"
            )

        try:
            if action == "list_projects":
                return await self._list_projects()
            elif action == "get_project":
                return await self._get_project(params.get("project_id"))
            elif action == "create_project":
                return await self._create_project(
                    params.get("name"),
                    params.get("framework"),
                    params.get("git_repo")
                )
            elif action == "deploy":
                return await self._deploy(
                    params.get("project_id"),
                    params.get("target", "production")
                )
            elif action == "list_deployments":
                return await self._list_deployments(
                    params.get("project_id"),
                    params.get("limit", 10)
                )
            elif action == "get_deployment":
                return await self._get_deployment(params.get("deployment_id"))
            elif action == "list_domains":
                return await self._list_domains()
            elif action == "add_domain":
                return await self._add_domain(
                    params.get("project_id"),
                    params.get("domain")
                )
            elif action == "remove_domain":
                return await self._remove_domain(
                    params.get("project_id"),
                    params.get("domain")
                )
            elif action == "set_env":
                return await self._set_env(
                    params.get("project_id"),
                    params.get("key"),
                    params.get("value"),
                    params.get("target", ["production", "preview", "development"])
                )
            elif action == "list_env":
                return await self._list_env(params.get("project_id"))
            elif action == "delete_project":
                return await self._delete_project(params.get("project_id"))
            elif action == "get_user":
                return await self._get_user()
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Vercel error: {str(e)}")

    async def _list_projects(self) -> SkillResult:
        """List all projects"""
        response = await self.http.get(
            f"{self.API_BASE}/v9/projects",
            headers=self._get_headers()
        )

        if response.status_code == 200:
            data = response.json()
            projects = [
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "framework": p.get("framework"),
                    "url": f"https://{p.get('name')}.vercel.app"
                }
                for p in data.get("projects", [])
            ]
            return SkillResult(
                success=True,
                message=f"Found {len(projects)} projects",
                data={"projects": projects}
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _get_project(self, project_id: str) -> SkillResult:
        """Get project details"""
        if not project_id:
            return SkillResult(success=False, message="Project ID required")

        response = await self.http.get(
            f"{self.API_BASE}/v9/projects/{project_id}",
            headers=self._get_headers()
        )

        if response.status_code == 200:
            p = response.json()
            return SkillResult(
                success=True,
                message=f"Got project: {p.get('name')}",
                data={
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "framework": p.get("framework"),
                    "node_version": p.get("nodeVersion"),
                    "domains": [d.get("name") for d in p.get("alias", [])],
                    "created": p.get("createdAt"),
                    "updated": p.get("updatedAt")
                }
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _create_project(
        self,
        name: str,
        framework: str = None,
        git_repo: str = None
    ) -> SkillResult:
        """Create a new project"""
        if not name:
            return SkillResult(success=False, message="Project name required")

        data = {"name": name}
        if framework:
            data["framework"] = framework
        if git_repo:
            data["gitRepository"] = {
                "type": "github",
                "repo": git_repo
            }

        response = await self.http.post(
            f"{self.API_BASE}/v10/projects",
            headers=self._get_headers(),
            json=data
        )

        if response.status_code in [200, 201]:
            p = response.json()
            return SkillResult(
                success=True,
                message=f"Created project: {p.get('name')}",
                data={
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "url": f"https://{p.get('name')}.vercel.app"
                },
                asset_created={
                    "type": "vercel_project",
                    "name": p.get("name"),
                    "id": p.get("id")
                }
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _deploy(self, project_id: str, target: str = "production") -> SkillResult:
        """Trigger a deployment"""
        if not project_id:
            return SkillResult(success=False, message="Project ID required")

        # Get project to find git repo
        project_resp = await self.http.get(
            f"{self.API_BASE}/v9/projects/{project_id}",
            headers=self._get_headers()
        )

        if project_resp.status_code != 200:
            return SkillResult(success=False, message="Project not found")

        project = project_resp.json()

        # Trigger deployment via deployment hook or direct API
        data = {
            "name": project.get("name"),
            "target": target
        }

        if project.get("link"):
            data["gitSource"] = {
                "type": project["link"].get("type", "github"),
                "ref": "main",
                "repoId": project["link"].get("repoId")
            }

        response = await self.http.post(
            f"{self.API_BASE}/v13/deployments",
            headers=self._get_headers(),
            json=data
        )

        if response.status_code in [200, 201]:
            d = response.json()
            return SkillResult(
                success=True,
                message=f"Deployment started: {d.get('id')}",
                data={
                    "id": d.get("id"),
                    "url": d.get("url"),
                    "state": d.get("readyState"),
                    "target": target
                }
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _list_deployments(self, project_id: str, limit: int = 10) -> SkillResult:
        """List deployments"""
        params = {"limit": limit}
        if project_id:
            params["projectId"] = project_id

        response = await self.http.get(
            f"{self.API_BASE}/v6/deployments",
            headers=self._get_headers(),
            params=params
        )

        if response.status_code == 200:
            data = response.json()
            deployments = [
                {
                    "id": d.get("uid"),
                    "url": d.get("url"),
                    "state": d.get("state"),
                    "target": d.get("target"),
                    "created": d.get("created")
                }
                for d in data.get("deployments", [])
            ]
            return SkillResult(
                success=True,
                message=f"Found {len(deployments)} deployments",
                data={"deployments": deployments}
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _get_deployment(self, deployment_id: str) -> SkillResult:
        """Get deployment details"""
        if not deployment_id:
            return SkillResult(success=False, message="Deployment ID required")

        response = await self.http.get(
            f"{self.API_BASE}/v13/deployments/{deployment_id}",
            headers=self._get_headers()
        )

        if response.status_code == 200:
            d = response.json()
            return SkillResult(
                success=True,
                message=f"Deployment {d.get('readyState')}",
                data={
                    "id": d.get("id"),
                    "url": d.get("url"),
                    "state": d.get("readyState"),
                    "target": d.get("target"),
                    "created": d.get("createdAt"),
                    "ready": d.get("ready")
                }
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _list_domains(self) -> SkillResult:
        """List all domains"""
        response = await self.http.get(
            f"{self.API_BASE}/v5/domains",
            headers=self._get_headers()
        )

        if response.status_code == 200:
            data = response.json()
            domains = [
                {
                    "name": d.get("name"),
                    "verified": d.get("verified"),
                    "configured": d.get("configured")
                }
                for d in data.get("domains", [])
            ]
            return SkillResult(
                success=True,
                message=f"Found {len(domains)} domains",
                data={"domains": domains}
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _add_domain(self, project_id: str, domain: str) -> SkillResult:
        """Add domain to project"""
        if not project_id or not domain:
            return SkillResult(success=False, message="Project ID and domain required")

        response = await self.http.post(
            f"{self.API_BASE}/v10/projects/{project_id}/domains",
            headers=self._get_headers(),
            json={"name": domain}
        )

        if response.status_code in [200, 201]:
            d = response.json()
            return SkillResult(
                success=True,
                message=f"Added domain: {domain}",
                data={
                    "domain": d.get("name"),
                    "verified": d.get("verified"),
                    "configured": d.get("configured")
                }
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _remove_domain(self, project_id: str, domain: str) -> SkillResult:
        """Remove domain from project"""
        if not project_id or not domain:
            return SkillResult(success=False, message="Project ID and domain required")

        response = await self.http.delete(
            f"{self.API_BASE}/v9/projects/{project_id}/domains/{domain}",
            headers=self._get_headers()
        )

        if response.status_code in [200, 204]:
            return SkillResult(
                success=True,
                message=f"Removed domain: {domain}"
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _set_env(
        self,
        project_id: str,
        key: str,
        value: str,
        target: List[str] = None
    ) -> SkillResult:
        """Set environment variable"""
        if not project_id or not key or not value:
            return SkillResult(success=False, message="Project ID, key, and value required")

        if target is None:
            target = ["production", "preview", "development"]
        if isinstance(target, str):
            target = [target]

        response = await self.http.post(
            f"{self.API_BASE}/v10/projects/{project_id}/env",
            headers=self._get_headers(),
            json={
                "key": key,
                "value": value,
                "target": target,
                "type": "encrypted"
            }
        )

        if response.status_code in [200, 201]:
            return SkillResult(
                success=True,
                message=f"Set env var: {key}",
                data={"key": key, "target": target}
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _list_env(self, project_id: str) -> SkillResult:
        """List environment variables"""
        if not project_id:
            return SkillResult(success=False, message="Project ID required")

        response = await self.http.get(
            f"{self.API_BASE}/v9/projects/{project_id}/env",
            headers=self._get_headers()
        )

        if response.status_code == 200:
            data = response.json()
            envs = [
                {
                    "key": e.get("key"),
                    "target": e.get("target"),
                    "type": e.get("type")
                }
                for e in data.get("envs", [])
            ]
            return SkillResult(
                success=True,
                message=f"Found {len(envs)} env vars",
                data={"envs": envs}
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _delete_project(self, project_id: str) -> SkillResult:
        """Delete a project"""
        if not project_id:
            return SkillResult(success=False, message="Project ID required")

        response = await self.http.delete(
            f"{self.API_BASE}/v9/projects/{project_id}",
            headers=self._get_headers()
        )

        if response.status_code in [200, 204]:
            return SkillResult(
                success=True,
                message=f"Deleted project: {project_id}"
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def _get_user(self) -> SkillResult:
        """Get current user info"""
        response = await self.http.get(
            f"{self.API_BASE}/v2/user",
            headers=self._get_headers()
        )

        if response.status_code == 200:
            u = response.json().get("user", {})
            return SkillResult(
                success=True,
                message=f"User: {u.get('username')}",
                data={
                    "id": u.get("id"),
                    "username": u.get("username"),
                    "email": u.get("email"),
                    "name": u.get("name")
                }
            )
        return SkillResult(success=False, message=f"Failed: {response.text}")

    async def close(self):
        """Clean up"""
        await self.http.aclose()
