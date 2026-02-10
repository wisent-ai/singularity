"""
Vercel Skill - Project & Deployment Actions

Contains project management and deployment action implementations.
"""

from typing import Dict
from singularity.skills.base import SkillResult


async def list_projects(skill) -> SkillResult:
    """List all projects"""
    response = await skill.http.get(
        f"{skill.API_BASE}/v9/projects",
        headers=skill._get_headers()
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


async def get_project(skill, project_id: str) -> SkillResult:
    """Get project details"""
    if not project_id:
        return SkillResult(success=False, message="Project ID required")

    response = await skill.http.get(
        f"{skill.API_BASE}/v9/projects/{project_id}",
        headers=skill._get_headers()
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


async def create_project(skill, name: str, framework: str = None, git_repo: str = None) -> SkillResult:
    """Create a new project"""
    if not name:
        return SkillResult(success=False, message="Project name required")

    data = {"name": name}
    if framework:
        data["framework"] = framework
    if git_repo:
        data["gitRepository"] = {"type": "github", "repo": git_repo}

    response = await skill.http.post(
        f"{skill.API_BASE}/v10/projects",
        headers=skill._get_headers(),
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


async def delete_project(skill, project_id: str) -> SkillResult:
    """Delete a project"""
    if not project_id:
        return SkillResult(success=False, message="Project ID required")

    response = await skill.http.delete(
        f"{skill.API_BASE}/v9/projects/{project_id}",
        headers=skill._get_headers()
    )

    if response.status_code in [200, 204]:
        return SkillResult(success=True, message=f"Deleted project: {project_id}")
    return SkillResult(success=False, message=f"Failed: {response.text}")


async def deploy(skill, project_id: str, target: str = "production") -> SkillResult:
    """Trigger a deployment"""
    if not project_id:
        return SkillResult(success=False, message="Project ID required")

    project_resp = await skill.http.get(
        f"{skill.API_BASE}/v9/projects/{project_id}",
        headers=skill._get_headers()
    )

    if project_resp.status_code != 200:
        return SkillResult(success=False, message="Project not found")

    project = project_resp.json()
    data = {"name": project.get("name"), "target": target}

    if project.get("link"):
        data["gitSource"] = {
            "type": project["link"].get("type", "github"),
            "ref": "main",
            "repoId": project["link"].get("repoId")
        }

    response = await skill.http.post(
        f"{skill.API_BASE}/v13/deployments",
        headers=skill._get_headers(),
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


async def list_deployments(skill, project_id: str, limit: int = 10) -> SkillResult:
    """List deployments"""
    params = {"limit": limit}
    if project_id:
        params["projectId"] = project_id

    response = await skill.http.get(
        f"{skill.API_BASE}/v6/deployments",
        headers=skill._get_headers(),
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


async def get_deployment(skill, deployment_id: str) -> SkillResult:
    """Get deployment details"""
    if not deployment_id:
        return SkillResult(success=False, message="Deployment ID required")

    response = await skill.http.get(
        f"{skill.API_BASE}/v13/deployments/{deployment_id}",
        headers=skill._get_headers()
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
