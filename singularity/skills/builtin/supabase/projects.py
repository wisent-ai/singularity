"""
Supabase Skill - Project Management

Contains list_projects, create_project, get_project, delete_project, get_api_keys.
"""

from typing import Dict
from singularity.skills.base import SkillResult


async def list_projects(skill) -> SkillResult:
    """List all projects"""
    response = await skill.http.get(
        f"{skill.API_BASE}/projects",
        headers=skill._get_headers()
    )

    if response.status_code == 200:
        projects = response.json()
        return SkillResult(
            success=True,
            message=f"Found {len(projects)} projects",
            data={"projects": projects}
        )
    else:
        return SkillResult(success=False, message=f"Failed to list projects: {response.text}")


async def create_project(skill, name: str, organization_id: str,
                         region: str, db_password: str, plan: str) -> SkillResult:
    """Create a new Supabase project"""
    if not all([name, organization_id, db_password]):
        return SkillResult(success=False, message="name, organization_id, and db_password are required")

    payload = {
        "name": name,
        "organization_id": organization_id,
        "region": region,
        "db_pass": db_password,
        "plan": plan
    }

    response = await skill.http.post(
        f"{skill.API_BASE}/projects",
        headers=skill._get_headers(),
        json=payload
    )

    if response.status_code in [200, 201]:
        project = response.json()
        project_id = project.get("id")
        skill._projects_cache[project_id] = project

        return SkillResult(
            success=True,
            message=f"Created project: {name}",
            data={
                "project_id": project_id,
                "name": name,
                "region": region,
                "status": project.get("status"),
                "api_url": f"https://{project_id}.supabase.co",
                "note": "Project is provisioning. Use get_api_keys once status is ACTIVE_HEALTHY."
            }
        )
    else:
        return SkillResult(success=False, message=f"Failed to create project: {response.text}")


async def get_project(skill, project_id: str) -> SkillResult:
    """Get project details"""
    if not project_id:
        return SkillResult(success=False, message="project_id required")

    response = await skill.http.get(
        f"{skill.API_BASE}/projects/{project_id}",
        headers=skill._get_headers()
    )

    if response.status_code == 200:
        project = response.json()
        return SkillResult(
            success=True,
            message=f"Project: {project.get('name')}",
            data={
                "project": project,
                "api_url": f"https://{project_id}.supabase.co",
                "db_host": f"db.{project_id}.supabase.co"
            }
        )
    else:
        return SkillResult(success=False, message=f"Failed to get project: {response.text}")


async def delete_project(skill, project_id: str) -> SkillResult:
    """Delete a project"""
    if not project_id:
        return SkillResult(success=False, message="project_id required")

    response = await skill.http.delete(
        f"{skill.API_BASE}/projects/{project_id}",
        headers=skill._get_headers()
    )

    if response.status_code in [200, 204]:
        if project_id in skill._projects_cache:
            del skill._projects_cache[project_id]
        return SkillResult(success=True, message=f"Deleted project: {project_id}")
    else:
        return SkillResult(success=False, message=f"Failed to delete project: {response.text}")


async def get_api_keys(skill, project_id: str) -> SkillResult:
    """Get project API keys"""
    if not project_id:
        return SkillResult(success=False, message="project_id required")

    response = await skill.http.get(
        f"{skill.API_BASE}/projects/{project_id}/api-keys",
        headers=skill._get_headers()
    )

    if response.status_code == 200:
        keys = response.json()
        key_dict = {}
        for key in keys:
            key_dict[key.get("name")] = key.get("api_key")

        return SkillResult(
            success=True,
            message="Retrieved API keys",
            data={
                "anon_key": key_dict.get("anon"),
                "service_role_key": key_dict.get("service_role"),
                "api_url": f"https://{project_id}.supabase.co"
            }
        )
    else:
        return SkillResult(success=False, message=f"Failed to get API keys: {response.text}")
