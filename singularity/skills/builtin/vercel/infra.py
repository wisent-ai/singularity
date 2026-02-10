"""
Vercel Skill - Infrastructure Actions

Contains domain, environment variable, and user action implementations.
"""

from typing import Dict, List
from singularity.skills.base import SkillResult


async def list_domains(skill) -> SkillResult:
    """List all domains"""
    response = await skill.http.get(
        f"{skill.API_BASE}/v5/domains",
        headers=skill._get_headers()
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


async def add_domain(skill, project_id: str, domain: str) -> SkillResult:
    """Add domain to project"""
    if not project_id or not domain:
        return SkillResult(success=False, message="Project ID and domain required")

    response = await skill.http.post(
        f"{skill.API_BASE}/v10/projects/{project_id}/domains",
        headers=skill._get_headers(),
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


async def remove_domain(skill, project_id: str, domain: str) -> SkillResult:
    """Remove domain from project"""
    if not project_id or not domain:
        return SkillResult(success=False, message="Project ID and domain required")

    response = await skill.http.delete(
        f"{skill.API_BASE}/v9/projects/{project_id}/domains/{domain}",
        headers=skill._get_headers()
    )

    if response.status_code in [200, 204]:
        return SkillResult(success=True, message=f"Removed domain: {domain}")
    return SkillResult(success=False, message=f"Failed: {response.text}")


async def set_env(skill, project_id: str, key: str, value: str, target: List[str] = None) -> SkillResult:
    """Set environment variable"""
    if not project_id or not key or not value:
        return SkillResult(success=False, message="Project ID, key, and value required")

    if target is None:
        target = ["production", "preview", "development"]
    if isinstance(target, str):
        target = [target]

    response = await skill.http.post(
        f"{skill.API_BASE}/v10/projects/{project_id}/env",
        headers=skill._get_headers(),
        json={"key": key, "value": value, "target": target, "type": "encrypted"}
    )

    if response.status_code in [200, 201]:
        return SkillResult(
            success=True,
            message=f"Set env var: {key}",
            data={"key": key, "target": target}
        )
    return SkillResult(success=False, message=f"Failed: {response.text}")


async def list_env(skill, project_id: str) -> SkillResult:
    """List environment variables"""
    if not project_id:
        return SkillResult(success=False, message="Project ID required")

    response = await skill.http.get(
        f"{skill.API_BASE}/v9/projects/{project_id}/env",
        headers=skill._get_headers()
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


async def get_user(skill) -> SkillResult:
    """Get current user info"""
    response = await skill.http.get(
        f"{skill.API_BASE}/v2/user",
        headers=skill._get_headers()
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
