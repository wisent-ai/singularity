"""
Supabase Skill - Auth, Storage, Orgs, Secrets

Contains get_auth_config, update_auth_config, configure_oauth_provider,
list_buckets, create_bucket, list_organizations, list_secrets, create_secret.
"""

from typing import Dict
from singularity.skills.base import SkillResult


async def get_auth_config(skill, project_id: str) -> SkillResult:
    """Get auth configuration"""
    if not project_id:
        return SkillResult(success=False, message="project_id required")

    response = await skill.http.get(
        f"{skill.API_BASE}/projects/{project_id}/config/auth",
        headers=skill._get_headers()
    )

    if response.status_code == 200:
        config = response.json()
        return SkillResult(success=True, message="Auth configuration retrieved", data={"config": config})
    else:
        return SkillResult(success=False, message=f"Failed to get auth config: {response.text}")


async def update_auth_config(skill, params: Dict) -> SkillResult:
    """Update auth settings"""
    project_id = params.get("project_id")
    if not project_id:
        return SkillResult(success=False, message="project_id required")

    payload = {}
    if "site_url" in params:
        payload["site_url"] = params["site_url"]
    if "jwt_expiry" in params:
        payload["jwt_exp"] = params["jwt_expiry"]
    if "disable_signup" in params:
        payload["disable_signup"] = params["disable_signup"]
    if "external_email_enabled" in params:
        payload["external_email_enabled"] = params["external_email_enabled"]
    if "external_phone_enabled" in params:
        payload["external_phone_enabled"] = params["external_phone_enabled"]

    if not payload:
        return SkillResult(success=False, message="No settings to update")

    response = await skill.http.patch(
        f"{skill.API_BASE}/projects/{project_id}/config/auth",
        headers=skill._get_headers(),
        json=payload
    )

    if response.status_code == 200:
        return SkillResult(success=True, message="Auth configuration updated", data={"updated": list(payload.keys())})
    else:
        return SkillResult(success=False, message=f"Failed to update auth config: {response.text}")


async def configure_oauth_provider(skill, params: Dict) -> SkillResult:
    """Configure OAuth provider"""
    project_id = params.get("project_id")
    provider = params.get("provider")
    enabled = params.get("enabled")

    if not project_id or not provider or enabled is None:
        return SkillResult(success=False, message="project_id, provider, and enabled are required")

    provider_config_key = f"external_{provider}_enabled"
    payload = {provider_config_key: enabled}

    if params.get("client_id"):
        payload[f"external_{provider}_client_id"] = params["client_id"]
    if params.get("client_secret"):
        payload[f"external_{provider}_secret"] = params["client_secret"]

    response = await skill.http.patch(
        f"{skill.API_BASE}/projects/{project_id}/config/auth",
        headers=skill._get_headers(),
        json=payload
    )

    if response.status_code == 200:
        return SkillResult(
            success=True,
            message=f"{'Enabled' if enabled else 'Disabled'} {provider} OAuth",
            data={"provider": provider, "enabled": enabled}
        )
    else:
        return SkillResult(success=False, message=f"Failed to configure OAuth: {response.text}")


async def list_buckets(skill, project_id: str) -> SkillResult:
    """List storage buckets"""
    if not project_id:
        return SkillResult(success=False, message="project_id required")

    response = await skill.http.get(
        f"{skill.API_BASE}/projects/{project_id}/storage/buckets",
        headers=skill._get_headers()
    )

    if response.status_code == 200:
        buckets = response.json()
        return SkillResult(success=True, message=f"Found {len(buckets)} buckets", data={"buckets": buckets})
    else:
        return SkillResult(success=False, message=f"Failed to list buckets: {response.text}")


async def create_bucket(skill, params: Dict) -> SkillResult:
    """Create a storage bucket"""
    project_id = params.get("project_id")
    name = params.get("name")

    if not project_id or not name:
        return SkillResult(success=False, message="project_id and name required")

    payload = {"name": name, "public": params.get("public", False)}

    if params.get("file_size_limit"):
        payload["file_size_limit"] = params["file_size_limit"]
    if params.get("allowed_mime_types"):
        payload["allowed_mime_types"] = params["allowed_mime_types"]

    response = await skill.http.post(
        f"{skill.API_BASE}/projects/{project_id}/storage/buckets",
        headers=skill._get_headers(),
        json=payload
    )

    if response.status_code in [200, 201]:
        return SkillResult(success=True, message=f"Created bucket: {name}", data={"bucket": name, "public": payload["public"]})
    else:
        return SkillResult(success=False, message=f"Failed to create bucket: {response.text}")


async def list_organizations(skill) -> SkillResult:
    """List all organizations"""
    response = await skill.http.get(
        f"{skill.API_BASE}/organizations",
        headers=skill._get_headers()
    )

    if response.status_code == 200:
        orgs = response.json()
        return SkillResult(success=True, message=f"Found {len(orgs)} organizations", data={"organizations": orgs})
    else:
        return SkillResult(success=False, message=f"Failed to list organizations: {response.text}")


async def list_secrets(skill, project_id: str) -> SkillResult:
    """List project secrets (names only)"""
    if not project_id:
        return SkillResult(success=False, message="project_id required")

    response = await skill.http.get(
        f"{skill.API_BASE}/projects/{project_id}/secrets",
        headers=skill._get_headers()
    )

    if response.status_code == 200:
        secrets = response.json()
        return SkillResult(success=True, message=f"Found {len(secrets)} secrets", data={"secrets": [s.get("name") for s in secrets]})
    else:
        return SkillResult(success=False, message=f"Failed to list secrets: {response.text}")


async def create_secret(skill, project_id: str, name: str, value: str) -> SkillResult:
    """Create or update a project secret"""
    if not project_id or not name or not value:
        return SkillResult(success=False, message="project_id, name, and value required")

    response = await skill.http.post(
        f"{skill.API_BASE}/projects/{project_id}/secrets",
        headers=skill._get_headers(),
        json=[{"name": name, "value": value}]
    )

    if response.status_code in [200, 201]:
        return SkillResult(success=True, message=f"Created secret: {name}", data={"name": name})
    else:
        return SkillResult(success=False, message=f"Failed to create secret: {response.text}")
