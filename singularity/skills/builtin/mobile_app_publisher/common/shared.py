#!/usr/bin/env python3
"""
Shared utilities and cross-platform methods for Mobile App Publisher.

Contains project management, asset generation, build status, and
store listing methods that serve both iOS and Android platforms.
"""

from typing import Dict, Any, Optional

from ....base import SkillAction, SkillResult


# ==================== Shared Skill Actions ====================

SHARED_ACTIONS = [
    # Project Management
    SkillAction(
        name="list_projects",
        description="List all mobile app projects in workspace",
        parameters={},
        estimated_cost=0,
        success_probability=0.95,
    ),
    SkillAction(
        name="get_project",
        description="Get details of a specific project",
        parameters={
            "project_name": {"type": "string", "required": True, "description": "Project name"},
        },
        estimated_cost=0,
        success_probability=0.95,
    ),
    # App Generation (shared)
    SkillAction(
        name="generate_assets",
        description="Generate app icon, splash screen, and screenshots",
        parameters={
            "project_name": {"type": "string", "required": True, "description": "Project name"},
            "platform": {"type": "string", "required": True, "description": "'ios' or 'android'"},
            "icon_prompt": {"type": "string", "required": True, "description": "Description of the app icon"},
            "screenshot_prompts": {"type": "array", "required": False, "description": "Prompts for store screenshots"},
        },
        estimated_cost=0.50,
        success_probability=0.90,
    ),
    # Build Management (shared)
    SkillAction(
        name="get_build_status",
        description="Get the status of a build",
        parameters={
            "project_name": {"type": "string", "required": True, "description": "Project name"},
            "platform": {"type": "string", "required": True, "description": "'ios' or 'android'"},
        },
        estimated_cost=0,
        success_probability=0.95,
    ),
    # Store Submission (shared)
    SkillAction(
        name="get_submission_status",
        description="Check the status of a store submission",
        parameters={
            "platform": {"type": "string", "required": True, "description": "'ios' or 'android'"},
            "project_name": {"type": "string", "required": True, "description": "Project name"},
        },
        estimated_cost=0,
        success_probability=0.95,
    ),
    # Store Listing
    SkillAction(
        name="update_store_listing",
        description="Update app metadata on App Store or Google Play",
        parameters={
            "platform": {"type": "string", "required": True, "description": "'ios' or 'android'"},
            "app_id": {"type": "string", "required": True, "description": "App ID / package name"},
            "name": {"type": "string", "required": False, "description": "New app name"},
            "description": {"type": "string", "required": False, "description": "New description"},
            "keywords": {"type": "array", "required": False, "description": "Search keywords (iOS only)"},
            "release_notes": {"type": "string", "required": False, "description": "What's new text"},
        },
        estimated_cost=0,
        success_probability=0.85,
    ),
    SkillAction(
        name="upload_screenshots",
        description="Upload screenshots to App Store or Google Play",
        parameters={
            "platform": {"type": "string", "required": True, "description": "'ios' or 'android'"},
            "app_id": {"type": "string", "required": True, "description": "App ID / package name"},
            "screenshot_urls": {"type": "array", "required": True, "description": "URLs of screenshots to upload"},
        },
        estimated_cost=0,
        success_probability=0.85,
    ),
]


class SharedPublisherMixin:
    """Mixin providing cross-platform project management and store listing methods."""

    async def _list_projects(self) -> SkillResult:
        """List all mobile app projects in workspace."""
        projects = []
        if self._workspace.exists():
            for project_dir in self._workspace.iterdir():
                if project_dir.is_dir():
                    project_info = {
                        "name": project_dir.name,
                        "path": str(project_dir),
                        "platforms": [],
                    }
                    if (project_dir / "ios").exists():
                        project_info["platforms"].append("ios")
                    if (project_dir / "android").exists():
                        project_info["platforms"].append("android")
                    if project_info["platforms"]:
                        projects.append(project_info)
        return SkillResult(
            success=True,
            message=f"Found {len(projects)} projects",
            data={"projects": projects},
        )

    async def _get_project(self, project_name: str) -> SkillResult:
        """Get project details."""
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        project_dir = self._workspace / project_name
        if not project_dir.exists():
            return SkillResult(success=False, message="Project not found")
        project_info = {
            "name": project_name,
            "path": str(project_dir),
            "platforms": {},
        }
        # Check iOS project
        ios_dir = project_dir / "ios"
        if ios_dir.exists():
            xcodeproj = list(ios_dir.glob("*.xcodeproj"))
            project_info["platforms"]["ios"] = {
                "path": str(ios_dir),
                "xcodeproj": str(xcodeproj[0]) if xcodeproj else None,
            }
        # Check Android project
        android_dir = project_dir / "android"
        if android_dir.exists():
            project_info["platforms"]["android"] = {
                "path": str(android_dir),
                "has_gradle": (android_dir / "build.gradle.kts").exists(),
            }
        return SkillResult(
            success=True,
            message=f"Got project: {project_name}",
            data=project_info,
        )

    async def _generate_assets(
        self,
        project_name: str,
        platform: str,
        icon_prompt: str,
        screenshot_prompts: list = None,
    ) -> SkillResult:
        """Generate app icon and screenshots."""
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        if not platform:
            return SkillResult(success=False, message="Platform required")
        if not icon_prompt:
            return SkillResult(success=False, message="Icon prompt required")
        project_dir = self._workspace / project_name / platform
        if not project_dir.exists():
            return SkillResult(success=False, message=f"{platform} project not found")
        result = await self.asset_generator.generate_assets(
            project_dir=project_dir,
            platform=platform,
            icon_prompt=icon_prompt,
            screenshot_prompts=screenshot_prompts or [],
        )
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Failed to generate assets"))
        return SkillResult(
            success=True,
            message="Assets generated successfully",
            data={
                "icon_path": result.get("icon_path"),
                "screenshot_paths": result.get("screenshot_paths", []),
            },
            cost=0.50,
        )

    async def _get_build_status(self, project_name: str, platform: str) -> SkillResult:
        """Get build status."""
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        if not platform:
            return SkillResult(success=False, message="Platform required")
        project_dir = self._workspace / project_name / platform
        if not project_dir.exists():
            return SkillResult(success=False, message=f"{platform} project not found")
        # Check for build artifacts
        if platform == "ios":
            build_dir = project_dir / "build"
            ipa_files = list(build_dir.glob("*.ipa")) if build_dir.exists() else []
            status = "completed" if ipa_files else "no_build"
            artifact = str(ipa_files[0]) if ipa_files else None
        else:
            build_dir = project_dir / "app" / "build" / "outputs"
            aab_files = list(build_dir.rglob("*.aab")) if build_dir.exists() else []
            apk_files = list(build_dir.rglob("*.apk")) if build_dir.exists() else []
            status = "completed" if (aab_files or apk_files) else "no_build"
            artifact = str(aab_files[0] if aab_files else (apk_files[0] if apk_files else None))
        return SkillResult(
            success=True,
            message=f"Build status: {status}",
            data={"status": status, "platform": platform, "artifact_path": artifact},
        )

    async def _get_submission_status(self, platform: str, project_name: str) -> SkillResult:
        """Get submission status."""
        if not platform:
            return SkillResult(success=False, message="Platform required")
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        if platform == "ios":
            result = await self.app_store.get_submission_status(project_name)
        elif platform == "android":
            result = await self.google_play.get_submission_status(project_name)
        else:
            return SkillResult(success=False, message=f"Invalid platform: {platform}")
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Failed to get status"))
        return SkillResult(
            success=True,
            message=f"Submission status: {result.get('status')}",
            data=result,
        )

    async def _update_store_listing(
        self,
        platform: str,
        app_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        keywords: list = None,
        release_notes: Optional[str] = None,
    ) -> SkillResult:
        """Update store listing metadata."""
        if not platform:
            return SkillResult(success=False, message="Platform required")
        if not app_id:
            return SkillResult(success=False, message="App ID required")
        if platform == "ios":
            result = await self.app_store.update_listing(
                app_id=app_id,
                name=name,
                description=description,
                keywords=keywords,
                release_notes=release_notes,
            )
        elif platform == "android":
            result = await self.google_play.update_listing(
                package_name=app_id,
                name=name,
                description=description,
                release_notes=release_notes,
            )
        else:
            return SkillResult(success=False, message=f"Invalid platform: {platform}")
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Failed to update listing"))
        return SkillResult(success=True, message="Store listing updated", data=result)

    async def _upload_screenshots(
        self, platform: str, app_id: str, screenshot_urls: list,
    ) -> SkillResult:
        """Upload screenshots to store."""
        if not platform:
            return SkillResult(success=False, message="Platform required")
        if not app_id:
            return SkillResult(success=False, message="App ID required")
        if not screenshot_urls:
            return SkillResult(success=False, message="Screenshot URLs required")
        if platform == "ios":
            result = await self.app_store.upload_screenshots(app_id, screenshot_urls)
        elif platform == "android":
            result = await self.google_play.upload_screenshots(app_id, screenshot_urls)
        else:
            return SkillResult(success=False, message=f"Invalid platform: {platform}")
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Failed to upload screenshots"))
        return SkillResult(
            success=True,
            message=f"Uploaded {len(screenshot_urls)} screenshots",
            data=result,
        )
