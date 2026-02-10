#!/usr/bin/env python3
"""
iOS Publisher — App Store specific publishing logic.

Contains iOS project creation, app generation, Swift view management,
build orchestration, and App Store submission methods.
"""

from typing import Dict, Optional

from singularity.skills.base import SkillAction, SkillResult


# ==================== iOS Skill Actions ====================

IOS_ACTIONS = [
    SkillAction(
        name="create_ios_project",
        description="Create a new native iOS project using Swift/SwiftUI",
        parameters={
            "name": {"type": "string", "required": True, "description": "Project name"},
            "display_name": {"type": "string", "required": True, "description": "Display name for the app"},
            "bundle_id": {"type": "string", "required": False, "description": "Bundle identifier (e.g., com.example.myapp)"},
            "deployment_target": {"type": "string", "required": False, "description": "Min iOS version (default: 16.0)"},
        },
        estimated_cost=0,
        success_probability=0.95,
    ),
    SkillAction(
        name="generate_ios_app",
        description="Generate a complete iOS app from description using LLM (Swift/SwiftUI)",
        parameters={
            "project_name": {"type": "string", "required": True, "description": "Project name"},
            "description": {"type": "string", "required": True, "description": "Detailed description of the app"},
            "features": {"type": "array", "required": False, "description": "List of features to include"},
            "theme": {"type": "object", "required": False, "description": "Theme colors (primary, secondary, background)"},
        },
        estimated_cost=0.10,
        success_probability=0.85,
    ),
    SkillAction(
        name="add_swift_view",
        description="Add a new SwiftUI view to an iOS app",
        parameters={
            "project_name": {"type": "string", "required": True, "description": "Project name"},
            "view_name": {"type": "string", "required": True, "description": "Name of the view"},
            "description": {"type": "string", "required": True, "description": "Description of view functionality"},
        },
        estimated_cost=0.05,
        success_probability=0.85,
    ),
    SkillAction(
        name="build_ios",
        description="Build iOS app using Xcode (creates IPA)",
        parameters={
            "project_name": {"type": "string", "required": True, "description": "Project name"},
            "configuration": {"type": "string", "required": False, "description": "Build configuration (default: Release)"},
            "scheme": {"type": "string", "required": False, "description": "Build scheme (default: project name)"},
        },
        estimated_cost=0,
        success_probability=0.85,
    ),
    SkillAction(
        name="submit_to_app_store",
        description="Submit iOS build to App Store Connect via Fastlane",
        parameters={
            "project_name": {"type": "string", "required": True, "description": "Project name"},
            "app_name": {"type": "string", "required": True, "description": "App name for the store"},
            "description": {"type": "string", "required": True, "description": "App description"},
            "keywords": {"type": "array", "required": False, "description": "Search keywords"},
            "category": {"type": "string", "required": False, "description": "App Store category"},
            "submit_for_review": {"type": "boolean", "required": False, "description": "Auto-submit for review (default: false)"},
        },
        estimated_cost=0,
        success_probability=0.80,
    ),
]


class IOSPublisherMixin:
    """Mixin providing iOS-specific publishing methods."""

    async def _create_ios_project(
        self,
        name: str,
        display_name: str,
        bundle_id: Optional[str] = None,
        deployment_target: str = "16.0",
    ) -> SkillResult:
        """Create a new native iOS project using Swift/SwiftUI."""
        if not name:
            return SkillResult(success=False, message="Project name required")
        if not display_name:
            return SkillResult(success=False, message="Display name required")
        # Generate bundle ID if not provided
        if not bundle_id:
            bundle_id = f"com.agent.{name.lower().replace('-', '').replace('_', '')}"
        project_dir = self._workspace / name / "ios"
        result = await self.swift_generator.create_project_scaffold(
            project_dir=project_dir,
            name=name,
            display_name=display_name,
            bundle_id=bundle_id,
            deployment_target=deployment_target,
        )
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Failed to create iOS project"))
        return SkillResult(
            success=True,
            message=f"Created iOS project: {name}",
            data={
                "project_name": name,
                "display_name": display_name,
                "bundle_id": bundle_id,
                "project_dir": str(project_dir),
                "platform": "ios",
                "deployment_target": deployment_target,
            },
            asset_created={
                "type": "ios_project",
                "name": name,
                "path": str(project_dir),
            },
        )

    async def _generate_ios_app(
        self,
        project_name: str,
        description: str,
        features: list = None,
        theme: Dict = None,
    ) -> SkillResult:
        """Generate a complete iOS app from description using LLM."""
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        if not description:
            return SkillResult(success=False, message="App description required")
        if not self._llm:
            return SkillResult(success=False, message="LLM not configured - skill requires 'llm' wiring")
        project_dir = self._workspace / project_name / "ios"
        if not project_dir.exists():
            return SkillResult(
                success=False,
                message=f"iOS project not found. Create it first with create_ios_project action."
            )
        result = await self.swift_generator.generate_app(
            project_dir=project_dir,
            description=description,
            features=features or [],
            theme=theme,
        )
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Failed to generate iOS app"))
        return SkillResult(
            success=True,
            message="iOS app generated successfully",
            data={
                "project_name": project_name,
                "views": result.get("views", []),
                "files_created": result.get("files_created", []),
            },
            cost=0.10,
        )

    async def _add_swift_view(
        self,
        project_name: str,
        view_name: str,
        description: str,
    ) -> SkillResult:
        """Add a new SwiftUI view to an iOS app."""
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        if not view_name:
            return SkillResult(success=False, message="View name required")
        if not description:
            return SkillResult(success=False, message="View description required")
        if not self._llm:
            return SkillResult(success=False, message="LLM not configured")
        project_dir = self._workspace / project_name / "ios"
        if not project_dir.exists():
            return SkillResult(success=False, message="iOS project not found")
        result = await self.swift_generator.add_view(
            project_dir=project_dir,
            view_name=view_name,
            description=description,
        )
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Failed to add view"))
        return SkillResult(
            success=True,
            message=f"Added SwiftUI view: {view_name}",
            data={
                "view_name": view_name,
                "file_path": result.get("file_path"),
            },
            cost=0.05,
        )

    async def _build_ios(
        self,
        project_name: str,
        configuration: str = "Release",
        scheme: Optional[str] = None,
    ) -> SkillResult:
        """Build iOS app using Xcode."""
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        project_dir = self._workspace / project_name / "ios"
        if not project_dir.exists():
            return SkillResult(success=False, message="iOS project not found")
        result = await self.xcode.build(
            project_dir=str(project_dir),
            configuration=configuration,
            scheme=scheme or project_name,
        )
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Build failed"))
        return SkillResult(
            success=True,
            message="iOS build completed",
            data={
                "ipa_path": result.get("ipa_path"),
                "configuration": configuration,
                "build_time": result.get("build_time"),
            },
        )

    async def _submit_to_app_store(
        self,
        project_name: str,
        app_name: str,
        description: str,
        keywords: list = None,
        category: str = None,
        submit_for_review: bool = False,
    ) -> SkillResult:
        """Submit iOS build to App Store via Fastlane."""
        if not project_name:
            return SkillResult(success=False, message="Project name required")
        if not app_name:
            return SkillResult(success=False, message="App name required")
        if not description:
            return SkillResult(success=False, message="Description required")
        project_dir = self._workspace / project_name / "ios"
        if not project_dir.exists():
            return SkillResult(success=False, message="iOS project not found")
        result = await self.app_store.submit(
            project_dir=str(project_dir),
            app_name=app_name,
            description=description,
            keywords=keywords,
            category=category,
            submit_for_review=submit_for_review,
        )
        if not result["success"]:
            return SkillResult(success=False, message=result.get("error", "Submission failed"))
        return SkillResult(
            success=True,
            message="Submitted to App Store Connect",
            data={
                "app_id": result.get("app_id"),
                "version": result.get("version"),
                "status": result.get("status"),
            },
        )
