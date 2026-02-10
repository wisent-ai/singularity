"""Mobile App Publisher Skill - create and publish native apps."""

import os
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult
from .common.xcode_client import XcodeClient
from .ios.app_store_client import AppStoreClient
from .android.google_play_client import GooglePlayClient
from .ios.swift_generator import SwiftAppGenerator
from .android.kotlin_generator import KotlinAppGenerator
from .common.asset_generator import AssetGenerator
from . import handlers


def _action(name, desc, params=None, cost=0, prob=0.95):
    """Helper to build SkillAction compactly."""
    return SkillAction(name=name, description=desc, parameters=params or {}, estimated_cost=cost, success_probability=prob)


def _p(name, typ, req=True, desc=""):
    """Helper to build parameter dict entry."""
    return {name: {"type": typ, "required": req, "description": desc}}


class MobileAppPublisherSkill(Skill):
    """Create and publish native apps to App Store and Google Play."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        credentials = credentials or {}
        self.xcode = XcodeClient()
        self.app_store = AppStoreClient(
            key_id=credentials.get("APP_STORE_CONNECT_KEY_ID") or os.environ.get("APP_STORE_CONNECT_KEY_ID"),
            issuer_id=credentials.get("APP_STORE_CONNECT_ISSUER_ID") or os.environ.get("APP_STORE_CONNECT_ISSUER_ID"),
            private_key=credentials.get("APP_STORE_CONNECT_PRIVATE_KEY") or os.environ.get("APP_STORE_CONNECT_PRIVATE_KEY"),
            team_id=credentials.get("APPLE_TEAM_ID") or os.environ.get("APPLE_TEAM_ID"))
        self.google_play = GooglePlayClient(
            service_account_json=credentials.get("GOOGLE_PLAY_SERVICE_ACCOUNT") or os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT"))
        self.swift_generator = SwiftAppGenerator()
        self.kotlin_generator = KotlinAppGenerator()
        self.asset_generator = AssetGenerator()
        self._llm: Optional[Callable] = None
        self._workspace = Path(os.environ.get("AGENT_WORKSPACE", "/tmp/agent_apps"))
        self._workspace.mkdir(parents=True, exist_ok=True)

    def set_llm(self, llm_callable: Callable):
        self._llm = llm_callable
        self.swift_generator.set_llm(llm_callable)
        self.kotlin_generator.set_llm(llm_callable)

    @property
    def manifest(self) -> SkillManifest:
        np = _p("name", "string", True, "Project name")
        dn = _p("display_name", "string", True, "Display name")
        pn = _p("project_name", "string", True, "Project name")
        plat = _p("platform", "string", True, "'ios' or 'android'")
        desc_p = _p("description", "string", True, "Description")
        app_id = _p("app_id", "string", True, "App ID / package name")
        return SkillManifest(
            skill_id="mobile_app_publisher", name="Mobile App Publisher", version="1.0.0",
            category="dev", description="Create native Swift/Kotlin apps and publish to App Store and Google Play",
            required_credentials=["APP_STORE_CONNECT_KEY_ID", "APP_STORE_CONNECT_ISSUER_ID",
                                  "APP_STORE_CONNECT_PRIVATE_KEY", "GOOGLE_PLAY_SERVICE_ACCOUNT"],
            install_cost=0,
            actions=[
                _action("create_ios_project", "Create a new native iOS project using Swift/SwiftUI",
                        {**np, **dn, **_p("bundle_id", "string", False, "Bundle ID"), **_p("deployment_target", "string", False, "Min iOS")}),
                _action("create_android_project", "Create a new native Android project using Kotlin/Compose",
                        {**np, **dn, **_p("package_name", "string", False, "Package name"), **_p("min_sdk", "integer", False, "Min SDK")}),
                _action("list_projects", "List all mobile app projects in workspace"),
                _action("get_project", "Get details of a specific project", pn),
                _action("generate_ios_app", "Generate a complete iOS app from description using LLM",
                        {**pn, **desc_p, **_p("features", "array", False, "Features"), **_p("theme", "object", False, "Theme")}, 0.10, 0.85),
                _action("generate_android_app", "Generate a complete Android app from description using LLM",
                        {**pn, **desc_p, **_p("features", "array", False, "Features"), **_p("theme", "object", False, "Theme")}, 0.10, 0.85),
                _action("add_swift_view", "Add a new SwiftUI view to an iOS app",
                        {**pn, **_p("view_name", "string", True, "View name"), **desc_p}, 0.05, 0.85),
                _action("add_compose_screen", "Add a new Compose screen to an Android app",
                        {**pn, **_p("screen_name", "string", True, "Screen name"), **desc_p}, 0.05, 0.85),
                _action("generate_assets", "Generate app icon, splash screen, and screenshots",
                        {**pn, **plat, **_p("icon_prompt", "string", True, "Icon description"),
                         **_p("screenshot_prompts", "array", False, "Screenshot prompts")}, 0.50, 0.90),
                _action("build_ios", "Build iOS app using Xcode",
                        {**pn, **_p("configuration", "string", False, "Build config"), **_p("scheme", "string", False, "Scheme")}, 0, 0.85),
                _action("build_android", "Build Android app using Gradle",
                        {**pn, **_p("build_type", "string", False, "'apk' or 'bundle'")}, 0, 0.85),
                _action("get_build_status", "Get the status of a build", {**pn, **plat}),
                _action("submit_to_app_store", "Submit iOS build to App Store Connect",
                        {**pn, **_p("app_name", "string", True, "App name"), **desc_p,
                         **_p("keywords", "array", False, "Keywords"), **_p("category", "string", False, "Category"),
                         **_p("submit_for_review", "boolean", False, "Auto-submit")}, 0, 0.80),
                _action("submit_to_google_play", "Submit Android build to Google Play",
                        {**pn, **_p("app_name", "string", True, "App name"), **desc_p,
                         **_p("track", "string", False, "Release track")}, 0, 0.80),
                _action("get_submission_status", "Check store submission status", {**plat, **pn}),
                _action("update_store_listing", "Update app metadata on store",
                        {**plat, **app_id, **_p("name", "string", False, "New name"), **_p("description", "string", False, "New desc"),
                         **_p("keywords", "array", False, "Keywords"), **_p("release_notes", "string", False, "What's new")}, 0, 0.85),
                _action("upload_screenshots", "Upload screenshots to store",
                        {**plat, **app_id, **_p("screenshot_urls", "array", True, "Screenshot URLs")}, 0, 0.85),
            ])

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            dispatch = {
                "create_ios_project": lambda: handlers.create_ios_project(self, params.get("name"), params.get("display_name"), params.get("bundle_id"), params.get("deployment_target", "16.0")),
                "create_android_project": lambda: handlers.create_android_project(self, params.get("name"), params.get("display_name"), params.get("package_name"), params.get("min_sdk", 24)),
                "list_projects": lambda: handlers.list_projects(self),
                "get_project": lambda: handlers.get_project(self, params.get("project_name")),
                "generate_ios_app": lambda: handlers.generate_ios_app(self, params.get("project_name"), params.get("description"), params.get("features", []), params.get("theme")),
                "generate_android_app": lambda: handlers.generate_android_app(self, params.get("project_name"), params.get("description"), params.get("features", []), params.get("theme")),
                "add_swift_view": lambda: handlers.add_swift_view(self, params.get("project_name"), params.get("view_name"), params.get("description")),
                "add_compose_screen": lambda: handlers.add_compose_screen(self, params.get("project_name"), params.get("screen_name"), params.get("description")),
                "generate_assets": lambda: handlers.generate_assets(self, params.get("project_name"), params.get("platform"), params.get("icon_prompt"), params.get("screenshot_prompts")),
                "build_ios": lambda: handlers.build_ios(self, params.get("project_name"), params.get("configuration", "Release"), params.get("scheme")),
                "build_android": lambda: handlers.build_android(self, params.get("project_name"), params.get("build_type", "bundle")),
                "get_build_status": lambda: handlers.get_build_status(self, params.get("project_name"), params.get("platform")),
                "submit_to_app_store": lambda: handlers.submit_to_app_store(self, params.get("project_name"), params.get("app_name"), params.get("description"), params.get("keywords"), params.get("category"), params.get("submit_for_review", False)),
                "submit_to_google_play": lambda: handlers.submit_to_google_play(self, params.get("project_name"), params.get("app_name"), params.get("description"), params.get("track", "internal")),
                "get_submission_status": lambda: handlers.get_submission_status(self, params.get("platform"), params.get("project_name")),
                "update_store_listing": lambda: handlers.update_store_listing(self, params.get("platform"), params.get("app_id"), params.get("name"), params.get("description"), params.get("keywords"), params.get("release_notes")),
                "upload_screenshots": lambda: handlers.upload_screenshots(self, params.get("platform"), params.get("app_id"), params.get("screenshot_urls")),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {str(e)}")
