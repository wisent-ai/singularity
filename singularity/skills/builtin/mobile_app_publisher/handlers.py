"""Handler functions for MobileAppPublisherSkill actions."""

from typing import Dict, Optional

from singularity.skills.base import SkillResult


async def create_ios_project(skill, name, display_name, bundle_id=None, deployment_target="16.0"):
    if not name:
        return SkillResult(success=False, message="Project name required")
    if not display_name:
        return SkillResult(success=False, message="Display name required")
    if not bundle_id:
        bundle_id = f"com.agent.{name.lower().replace('-', '').replace('_', '')}"
    project_dir = skill._workspace / name / "ios"
    result = await skill.swift_generator.create_project_scaffold(
        project_dir=project_dir, name=name, display_name=display_name,
        bundle_id=bundle_id, deployment_target=deployment_target)
    if not result["success"]:
        return SkillResult(success=False, message=result.get("error", "Failed to create iOS project"))
    return SkillResult(success=True, message=f"Created iOS project: {name}",
        data={"project_name": name, "display_name": display_name, "bundle_id": bundle_id,
              "project_dir": str(project_dir), "platform": "ios", "deployment_target": deployment_target},
        asset_created={"type": "ios_project", "name": name, "path": str(project_dir)})


async def create_android_project(skill, name, display_name, package_name=None, min_sdk=24):
    if not name:
        return SkillResult(success=False, message="Project name required")
    if not display_name:
        return SkillResult(success=False, message="Display name required")
    if not package_name:
        package_name = f"com.agent.{name.lower().replace('-', '').replace('_', '')}"
    project_dir = skill._workspace / name / "android"
    result = await skill.kotlin_generator.create_project_scaffold(
        project_dir=project_dir, name=name, display_name=display_name,
        package_name=package_name, min_sdk=min_sdk)
    if not result["success"]:
        return SkillResult(success=False, message=result.get("error", "Failed to create Android project"))
    return SkillResult(success=True, message=f"Created Android project: {name}",
        data={"project_name": name, "display_name": display_name, "package_name": package_name,
              "project_dir": str(project_dir), "platform": "android", "min_sdk": min_sdk},
        asset_created={"type": "android_project", "name": name, "path": str(project_dir)})


async def list_projects(skill):
    projects = []
    if skill._workspace.exists():
        for d in skill._workspace.iterdir():
            if d.is_dir():
                info = {"name": d.name, "path": str(d), "platforms": []}
                if (d / "ios").exists(): info["platforms"].append("ios")
                if (d / "android").exists(): info["platforms"].append("android")
                if info["platforms"]: projects.append(info)
    return SkillResult(success=True, message=f"Found {len(projects)} projects", data={"projects": projects})


async def get_project(skill, project_name):
    if not project_name:
        return SkillResult(success=False, message="Project name required")
    d = skill._workspace / project_name
    if not d.exists():
        return SkillResult(success=False, message="Project not found")
    info = {"name": project_name, "path": str(d), "platforms": {}}
    ios_dir = d / "ios"
    if ios_dir.exists():
        xcp = list(ios_dir.glob("*.xcodeproj"))
        info["platforms"]["ios"] = {"path": str(ios_dir), "xcodeproj": str(xcp[0]) if xcp else None}
    android_dir = d / "android"
    if android_dir.exists():
        info["platforms"]["android"] = {"path": str(android_dir), "has_gradle": (android_dir / "build.gradle.kts").exists()}
    return SkillResult(success=True, message=f"Got project: {project_name}", data=info)


async def generate_ios_app(skill, project_name, description, features=None, theme=None):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not description: return SkillResult(success=False, message="App description required")
    if not skill._llm: return SkillResult(success=False, message="LLM not configured")
    d = skill._workspace / project_name / "ios"
    if not d.exists(): return SkillResult(success=False, message="iOS project not found. Create it first.")
    result = await skill.swift_generator.generate_app(project_dir=d, description=description, features=features or [], theme=theme)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message="iOS app generated",
        data={"project_name": project_name, "views": result.get("views", []), "files_created": result.get("files_created", [])}, cost=0.10)


async def generate_android_app(skill, project_name, description, features=None, theme=None):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not description: return SkillResult(success=False, message="App description required")
    if not skill._llm: return SkillResult(success=False, message="LLM not configured")
    d = skill._workspace / project_name / "android"
    if not d.exists(): return SkillResult(success=False, message="Android project not found. Create it first.")
    result = await skill.kotlin_generator.generate_app(project_dir=d, description=description, features=features or [], theme=theme)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message="Android app generated",
        data={"project_name": project_name, "screens": result.get("screens", []), "files_created": result.get("files_created", [])}, cost=0.10)


async def add_swift_view(skill, project_name, view_name, description):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not view_name: return SkillResult(success=False, message="View name required")
    if not description: return SkillResult(success=False, message="View description required")
    if not skill._llm: return SkillResult(success=False, message="LLM not configured")
    d = skill._workspace / project_name / "ios"
    if not d.exists(): return SkillResult(success=False, message="iOS project not found")
    result = await skill.swift_generator.add_view(project_dir=d, view_name=view_name, description=description)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message=f"Added SwiftUI view: {view_name}",
        data={"view_name": view_name, "file_path": result.get("file_path")}, cost=0.05)


async def add_compose_screen(skill, project_name, screen_name, description):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not screen_name: return SkillResult(success=False, message="Screen name required")
    if not description: return SkillResult(success=False, message="Screen description required")
    if not skill._llm: return SkillResult(success=False, message="LLM not configured")
    d = skill._workspace / project_name / "android"
    if not d.exists(): return SkillResult(success=False, message="Android project not found")
    result = await skill.kotlin_generator.add_screen(project_dir=d, screen_name=screen_name, description=description)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message=f"Added Compose screen: {screen_name}",
        data={"screen_name": screen_name, "file_path": result.get("file_path")}, cost=0.05)


async def generate_assets(skill, project_name, platform, icon_prompt, screenshot_prompts=None):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not platform: return SkillResult(success=False, message="Platform required")
    if not icon_prompt: return SkillResult(success=False, message="Icon prompt required")
    d = skill._workspace / project_name / platform
    if not d.exists(): return SkillResult(success=False, message=f"{platform} project not found")
    result = await skill.asset_generator.generate_assets(project_dir=d, platform=platform,
        icon_prompt=icon_prompt, screenshot_prompts=screenshot_prompts or [])
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message="Assets generated",
        data={"icon_path": result.get("icon_path"), "screenshot_paths": result.get("screenshot_paths", [])}, cost=0.50)


async def build_ios(skill, project_name, configuration="Release", scheme=None):
    if not project_name: return SkillResult(success=False, message="Project name required")
    d = skill._workspace / project_name / "ios"
    if not d.exists(): return SkillResult(success=False, message="iOS project not found")
    result = await skill.xcode.build(project_dir=str(d), configuration=configuration, scheme=scheme or project_name)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Build failed"))
    return SkillResult(success=True, message="iOS build completed",
        data={"ipa_path": result.get("ipa_path"), "configuration": configuration, "build_time": result.get("build_time")})


async def build_android(skill, project_name, build_type="bundle"):
    if not project_name: return SkillResult(success=False, message="Project name required")
    d = skill._workspace / project_name / "android"
    if not d.exists(): return SkillResult(success=False, message="Android project not found")
    result = await skill.kotlin_generator.build(project_dir=str(d), build_type=build_type)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Build failed"))
    return SkillResult(success=True, message="Android build completed",
        data={"output_path": result.get("output_path"), "build_type": build_type, "build_time": result.get("build_time")})


async def get_build_status(skill, project_name, platform):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not platform: return SkillResult(success=False, message="Platform required")
    d = skill._workspace / project_name / platform
    if not d.exists(): return SkillResult(success=False, message=f"{platform} project not found")
    if platform == "ios":
        bd = d / "build"
        ipas = list(bd.glob("*.ipa")) if bd.exists() else []
        status, artifact = ("completed", str(ipas[0])) if ipas else ("no_build", None)
    else:
        bd = d / "app" / "build" / "outputs"
        aabs = list(bd.rglob("*.aab")) if bd.exists() else []
        apks = list(bd.rglob("*.apk")) if bd.exists() else []
        files = aabs or apks
        status, artifact = ("completed", str(files[0])) if files else ("no_build", None)
    return SkillResult(success=True, message=f"Build status: {status}",
        data={"status": status, "platform": platform, "artifact_path": artifact})


async def submit_to_app_store(skill, project_name, app_name, description, keywords=None, category=None, submit_for_review=False):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not app_name: return SkillResult(success=False, message="App name required")
    if not description: return SkillResult(success=False, message="Description required")
    d = skill._workspace / project_name / "ios"
    if not d.exists(): return SkillResult(success=False, message="iOS project not found")
    result = await skill.app_store.submit(project_dir=str(d), app_name=app_name, description=description,
        keywords=keywords, category=category, submit_for_review=submit_for_review)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Submission failed"))
    return SkillResult(success=True, message="Submitted to App Store Connect",
        data={"app_id": result.get("app_id"), "version": result.get("version"), "status": result.get("status")})


async def submit_to_google_play(skill, project_name, app_name, description, track="internal"):
    if not project_name: return SkillResult(success=False, message="Project name required")
    if not app_name: return SkillResult(success=False, message="App name required")
    if not description: return SkillResult(success=False, message="Description required")
    d = skill._workspace / project_name / "android"
    if not d.exists(): return SkillResult(success=False, message="Android project not found")
    result = await skill.google_play.submit(project_dir=str(d), app_name=app_name, description=description, track=track)
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Submission failed"))
    return SkillResult(success=True, message=f"Submitted to Google Play ({track} track)",
        data={"package_name": result.get("package_name"), "version_code": result.get("version_code"), "track": track, "status": result.get("status")})


async def get_submission_status(skill, platform, project_name):
    if not platform: return SkillResult(success=False, message="Platform required")
    if not project_name: return SkillResult(success=False, message="Project name required")
    if platform == "ios": result = await skill.app_store.get_submission_status(project_name)
    elif platform == "android": result = await skill.google_play.get_submission_status(project_name)
    else: return SkillResult(success=False, message=f"Invalid platform: {platform}")
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message=f"Submission status: {result.get('status')}", data=result)


async def update_store_listing(skill, platform, app_id, name=None, description=None, keywords=None, release_notes=None):
    if not platform: return SkillResult(success=False, message="Platform required")
    if not app_id: return SkillResult(success=False, message="App ID required")
    if platform == "ios":
        result = await skill.app_store.update_listing(app_id=app_id, name=name, description=description, keywords=keywords, release_notes=release_notes)
    elif platform == "android":
        result = await skill.google_play.update_listing(package_name=app_id, name=name, description=description, release_notes=release_notes)
    else: return SkillResult(success=False, message=f"Invalid platform: {platform}")
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message="Store listing updated", data=result)


async def upload_screenshots(skill, platform, app_id, screenshot_urls):
    if not platform: return SkillResult(success=False, message="Platform required")
    if not app_id: return SkillResult(success=False, message="App ID required")
    if not screenshot_urls: return SkillResult(success=False, message="Screenshot URLs required")
    if platform == "ios": result = await skill.app_store.upload_screenshots(app_id, screenshot_urls)
    elif platform == "android": result = await skill.google_play.upload_screenshots(app_id, screenshot_urls)
    else: return SkillResult(success=False, message=f"Invalid platform: {platform}")
    if not result["success"]: return SkillResult(success=False, message=result.get("error", "Failed"))
    return SkillResult(success=True, message=f"Uploaded {len(screenshot_urls)} screenshots", data=result)
