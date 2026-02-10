#!/usr/bin/env python3
"""
Swift App Generator - Generate native iOS apps using Swift and SwiftUI.

Uses LLM to generate custom Swift code based on app descriptions.
LLM response parsing and prompt building are in swift_templates.py.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable

from .swift_templates import (
    add_view_prompt,
    app_file_template,
    build_generation_prompt,
    content_view_template,
    content_view_redirect_template,
    extract_swift_code,
    info_plist_template,
    parse_llm_response,
    pbxproj_template,
)


class SwiftAppGenerator:
    """
    Generator for native iOS apps using Swift and SwiftUI.

    Creates Xcode project scaffolds and generates custom SwiftUI views
    based on natural language descriptions using LLM.
    """

    def __init__(self):
        self._llm: Optional[Callable] = None

    def set_llm(self, llm_callable: Callable):
        """Set the LLM callable for code generation."""
        self._llm = llm_callable

    async def create_project_scaffold(
        self,
        project_dir: Path,
        name: str,
        display_name: str,
        bundle_id: str,
        deployment_target: str = "16.0",
    ) -> Dict:
        """Create a minimal iOS project scaffold."""
        try:
            project_dir.mkdir(parents=True, exist_ok=True)

            sources_dir = project_dir / name
            sources_dir.mkdir(exist_ok=True)

            assets_dir = sources_dir / "Assets.xcassets"
            assets_dir.mkdir(exist_ok=True)

            appicon_dir = assets_dir / "AppIcon.appiconset"
            appicon_dir.mkdir(exist_ok=True)

            files_created = []

            # Main App file
            app_file = sources_dir / f"{name}App.swift"
            app_file.write_text(app_file_template(name))
            files_created.append(str(app_file))

            # ContentView
            content_view = sources_dir / "ContentView.swift"
            content_view.write_text(content_view_template(display_name))
            files_created.append(str(content_view))

            # Info.plist
            info_plist = sources_dir / "Info.plist"
            info_plist.write_text(info_plist_template(display_name, bundle_id))
            files_created.append(str(info_plist))

            # Assets catalog contents
            assets_contents = assets_dir / "Contents.json"
            assets_contents.write_text(json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2))
            files_created.append(str(assets_contents))

            # AppIcon contents
            appicon_contents = appicon_dir / "Contents.json"
            appicon_contents.write_text(json.dumps({
                "images": [{"idiom": "universal", "platform": "ios", "size": "1024x1024"}],
                "info": {"author": "xcode", "version": 1},
            }, indent=2))
            files_created.append(str(appicon_contents))

            # xcodeproj
            xcodeproj_dir = project_dir / f"{name}.xcodeproj"
            xcodeproj_dir.mkdir(exist_ok=True)

            pbxproj = pbxproj_template(name=name, bundle_id=bundle_id, deployment_target=deployment_target)
            pbxproj_path = xcodeproj_dir / "project.pbxproj"
            pbxproj_path.write_text(pbxproj)
            files_created.append(str(pbxproj_path))

            return {"success": True, "project_dir": str(project_dir), "files_created": files_created}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate_app(
        self,
        project_dir: Path,
        description: str,
        features: List[str],
        theme: Optional[Dict] = None,
    ) -> Dict:
        """Generate a complete iOS app from description using LLM."""
        if not self._llm:
            return {"success": False, "error": "LLM not configured"}

        try:
            prompt = build_generation_prompt(description, features, theme)
            response = await self._llm(prompt)
            views = parse_llm_response(response)

            if not views:
                return {"success": False, "error": "Failed to parse LLM response"}

            sources_dirs = list(project_dir.glob("*/"))
            if not sources_dirs:
                return {"success": False, "error": "Project sources directory not found"}

            sources_dir = sources_dirs[0]
            files_created = []

            for view in views:
                view_file = sources_dir / f"{view['name']}.swift"
                view_file.write_text(view['code'])
                files_created.append(str(view_file))

            content_view_path = sources_dir / "ContentView.swift"
            if views and content_view_path.exists():
                main_view = views[0]
                content_view_path.write_text(content_view_redirect_template(main_view['name']))

            return {"success": True, "views": [v['name'] for v in views], "files_created": files_created}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def add_view(
        self,
        project_dir: Path,
        view_name: str,
        description: str,
    ) -> Dict:
        """Add a new SwiftUI view to an existing project."""
        if not self._llm:
            return {"success": False, "error": "LLM not configured"}

        try:
            prompt = add_view_prompt(view_name, description)
            response = await self._llm(prompt)
            code = extract_swift_code(response)

            if not code:
                return {"success": False, "error": "Failed to generate Swift code"}

            sources_dirs = list(project_dir.glob("*/"))
            if not sources_dirs:
                return {"success": False, "error": "Project sources directory not found"}

            sources_dir = sources_dirs[0]
            view_file = sources_dir / f"{view_name}.swift"
            view_file.write_text(code)

            return {"success": True, "file_path": str(view_file), "view_name": view_name}

        except Exception as e:
            return {"success": False, "error": str(e)}
