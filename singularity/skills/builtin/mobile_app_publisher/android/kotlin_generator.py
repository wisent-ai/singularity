#!/usr/bin/env python3
"""
Kotlin App Generator - Generate native Android apps using Kotlin and Jetpack Compose.

Uses LLM to generate custom Kotlin code based on app descriptions.
The build() method delegates to kotlin_helpers.run_gradle_build().
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Callable

from . import kotlin_templates as templates
from .kotlin_helpers import run_gradle_build


class KotlinAppGenerator:
    """
    Generator for native Android apps using Kotlin and Jetpack Compose.

    Creates Gradle project scaffolds and generates custom Compose screens
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
        package_name: str,
        min_sdk: int = 24,
    ) -> Dict:
        """Create a minimal Android project scaffold with Jetpack Compose."""
        try:
            project_dir.mkdir(parents=True, exist_ok=True)

            files_created = []
            package_path = package_name.replace(".", "/")

            # Create directory structure
            app_dir = project_dir / "app"
            src_main = app_dir / "src" / "main"
            kotlin_dir = src_main / "java" / package_path
            res_dir = src_main / "res"

            for d in [kotlin_dir, res_dir / "values", res_dir / "mipmap-hdpi"]:
                d.mkdir(parents=True, exist_ok=True)

            # Write all scaffold files using templates
            file_specs = [
                (project_dir / "build.gradle.kts", templates.root_build_gradle()),
                (project_dir / "settings.gradle.kts", templates.settings_gradle(name)),
                (project_dir / "gradle.properties", templates.gradle_properties()),
                (app_dir / "build.gradle.kts", templates.app_build_gradle(package_name, min_sdk)),
                (app_dir / "proguard-rules.pro", templates.proguard_rules()),
                (src_main / "AndroidManifest.xml", templates.android_manifest(name, display_name)),
                (kotlin_dir / "MainActivity.kt", templates.main_activity_kt(package_name, name)),
                (kotlin_dir / "MainScreen.kt", templates.main_screen_kt(package_name, name, display_name)),
            ]

            # Theme files
            theme_dir = kotlin_dir / "ui" / "theme"
            theme_dir.mkdir(parents=True, exist_ok=True)

            file_specs.extend([
                (theme_dir / "Color.kt", templates.color_kt(package_name)),
                (theme_dir / "Theme.kt", templates.theme_kt(package_name, name)),
                (theme_dir / "Type.kt", templates.type_kt(package_name)),
            ])

            # Resource files
            file_specs.extend([
                (res_dir / "values" / "strings.xml", templates.strings_xml(display_name)),
                (res_dir / "values" / "themes.xml", templates.themes_xml(name)),
            ])

            # Gradle wrapper
            gradle_wrapper_dir = project_dir / "gradle" / "wrapper"
            gradle_wrapper_dir.mkdir(parents=True, exist_ok=True)

            file_specs.append(
                (gradle_wrapper_dir / "gradle-wrapper.properties", templates.gradle_wrapper_properties()),
            )

            for file_path, content in file_specs:
                file_path.write_text(content)
                files_created.append(str(file_path))

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
        """Generate a complete Android app from description using LLM."""
        if not self._llm:
            return {"success": False, "error": "LLM not configured"}

        try:
            prompt = templates.generation_prompt(description, features, theme)
            response = await self._llm(prompt)
            screens = templates.parse_llm_response(response)

            if not screens:
                return {"success": False, "error": "Failed to parse LLM response"}

            kotlin_dirs = list(project_dir.glob("app/src/main/java/**/"))
            if not kotlin_dirs:
                return {"success": False, "error": "Kotlin source directory not found"}

            kotlin_dir = max(kotlin_dirs, key=lambda p: len(p.parts))
            files_created = []

            for screen in screens:
                screen_file = kotlin_dir / f"{screen['name']}.kt"
                screen_file.write_text(screen['code'])
                files_created.append(str(screen_file))

            # Update MainScreen to use the new screens
            main_screen_path = kotlin_dir / "MainScreen.kt"
            if screens and main_screen_path.exists():
                existing = main_screen_path.read_text()
                package_match = re.search(r'package\s+([\w.]+)', existing)
                package_name = package_match.group(1) if package_match else "com.example"
                main_screen = screens[0]
                main_screen_path.write_text(f'''package {package_name}

{main_screen['code'].split("package")[0] if "package" not in main_screen['code'] else main_screen['code']}
''')

            return {"success": True, "screens": [s['name'] for s in screens], "files_created": files_created}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def add_screen(
        self,
        project_dir: Path,
        screen_name: str,
        description: str,
    ) -> Dict:
        """Add a new Jetpack Compose screen to an existing project."""
        if not self._llm:
            return {"success": False, "error": "LLM not configured"}

        try:
            kotlin_dirs = list(project_dir.glob("app/src/main/java/**/"))
            if not kotlin_dirs:
                return {"success": False, "error": "Kotlin source directory not found"}

            kotlin_dir = max(kotlin_dirs, key=lambda p: len(p.parts))

            existing_files = list(kotlin_dir.glob("*.kt"))
            package_name = "com.example"
            if existing_files:
                content = existing_files[0].read_text()
                match = re.search(r'package\s+([\w.]+)', content)
                if match:
                    package_name = match.group(1)

            prompt = templates.add_screen_prompt(screen_name, description, package_name)
            response = await self._llm(prompt)
            code = templates.extract_kotlin_code(response)

            if not code:
                return {"success": False, "error": "Failed to generate Kotlin code"}

            screen_file = kotlin_dir / f"{screen_name}.kt"
            screen_file.write_text(code)

            return {"success": True, "file_path": str(screen_file), "screen_name": screen_name}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def build(self, project_dir: str, build_type: str = "bundle") -> Dict:
        """Build Android app using Gradle."""
        return await run_gradle_build(project_dir, build_type)
