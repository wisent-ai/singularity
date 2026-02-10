#!/usr/bin/env python3
"""
Kotlin LLM helpers - prompt builders, response parsing, and build logic.

Extracted from kotlin_templates.py and kotlin_generator.py to keep each
module under 300 lines.
"""

import asyncio
import re
import time
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# LLM prompt templates (from kotlin_templates.py)
# ---------------------------------------------------------------------------

def generation_prompt(
    description: str,
    features: List[str],
    theme: Optional[Dict],
) -> str:
    """Build the prompt for LLM-based app generation."""
    features_text = "\n".join(f"- {f}" for f in features) if features else "None specified"

    theme_text = ""
    if theme:
        theme_text = f"""
Theme:
- Primary color: {theme.get('primary', 'Purple')}
- Secondary color: {theme.get('secondary', 'PurpleGrey')}
- Background: {theme.get('background', 'White')}
"""

    return f"""Generate a complete Android app using Jetpack Compose based on this description:

{description}

Features to include:
{features_text}
{theme_text}
Requirements:
1. Create multiple Compose screens as needed
2. Use Jetpack Compose with Material 3
3. Follow Material Design guidelines
4. Include proper navigation if multiple screens
5. Use appropriate state management
6. Add helpful comments
7. Each screen should have a @Preview

Format your response as multiple Kotlin files, each starting with:
// FILE: ScreenName.kt
package com.example.app
...

Generate 2-4 screens depending on complexity. The first screen will be the main screen."""


def add_screen_prompt(screen_name: str, description: str, package_name: str) -> str:
    """Build the prompt for adding a single Compose screen via LLM."""
    return f"""Generate a Jetpack Compose screen named {screen_name} with the following functionality:

{description}

Package name: {package_name}

Requirements:
- Use Jetpack Compose with Material 3
- Follow Material Design guidelines
- Include appropriate state management (remember, mutableStateOf)
- Add helpful comments
- Include a @Preview at the bottom

Return ONLY the Kotlin code, no explanations. Start with 'package {package_name}'."""


def parse_llm_response(response: str) -> List[Dict]:
    """Parse LLM response to extract Kotlin screen definitions."""
    screens = []

    # Split by file markers
    file_pattern = r'// FILE:\s*(\w+)\.kt\s*\n(.*?)(?=// FILE:|$)'
    matches = re.findall(file_pattern, response, re.DOTALL)

    for name, code in matches:
        code = code.strip()
        if code and '@Composable' in code:
            screens.append({
                'name': name,
                'code': code,
            })

    # If no file markers found, try to extract as single screen
    if not screens and '@Composable' in response:
        # Try to find function name
        func_match = re.search(r'fun\s+(\w+)\s*\(', response)
        if func_match:
            screens.append({
                'name': func_match.group(1),
                'code': extract_kotlin_code(response),
            })

    return screens


def extract_kotlin_code(response: str) -> str:
    """Extract Kotlin code from LLM response."""
    # Try to find code block
    code_match = re.search(r'```kotlin\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # If no code block, look for package declaration
    if 'package ' in response:
        start = response.find('package ')
        return response[start:].strip()

    return response.strip()


# ---------------------------------------------------------------------------
# Build helper (from kotlin_generator.py)
# ---------------------------------------------------------------------------

async def run_gradle_build(
    project_dir: str,
    build_type: str = "bundle",
) -> Dict:
    """
    Build Android app using Gradle.

    Args:
        project_dir: Path to the Android project
        build_type: 'apk' or 'bundle'

    Returns:
        Dict with success status and output path
    """
    project_path = Path(project_dir)

    if not (project_path / "gradlew").exists() and not (project_path / "build.gradle.kts").exists():
        return {"success": False, "error": "Not a valid Gradle project"}

    start_time = time.time()

    try:
        # Determine gradle command
        if (project_path / "gradlew").exists():
            gradle_cmd = str(project_path / "gradlew")
        else:
            gradle_cmd = "gradle"

        # Build task
        if build_type == "bundle":
            task = "bundleRelease"
            output_pattern = "app/build/outputs/bundle/release/*.aab"
        else:
            task = "assembleRelease"
            output_pattern = "app/build/outputs/apk/release/*.apk"

        process = await asyncio.create_subprocess_exec(
            gradle_cmd, task,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_path),
        )

        stdout, stderr = await process.communicate()

        build_time = time.time() - start_time

        if process.returncode != 0:
            return {
                "success": False,
                "error": f"Build failed: {stderr.decode()[-500:]}",
            }

        # Find output file
        output_files = list(project_path.glob(output_pattern))

        return {
            "success": True,
            "output_path": str(output_files[0]) if output_files else None,
            "build_type": build_type,
            "build_time": round(build_time, 2),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Theme templates (from kotlin_templates.py)
# ---------------------------------------------------------------------------

def color_kt(package_name: str) -> str:
    return f'''package {package_name}.ui.theme

import androidx.compose.ui.graphics.Color

val Purple80 = Color(0xFFD0BCFF)
val PurpleGrey80 = Color(0xFFCCC2DC)
val Pink80 = Color(0xFFEFB8C8)

val Purple40 = Color(0xFF6650a4)
val PurpleGrey40 = Color(0xFF625b71)
val Pink40 = Color(0xFF7D5260)
'''


def theme_kt(package_name: str, name: str) -> str:
    return f'''package {package_name}.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val DarkColorScheme = darkColorScheme(
    primary = Purple80,
    secondary = PurpleGrey80,
    tertiary = Pink80
)

private val LightColorScheme = lightColorScheme(
    primary = Purple40,
    secondary = PurpleGrey40,
    tertiary = Pink40
)

@Composable
fun {name}Theme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {{
    val colorScheme = when {{
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {{
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }}
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }}
    val view = LocalView.current
    if (!view.isInEditMode) {{
        SideEffect {{
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = darkTheme
        }}
    }}

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}}
'''


def type_kt(package_name: str) -> str:
    return f'''package {package_name}.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val Typography = Typography(
    bodyLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.5.sp
    )
)
'''
