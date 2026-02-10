#!/usr/bin/env python3
"""
Asset Generator - Generate app icons and screenshots using AI.

Uses the media_generation skill (Vertex AI) to create app assets.
Icon sizing and placeholder creation are in asset_helpers.py.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from .asset_helpers import create_icon_sizes, create_placeholder_icon


class AssetGenerator:
    """
    Generator for app icons and screenshots using AI image generation.

    Creates properly sized assets for iOS and Android app stores.
    """

    # Screenshot sizes for stores
    IOS_SCREENSHOT_SIZES = {
        "iphone_6.7": (1290, 2796),   # iPhone 15 Pro Max
        "iphone_6.5": (1284, 2778),   # iPhone 14 Plus
        "iphone_5.5": (1242, 2208),   # iPhone 8 Plus
        "ipad_12.9": (2048, 2732),    # iPad Pro 12.9"
    }

    ANDROID_SCREENSHOT_SIZES = {
        "phone": (1080, 1920),
        "tablet_7": (1200, 1920),
        "tablet_10": (1920, 1200),
    }

    def __init__(self):
        self._vertex_project = os.environ.get("VERTEX_PROJECT")
        self._vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")

    async def generate_assets(
        self,
        project_dir: Path,
        platform: str,
        icon_prompt: str,
        screenshot_prompts: List[str] = None,
    ) -> Dict:
        """Generate app icon and screenshots."""
        try:
            results = {
                "success": True,
                "icon_path": None,
                "screenshot_paths": [],
            }

            icon_result = await self._generate_icon(project_dir, platform, icon_prompt)
            if icon_result["success"]:
                results["icon_path"] = icon_result["path"]
            else:
                results["icon_error"] = icon_result.get("error")

            if screenshot_prompts:
                for i, prompt in enumerate(screenshot_prompts):
                    screenshot_result = await self._generate_screenshot(
                        project_dir, platform, prompt, i
                    )
                    if screenshot_result["success"]:
                        results["screenshot_paths"].append(screenshot_result["path"])

            return results

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_icon(self, project_dir: Path, platform: str, prompt: str) -> Dict:
        """Generate an app icon."""
        enhanced_prompt = (
            f"App icon design: {prompt}\n\nRequirements:\n"
            "- Simple, recognizable design\n- Works at small sizes\n"
            "- No text or small details\n- Centered composition\n"
            "- Vibrant colors\n- Professional mobile app icon style\n"
            "- Square format with slight rounded corners\n- Clean, modern aesthetic"
        )

        try:
            if self._vertex_project:
                return await self._generate_with_vertex(
                    project_dir, platform, enhanced_prompt, "icon"
                )
            return await create_placeholder_icon(project_dir, platform)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_screenshot(self, project_dir: Path, platform: str, prompt: str, index: int) -> Dict:
        """Generate a store screenshot."""
        if platform == "ios":
            size = self.IOS_SCREENSHOT_SIZES["iphone_6.7"]
        else:
            size = self.ANDROID_SCREENSHOT_SIZES["phone"]

        enhanced_prompt = (
            f"Mobile app screenshot: {prompt}\n\nRequirements:\n"
            f"- Portrait orientation ({size[0]}x{size[1]} pixels)\n"
            "- Shows a mobile phone screen\n- Clean UI design\n"
            "- Professional app screenshot style\n- High quality, crisp rendering"
        )

        try:
            if self._vertex_project:
                return await self._generate_with_vertex(
                    project_dir, platform, enhanced_prompt, f"screenshot_{index}"
                )
            return {"success": False, "error": "Vertex AI not configured"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_with_vertex(self, project_dir: Path, platform: str, prompt: str, asset_type: str) -> Dict:
        """Generate image using Vertex AI Imagen."""
        try:
            import vertexai
            from vertexai.preview.vision_models import ImageGenerationModel

            vertexai.init(project=self._vertex_project, location=self._vertex_location)
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

            response = model.generate_images(
                prompt=prompt, number_of_images=1,
                aspect_ratio="1:1" if "icon" in asset_type else "9:16",
                safety_filter_level="block_few",
            )

            if not response.images:
                return {"success": False, "error": "No image generated"}

            # Determine output path
            if platform == "ios":
                if "icon" in asset_type:
                    output_dir = project_dir / "Assets.xcassets" / "AppIcon.appiconset"
                else:
                    output_dir = project_dir / "Screenshots"
            else:
                if "icon" in asset_type:
                    output_dir = project_dir / "app" / "src" / "main" / "res" / "mipmap-xxxhdpi"
                else:
                    output_dir = project_dir / "fastlane" / "metadata" / "android" / "en-US" / "images" / "phoneScreenshots"

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / ("icon_1024.png" if "icon" in asset_type else f"{asset_type}.png")
            response.images[0].save(str(output_path))

            if "icon" in asset_type:
                await create_icon_sizes(output_path, project_dir, platform)

            return {"success": True, "path": str(output_path)}

        except ImportError:
            return {"success": False, "error": "Vertex AI SDK required: pip install google-cloud-aiplatform"}
        except Exception as e:
            return {"success": False, "error": str(e)}
