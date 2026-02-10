#!/usr/bin/env python3
"""
Asset generation helpers - icon sizing and placeholder creation.

Extracted from AssetGenerator to keep that module under 300 lines.
"""

import json
from pathlib import Path
from typing import Dict, List


# Icon size constants
IOS_ICON_SIZES = [
    (1024, 1024),  # App Store
    (180, 180),    # iPhone @3x
    (120, 120),    # iPhone @2x
    (167, 167),    # iPad Pro @2x
    (152, 152),    # iPad @2x
    (76, 76),      # iPad @1x
]

ANDROID_ICON_SIZES = [
    (512, 512),    # Play Store
    (192, 192),    # xxxhdpi
    (144, 144),    # xxhdpi
    (96, 96),      # xhdpi
    (72, 72),      # hdpi
    (48, 48),      # mdpi
]


async def create_icon_sizes(
    source_path: Path,
    project_dir: Path,
    platform: str,
) -> None:
    """Create multiple icon sizes from source."""
    try:
        from PIL import Image

        source = Image.open(source_path)

        if platform == "ios":
            sizes = IOS_ICON_SIZES
            base_dir = project_dir / "Assets.xcassets" / "AppIcon.appiconset"

            # Update Contents.json
            contents = {
                "images": [],
                "info": {"author": "xcode", "version": 1},
            }

            for size in sizes:
                resized = source.resize(size, Image.Resampling.LANCZOS)
                filename = f"icon_{size[0]}x{size[1]}.png"
                resized.save(base_dir / filename)

                contents["images"].append({
                    "filename": filename,
                    "idiom": "universal",
                    "platform": "ios",
                    "size": f"{size[0]}x{size[1]}",
                })

            (base_dir / "Contents.json").write_text(json.dumps(contents, indent=2))

        else:  # android
            # Create adaptive icon resources
            res_dir = project_dir / "app" / "src" / "main" / "res"

            densities = [
                ("mipmap-mdpi", 48),
                ("mipmap-hdpi", 72),
                ("mipmap-xhdpi", 96),
                ("mipmap-xxhdpi", 144),
                ("mipmap-xxxhdpi", 192),
            ]

            for density, size in densities:
                density_dir = res_dir / density
                density_dir.mkdir(parents=True, exist_ok=True)

                resized = source.resize((size, size), Image.Resampling.LANCZOS)
                resized.save(density_dir / "ic_launcher.png")
                resized.save(density_dir / "ic_launcher_round.png")

    except ImportError:
        pass  # PIL not available, skip resizing


async def create_placeholder_icon(
    project_dir: Path,
    platform: str,
) -> Dict:
    """Create a placeholder icon when image generation is unavailable."""
    try:
        from PIL import Image, ImageDraw

        # Create a simple gradient icon
        size = (1024, 1024)
        img = Image.new("RGB", size, "#4A90D9")

        draw = ImageDraw.Draw(img)

        # Add a simple shape
        margin = 200
        draw.ellipse(
            [margin, margin, size[0] - margin, size[1] - margin],
            fill="#FFFFFF",
        )

        # Save
        if platform == "ios":
            output_dir = project_dir / "Assets.xcassets" / "AppIcon.appiconset"
        else:
            output_dir = project_dir / "app" / "src" / "main" / "res" / "mipmap-xxxhdpi"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "icon_1024.png"
        img.save(output_path)

        await create_icon_sizes(output_path, project_dir, platform)

        return {"success": True, "path": str(output_path), "placeholder": True}

    except ImportError:
        return {
            "success": False,
            "error": "PIL required for placeholder: pip install Pillow",
        }
