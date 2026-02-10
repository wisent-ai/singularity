#!/usr/bin/env python3
"""
Xcode Client - Build iOS apps using xcodebuild and create IPAs.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Optional


class XcodeClient:
    """
    Client for building iOS apps using Xcode command-line tools.

    Uses xcodebuild for compilation and creates distributable IPA files.
    """

    def __init__(self):
        self._build_processes: Dict[str, asyncio.subprocess.Process] = {}

    async def build(
        self,
        project_dir: str,
        configuration: str = "Release",
        scheme: Optional[str] = None,
        destination: str = "generic/platform=iOS",
    ) -> Dict:
        """
        Build an iOS project using xcodebuild.

        Args:
            project_dir: Path to the iOS project directory
            configuration: Build configuration (Debug/Release)
            scheme: Build scheme (defaults to project name)
            destination: Build destination

        Returns:
            Dict with success status, ipa_path, and build_time
        """
        project_path = Path(project_dir)

        # Find xcodeproj or xcworkspace
        xcworkspace = list(project_path.glob("*.xcworkspace"))
        xcodeproj = list(project_path.glob("*.xcodeproj"))

        if not xcworkspace and not xcodeproj:
            return {"success": False, "error": "No Xcode project found"}

        # Determine scheme name
        if not scheme:
            if xcworkspace:
                scheme = xcworkspace[0].stem
            elif xcodeproj:
                scheme = xcodeproj[0].stem

        # Build directory
        build_dir = project_path / "build"
        archive_path = build_dir / f"{scheme}.xcarchive"
        export_path = build_dir / "export"

        build_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            # Step 1: Archive the project
            if xcworkspace:
                archive_cmd = [
                    "xcodebuild",
                    "-workspace", str(xcworkspace[0]),
                    "-scheme", scheme,
                    "-configuration", configuration,
                    "-destination", destination,
                    "-archivePath", str(archive_path),
                    "archive",
                    "CODE_SIGN_IDENTITY=-",
                    "CODE_SIGNING_REQUIRED=NO",
                    "CODE_SIGNING_ALLOWED=NO",
                ]
            else:
                archive_cmd = [
                    "xcodebuild",
                    "-project", str(xcodeproj[0]),
                    "-scheme", scheme,
                    "-configuration", configuration,
                    "-destination", destination,
                    "-archivePath", str(archive_path),
                    "archive",
                    "CODE_SIGN_IDENTITY=-",
                    "CODE_SIGNING_REQUIRED=NO",
                    "CODE_SIGNING_ALLOWED=NO",
                ]

            process = await asyncio.create_subprocess_exec(
                *archive_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_path),
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return {
                    "success": False,
                    "error": f"Archive failed: {stderr.decode()[-500:]}",
                }

            # Step 2: Export IPA (for distribution)
            export_options_path = project_path / "ExportOptions.plist"
            if not export_options_path.exists():
                # Create default export options
                export_options = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>app-store</string>
    <key>uploadSymbols</key>
    <true/>
    <key>compileBitcode</key>
    <false/>
</dict>
</plist>"""
                export_options_path.write_text(export_options)

            export_cmd = [
                "xcodebuild",
                "-exportArchive",
                "-archivePath", str(archive_path),
                "-exportPath", str(export_path),
                "-exportOptionsPlist", str(export_options_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *export_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_path),
            )

            stdout, stderr = await process.communicate()

            build_time = time.time() - start_time

            # Find the IPA file
            ipa_files = list(export_path.glob("*.ipa"))

            if ipa_files:
                return {
                    "success": True,
                    "ipa_path": str(ipa_files[0]),
                    "archive_path": str(archive_path),
                    "build_time": round(build_time, 2),
                }
            elif archive_path.exists():
                # Archive succeeded but export may have failed (e.g., no signing)
                return {
                    "success": True,
                    "ipa_path": None,
                    "archive_path": str(archive_path),
                    "build_time": round(build_time, 2),
                    "note": "Archive created but IPA export requires code signing",
                }
            else:
                return {
                    "success": False,
                    "error": f"Export failed: {stderr.decode()[-500:]}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clean(self, project_dir: str) -> Dict:
        """Clean the build directory."""
        project_path = Path(project_dir)
        build_dir = project_path / "build"

        try:
            if build_dir.exists():
                import shutil
                shutil.rmtree(build_dir)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_schemes(self, project_dir: str) -> Dict:
        """List available schemes in the project."""
        project_path = Path(project_dir)

        xcworkspace = list(project_path.glob("*.xcworkspace"))
        xcodeproj = list(project_path.glob("*.xcodeproj"))

        if not xcworkspace and not xcodeproj:
            return {"success": False, "error": "No Xcode project found"}

        try:
            if xcworkspace:
                cmd = ["xcodebuild", "-workspace", str(xcworkspace[0]), "-list"]
            else:
                cmd = ["xcodebuild", "-project", str(xcodeproj[0]), "-list"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode()
                schemes = []
                in_schemes_section = False
                for line in output.split("\n"):
                    if "Schemes:" in line:
                        in_schemes_section = True
                        continue
                    if in_schemes_section:
                        scheme = line.strip()
                        if scheme and not scheme.startswith("Build Configurations"):
                            schemes.append(scheme)
                        if not scheme:
                            break

                return {"success": True, "schemes": schemes}
            else:
                return {"success": False, "error": stderr.decode()}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_tests(
        self,
        project_dir: str,
        scheme: Optional[str] = None,
        destination: str = "platform=iOS Simulator,name=iPhone 15",
    ) -> Dict:
        """Run unit tests for the project."""
        project_path = Path(project_dir)

        xcworkspace = list(project_path.glob("*.xcworkspace"))
        xcodeproj = list(project_path.glob("*.xcodeproj"))

        if not xcworkspace and not xcodeproj:
            return {"success": False, "error": "No Xcode project found"}

        if not scheme:
            if xcworkspace:
                scheme = xcworkspace[0].stem
            elif xcodeproj:
                scheme = xcodeproj[0].stem

        try:
            if xcworkspace:
                cmd = [
                    "xcodebuild",
                    "-workspace", str(xcworkspace[0]),
                    "-scheme", scheme,
                    "-destination", destination,
                    "test",
                ]
            else:
                cmd = [
                    "xcodebuild",
                    "-project", str(xcodeproj[0]),
                    "-scheme", scheme,
                    "-destination", destination,
                    "test",
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {"success": True, "output": stdout.decode()[-2000:]}
            else:
                return {"success": False, "error": stderr.decode()[-1000:]}

        except Exception as e:
            return {"success": False, "error": str(e)}
