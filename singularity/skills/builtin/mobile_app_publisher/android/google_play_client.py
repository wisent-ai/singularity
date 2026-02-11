#!/usr/bin/env python3
"""
Google Play Client - Submit and manage Android apps on Google Play.

Uses Google Play Developer API with service account authentication and Fastlane for uploads.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .helpers.google_play_helpers import (
        get_submission_status,
        update_listing,
        upload_screenshots,
        list_tracks,
    )
except ImportError:
    # Stubs for when the helpers module is not available
    async def get_submission_status(*a, **kw): return {"error": "helpers not available"}  # type: ignore[misc]
    async def update_listing(*a, **kw): return {"error": "helpers not available"}  # type: ignore[misc]
    async def upload_screenshots(*a, **kw): return {"error": "helpers not available"}  # type: ignore[misc]
    async def list_tracks(*a, **kw): return {"error": "helpers not available"}  # type: ignore[misc]


class GooglePlayClient:
    """
    Client for Google Play Developer API.

    Handles app submission, metadata updates, and status monitoring.
    Uses service account authentication and Fastlane for binary uploads.
    """

    def __init__(self, service_account_json: Optional[str] = None):
        """
        Initialize Google Play client.

        Args:
            service_account_json: Service account JSON as string or path to file
        """
        self.service_account_json = service_account_json
        self._credentials = None

    def _get_credentials(self):
        """Get Google API credentials from service account."""
        if self._credentials:
            return self._credentials

        if not self.service_account_json:
            raise ValueError("Service account JSON not provided")

        try:
            from google.oauth2 import service_account

            # Check if it's a path or JSON string
            if self.service_account_json.strip().startswith("{"):
                info = json.loads(self.service_account_json)
            else:
                with open(self.service_account_json) as f:
                    info = json.load(f)

            self._credentials = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/androidpublisher"],
            )
            return self._credentials

        except ImportError:
            raise ImportError("google-auth library required: pip install google-auth")

    async def submit(
        self,
        project_dir: str,
        app_name: str,
        description: str,
        track: str = "internal",
    ) -> Dict:
        """
        Submit an Android app to Google Play using Fastlane.

        Args:
            project_dir: Path to the Android project
            app_name: App name for the store
            description: App description
            track: Release track (internal, alpha, beta, production)

        Returns:
            Dict with success status and submission details
        """
        project_path = Path(project_dir)

        # Find AAB or APK
        build_dir = project_path / "app" / "build" / "outputs"
        aab_files = list(build_dir.glob("**/*.aab")) if build_dir.exists() else []
        apk_files = list(build_dir.glob("**/*.apk")) if build_dir.exists() else []

        if not aab_files and not apk_files:
            return {"success": False, "error": "No AAB or APK found. Build first."}

        artifact = aab_files[0] if aab_files else apk_files[0]

        # Get package name from build.gradle.kts
        package_name = None
        gradle_files = list(project_path.glob("**/build.gradle.kts"))
        for gf in gradle_files:
            content = gf.read_text()
            import re
            match = re.search(r'applicationId\s*=\s*"([\w.]+)"', content)
            if match:
                package_name = match.group(1)
                break

        if not package_name:
            package_name = "com.example.app"

        # Create Fastlane configuration
        fastlane_dir = project_path / "fastlane"
        fastlane_dir.mkdir(exist_ok=True)

        # Write service account JSON to temp file
        service_account_path = fastlane_dir / "play-store-credentials.json"
        if self.service_account_json:
            if self.service_account_json.strip().startswith("{"):
                service_account_path.write_text(self.service_account_json)
            else:
                # It's a path, copy content
                import shutil
                shutil.copy(self.service_account_json, service_account_path)

        # Create Appfile
        appfile = fastlane_dir / "Appfile"
        appfile.write_text(f'''json_key_file("{service_account_path}")
package_name("{package_name}")
''')

        # Create Fastfile for upload
        fastfile = fastlane_dir / "Fastfile"
        fastfile.write_text(f'''default_platform(:android)

platform :android do
  desc "Upload to Google Play"
  lane :upload do
    upload_to_play_store(
      track: "{track}",
      {"aab: '" + str(artifact) + "'," if str(artifact).endswith('.aab') else "apk: '" + str(artifact) + "',"}
      skip_upload_metadata: false,
      skip_upload_images: true,
      skip_upload_screenshots: true
    )
  end
end
''')

        # Create metadata
        metadata_dir = fastlane_dir / "metadata" / "android" / "en-US"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        (metadata_dir / "title.txt").write_text(app_name)
        (metadata_dir / "full_description.txt").write_text(description)
        (metadata_dir / "short_description.txt").write_text(description[:80] if len(description) > 80 else description)

        try:
            # Run fastlane upload
            process = await asyncio.create_subprocess_exec(
                "fastlane", "upload",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_path),
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {
                    "success": True,
                    "package_name": package_name,
                    "version_code": "1",
                    "track": track,
                    "status": "uploaded",
                }
            else:
                error_msg = stderr.decode()

                # Check for common errors
                if "applicationNotFound" in error_msg:
                    return {
                        "success": False,
                        "error": "App not found in Google Play Console. Create it first at https://play.google.com/console",
                    }
                elif "apkUpgradeVersionConflict" in error_msg:
                    return {
                        "success": False,
                        "error": "Version code conflict. Increment versionCode in build.gradle.kts",
                    }

                return {
                    "success": False,
                    "error": f"Fastlane failed: {error_msg[-500:]}",
                }

        except FileNotFoundError:
            return {
                "success": False,
                "error": "Fastlane not installed. Install with: gem install fastlane",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_submission_status(self, package_name: str) -> Dict:
        """Get the current status of an app on Google Play."""
        return await get_submission_status(self, package_name)

    async def update_listing(
        self,
        package_name: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        release_notes: Optional[str] = None,
    ) -> Dict:
        """Update Google Play store listing metadata."""
        return await update_listing(self, package_name, name, description, release_notes)

    async def upload_screenshots(
        self,
        package_name: str,
        screenshot_urls: List[str],
    ) -> Dict:
        """Upload screenshots to Google Play."""
        return await upload_screenshots(self, package_name, screenshot_urls)

    async def list_tracks(self, package_name: str) -> Dict:
        """List all tracks and their releases."""
        return await list_tracks(self, package_name)
