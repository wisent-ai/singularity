#!/usr/bin/env python3
"""
App Store Connect Client - Submit and manage iOS apps on the App Store.

Uses App Store Connect API with JWT authentication and Fastlane for submissions.
Heavy API methods are delegated to app_store_helpers.py.
"""

import asyncio
import jwt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .app_store_helpers import (
    get_submission_status as _get_submission_status,
    update_listing as _update_listing,
    upload_screenshots as _upload_screenshots,
    list_apps as _list_apps,
)


class AppStoreClient:
    """
    Client for App Store Connect API.

    Handles app submission, metadata updates, and status monitoring.
    Uses JWT for API authentication and Fastlane for binary uploads.
    """

    API_BASE = "https://api.appstoreconnect.apple.com/v1"

    def __init__(
        self,
        key_id: Optional[str] = None,
        issuer_id: Optional[str] = None,
        private_key: Optional[str] = None,
        team_id: Optional[str] = None,
    ):
        self.key_id = key_id
        self.issuer_id = issuer_id
        self.private_key = private_key
        self.team_id = team_id
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    def _generate_token(self) -> str:
        """Generate JWT for App Store Connect API."""
        if not all([self.key_id, self.issuer_id, self.private_key]):
            raise ValueError("Missing App Store Connect credentials")

        now = datetime.utcnow()
        expiry = now + timedelta(minutes=20)

        payload = {
            "iss": self.issuer_id,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "aud": "appstoreconnect-v1",
        }

        key = self.private_key
        if "-----BEGIN" not in key:
            import base64
            try:
                key = base64.b64decode(key).decode()
            except Exception:
                pass

        token = jwt.encode(
            payload, key, algorithm="ES256",
            headers={"kid": self.key_id},
        )
        self._token = token
        self._token_expiry = expiry
        return token

    def _get_token(self) -> str:
        """Get a valid JWT token, generating if needed."""
        if self._token and self._token_expiry and datetime.utcnow() < self._token_expiry:
            return self._token
        return self._generate_token()

    def _get_headers(self) -> Dict:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

    async def submit(
        self,
        project_dir: str,
        app_name: str,
        description: str,
        keywords: Optional[List[str]] = None,
        category: Optional[str] = None,
        submit_for_review: bool = False,
    ) -> Dict:
        """Submit an iOS app to App Store Connect using Fastlane."""
        project_path = Path(project_dir)

        build_dir = project_path / "build"
        ipa_files = list(build_dir.glob("**/*.ipa")) if build_dir.exists() else []
        archive_files = list(build_dir.glob("**/*.xcarchive")) if build_dir.exists() else []

        if not ipa_files and not archive_files:
            return {"success": False, "error": "No IPA or archive found. Build first."}

        fastlane_dir = project_path / "fastlane"
        fastlane_dir.mkdir(exist_ok=True)

        # Get bundle ID from project
        info_plist = list(project_path.glob("*/Info.plist"))
        bundle_id = "com.example.app"
        if info_plist:
            try:
                import plistlib
                with open(info_plist[0], "rb") as f:
                    plist = plistlib.load(f)
                    bundle_id = plist.get("CFBundleIdentifier", bundle_id)
            except Exception:
                pass

        # Create Appfile
        (fastlane_dir / "Appfile").write_text(
            f'app_identifier("{bundle_id}")\n'
            f'apple_id("{self.issuer_id}")\n'
            f'team_id("{self.team_id}")\n'
        )

        # Create Fastfile
        ipa_line = f"ipa: '{ipa_files[0]}'," if ipa_files else ""
        review_flag = "true" if submit_for_review else "false"
        (fastlane_dir / "Fastfile").write_text(
            f'default_platform(:ios)\n\nplatform :ios do\n'
            f'  desc "Upload to App Store Connect"\n  lane :upload do\n'
            f'    api_key = app_store_connect_api_key(\n'
            f'      key_id: "{self.key_id}",\n'
            f'      issuer_id: "{self.issuer_id}",\n'
            f'      key_content: ENV["APP_STORE_CONNECT_PRIVATE_KEY"],\n'
            f'      is_key_content_base64: false\n    )\n\n'
            f'    upload_to_app_store(\n      api_key: api_key,\n'
            f'      skip_metadata: false,\n      skip_screenshots: true,\n'
            f'      submit_for_review: {review_flag},\n'
            f'      automatic_release: false,\n      app_version: "1.0",\n'
            f'      {ipa_line}\n    )\n  end\nend\n'
        )

        # Create metadata
        metadata_dir = fastlane_dir / "metadata" / "en-US"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "name.txt").write_text(app_name)
        (metadata_dir / "description.txt").write_text(description)
        if keywords:
            (metadata_dir / "keywords.txt").write_text(", ".join(keywords))

        try:
            process = await asyncio.create_subprocess_exec(
                "fastlane", "upload",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_path),
                env={
                    **dict(__import__("os").environ),
                    "APP_STORE_CONNECT_PRIVATE_KEY": self.private_key or "",
                },
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {
                    "success": True, "app_id": bundle_id,
                    "version": "1.0", "status": "uploaded",
                    "submit_for_review": submit_for_review,
                }
            return {"success": False, "error": f"Fastlane failed: {stderr.decode()[-500:]}"}

        except FileNotFoundError:
            return {"success": False, "error": "Fastlane not installed. Install with: gem install fastlane"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_submission_status(self, app_id: str) -> Dict:
        """Get the current status of an app in App Store Connect."""
        return await _get_submission_status(self.API_BASE, self._get_headers, app_id)

    async def update_listing(self, app_id: str, name=None, description=None, keywords=None, release_notes=None) -> Dict:
        """Update app store listing metadata."""
        return await _update_listing(self.API_BASE, self._get_headers, app_id, name, description, keywords, release_notes)

    async def upload_screenshots(self, app_id: str, screenshot_urls: List[str]) -> Dict:
        """Upload screenshots to App Store Connect."""
        return await _upload_screenshots(self.API_BASE, self._get_headers, app_id, screenshot_urls)

    async def list_apps(self) -> Dict:
        """List all apps in App Store Connect."""
        return await _list_apps(self.API_BASE, self._get_headers)
