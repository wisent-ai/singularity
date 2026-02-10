#!/usr/bin/env python3
"""
App Store Connect Helpers - Submission, metadata, and screenshot operations.

Helper functions extracted from AppStoreClient for listing updates,
screenshot uploads, and app listing management.
"""

from typing import Dict, List, Optional

import httpx


async def get_submission_status(api_base: str, get_headers, app_id: str) -> Dict:
    """Get the current status of an app in App Store Connect."""
    try:
        async with httpx.AsyncClient() as client:
            # Get app by bundle ID
            response = await client.get(
                f"{api_base}/apps",
                headers=get_headers(),
                params={"filter[bundleId]": app_id},
            )

            if response.status_code != 200:
                return {"success": False, "error": f"API error: {response.text}"}

            data = response.json()
            apps = data.get("data", [])

            if not apps:
                return {"success": False, "error": "App not found"}

            app = apps[0]
            app_store_id = app["id"]

            # Get app store version
            version_response = await client.get(
                f"{api_base}/apps/{app_store_id}/appStoreVersions",
                headers=get_headers(),
                params={"limit": 1},
            )

            if version_response.status_code == 200:
                version_data = version_response.json()
                versions = version_data.get("data", [])

                if versions:
                    version = versions[0]
                    return {
                        "success": True,
                        "app_id": app_id,
                        "app_store_id": app_store_id,
                        "version": version["attributes"].get("versionString"),
                        "status": version["attributes"].get("appStoreState"),
                    }

            return {
                "success": True,
                "app_id": app_id,
                "app_store_id": app_store_id,
                "status": "no_version",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def update_listing(
    api_base: str,
    get_headers,
    app_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    release_notes: Optional[str] = None,
) -> Dict:
    """Update app store listing metadata."""
    try:
        async with httpx.AsyncClient() as client:
            # Get app by bundle ID
            response = await client.get(
                f"{api_base}/apps",
                headers=get_headers(),
                params={"filter[bundleId]": app_id},
            )

            if response.status_code != 200:
                return {"success": False, "error": f"API error: {response.text}"}

            data = response.json()
            apps = data.get("data", [])

            if not apps:
                return {"success": False, "error": "App not found"}

            app_store_id = apps[0]["id"]

            # Get app info localizations
            info_response = await client.get(
                f"{api_base}/apps/{app_store_id}/appInfos",
                headers=get_headers(),
            )

            if info_response.status_code != 200:
                return {"success": False, "error": "Failed to get app info"}

            # Get the current version
            version_response = await client.get(
                f"{api_base}/apps/{app_store_id}/appStoreVersions",
                headers=get_headers(),
                params={"filter[appStoreState]": "PREPARE_FOR_SUBMISSION,DEVELOPER_REJECTED"},
            )

            if version_response.status_code == 200:
                versions = version_response.json().get("data", [])
                if versions:
                    version_id = versions[0]["id"]

                    # Get localization
                    loc_response = await client.get(
                        f"{api_base}/appStoreVersions/{version_id}/appStoreVersionLocalizations",
                        headers=get_headers(),
                    )

                    if loc_response.status_code == 200:
                        localizations = loc_response.json().get("data", [])
                        if localizations:
                            loc_id = localizations[0]["id"]

                            # Update localization
                            update_data = {
                                "data": {
                                    "type": "appStoreVersionLocalizations",
                                    "id": loc_id,
                                    "attributes": {},
                                }
                            }

                            if description:
                                update_data["data"]["attributes"]["description"] = description
                            if release_notes:
                                update_data["data"]["attributes"]["whatsNew"] = release_notes
                            if keywords:
                                update_data["data"]["attributes"]["keywords"] = ", ".join(keywords)

                            patch_response = await client.patch(
                                f"{api_base}/appStoreVersionLocalizations/{loc_id}",
                                headers=get_headers(),
                                json=update_data,
                            )

                            if patch_response.status_code in [200, 204]:
                                return {
                                    "success": True,
                                    "app_id": app_id,
                                    "updated": list(update_data["data"]["attributes"].keys()),
                                }

            return {"success": False, "error": "No editable version found"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def upload_screenshots(
    api_base: str,
    get_headers,
    app_id: str,
    screenshot_urls: List[str],
) -> Dict:
    """Upload screenshots to App Store Connect."""
    try:
        async with httpx.AsyncClient() as client:
            # Get app
            response = await client.get(
                f"{api_base}/apps",
                headers=get_headers(),
                params={"filter[bundleId]": app_id},
            )

            if response.status_code != 200:
                return {"success": False, "error": "App not found"}

            apps = response.json().get("data", [])
            if not apps:
                return {"success": False, "error": "App not found"}

            app_store_id = apps[0]["id"]

            # Get editable version
            version_response = await client.get(
                f"{api_base}/apps/{app_store_id}/appStoreVersions",
                headers=get_headers(),
            )

            if version_response.status_code != 200:
                return {"success": False, "error": "Failed to get versions"}

            versions = version_response.json().get("data", [])
            if not versions:
                return {"success": False, "error": "No version found"}

            # Screenshots require complex upload process via App Store Connect
            # For now, return guidance to use Fastlane deliver
            return {
                "success": False,
                "error": "Screenshot upload requires Fastlane. Use 'fastlane deliver' with screenshots in fastlane/screenshots/",
                "hint": "Place screenshots in fastlane/screenshots/en-US/ directory",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_apps(api_base: str, get_headers) -> Dict:
    """List all apps in App Store Connect."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_base}/apps",
                headers=get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                apps = [
                    {
                        "id": app["id"],
                        "bundle_id": app["attributes"]["bundleId"],
                        "name": app["attributes"]["name"],
                    }
                    for app in data.get("data", [])
                ]
                return {"success": True, "apps": apps}
            else:
                return {"success": False, "error": f"API error: {response.text}"}

    except Exception as e:
        return {"success": False, "error": str(e)}
