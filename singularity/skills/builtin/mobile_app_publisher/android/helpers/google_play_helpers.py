#!/usr/bin/env python3
"""
Google Play Helpers - API interaction methods for Google Play.

Contains methods for status checking, listing updates, screenshots, and track listing.
"""

from typing import Dict, List, Optional


async def get_submission_status(client, package_name: str) -> Dict:
    """Get the current status of an app on Google Play."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        credentials = client._get_credentials()
        service = build("androidpublisher", "v3", credentials=credentials)

        # Get app details
        result = service.edits().insert(
            packageName=package_name,
            body={},
        ).execute()

        edit_id = result["id"]

        # Get tracks
        tracks = service.edits().tracks().list(
            packageName=package_name,
            editId=edit_id,
        ).execute()

        # Find the latest release
        latest_release = None
        latest_track = None

        for track in tracks.get("tracks", []):
            releases = track.get("releases", [])
            if releases:
                latest_release = releases[0]
                latest_track = track["track"]
                break

        # Delete the edit (we were just reading)
        service.edits().delete(
            packageName=package_name,
            editId=edit_id,
        ).execute()

        if latest_release:
            return {
                "success": True,
                "package_name": package_name,
                "track": latest_track,
                "status": latest_release.get("status"),
                "version_codes": latest_release.get("versionCodes", []),
            }
        else:
            return {
                "success": True,
                "package_name": package_name,
                "status": "no_release",
            }

    except ImportError:
        return {
            "success": False,
            "error": "google-api-python-client required: pip install google-api-python-client",
        }
    except Exception as e:
        error_str = str(e)
        if "applicationNotFound" in error_str:
            return {
                "success": False,
                "error": "App not found. Create it first in Google Play Console.",
            }
        return {"success": False, "error": error_str}


async def update_listing(
    client,
    package_name: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    release_notes: Optional[str] = None,
) -> Dict:
    """Update Google Play store listing metadata."""
    try:
        from googleapiclient.discovery import build

        credentials = client._get_credentials()
        service = build("androidpublisher", "v3", credentials=credentials)

        # Create edit
        edit = service.edits().insert(
            packageName=package_name,
            body={},
        ).execute()
        edit_id = edit["id"]

        updated = []

        # Update listing
        if name or description:
            listing_update = {}
            if name:
                listing_update["title"] = name
            if description:
                listing_update["fullDescription"] = description
                listing_update["shortDescription"] = description[:80]

            service.edits().listings().update(
                packageName=package_name,
                editId=edit_id,
                language="en-US",
                body=listing_update,
            ).execute()

            updated.extend(listing_update.keys())

        # Commit the edit
        service.edits().commit(
            packageName=package_name,
            editId=edit_id,
        ).execute()

        return {
            "success": True,
            "package_name": package_name,
            "updated": updated,
        }

    except ImportError:
        return {
            "success": False,
            "error": "google-api-python-client required: pip install google-api-python-client",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def upload_screenshots(
    client,
    package_name: str,
    screenshot_urls: List[str],
) -> Dict:
    """Upload screenshots to Google Play."""
    try:
        from googleapiclient.discovery import build
        import httpx

        credentials = client._get_credentials()
        service = build("androidpublisher", "v3", credentials=credentials)

        # Create edit
        edit = service.edits().insert(
            packageName=package_name,
            body={},
        ).execute()
        edit_id = edit["id"]

        uploaded = []

        async with httpx.AsyncClient() as http:
            for i, url in enumerate(screenshot_urls):
                # Download screenshot
                response = await http.get(url)
                if response.status_code != 200:
                    continue

                # Upload to Google Play
                # Note: This requires media upload which is complex
                # For now, use Fastlane for screenshots
                uploaded.append(url)

        # For complex uploads, recommend Fastlane
        return {
            "success": False,
            "error": "Screenshot upload requires Fastlane. Use 'fastlane supply' with images in fastlane/metadata/android/en-US/images/",
            "hint": "Place phone screenshots in phoneScreenshots/, tablet in sevenInchScreenshots/",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_tracks(client, package_name: str) -> Dict:
    """List all tracks and their releases."""
    try:
        from googleapiclient.discovery import build

        credentials = client._get_credentials()
        service = build("androidpublisher", "v3", credentials=credentials)

        # Create edit
        edit = service.edits().insert(
            packageName=package_name,
            body={},
        ).execute()
        edit_id = edit["id"]

        # Get tracks
        tracks = service.edits().tracks().list(
            packageName=package_name,
            editId=edit_id,
        ).execute()

        # Delete the edit
        service.edits().delete(
            packageName=package_name,
            editId=edit_id,
        ).execute()

        return {
            "success": True,
            "tracks": tracks.get("tracks", []),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
