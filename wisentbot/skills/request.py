#!/usr/bin/env python3
"""
Request Skill - Agent asks humans to implement things

The agent can request features, fixes, or help from human operators.
Requests are stored and viewable via web UI.
"""

import json
import os
import uuid
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from .base import Skill, SkillResult, SkillManifest, SkillAction


REQUESTS_FILE = Path(__file__).parent.parent / "data" / "requests.json"
LINEAR_API_URL = os.environ.get("LINEAR_API_URL", "http://localhost:3000/api/requests/linear")


class RequestSkill(Skill):
    """
    Request Skill - Agent asks for human help.

    The agent can:
    - Submit requests for features/fixes/help
    - Check status of previous requests
    - List all requests
    - Cancel requests
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not REQUESTS_FILE.exists():
            self._save_requests([])

    def _load_requests(self) -> List[Dict]:
        """Load requests from file"""
        try:
            with open(REQUESTS_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_requests(self, requests: List[Dict]):
        """Save requests to file"""
        with open(REQUESTS_FILE, 'w') as f:
            json.dump(requests, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="request",
            name="Human Request",
            version="1.0.0",
            category="communication",
            description="Request features, fixes, or help from human operators",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="submit",
                    description="Submit a new request to humans",
                    parameters={
                        "type": "Type: feature, fix, skill, resource, or other",
                        "title": "Short title for the request",
                        "description": "Detailed description of what you need",
                        "priority": "Priority: low, medium, high, critical",
                        "reason": "Why you need this"
                    },
                    estimated_cost=0,
                    success_probability=1.0
                ),
                SkillAction(
                    name="list",
                    description="List all requests",
                    parameters={
                        "status": "Filter by status: pending, approved, rejected, completed, all"
                    },
                    estimated_cost=0,
                    success_probability=1.0
                ),
                SkillAction(
                    name="get",
                    description="Get details of a specific request",
                    parameters={"request_id": "Request ID"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="cancel",
                    description="Cancel a pending request",
                    parameters={"request_id": "Request ID to cancel"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="add_comment",
                    description="Add a comment to a request",
                    parameters={
                        "request_id": "Request ID",
                        "comment": "Comment text"
                    },
                    estimated_cost=0,
                    success_probability=0.95
                ),
            ]
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            if action == "submit":
                return await self._submit(
                    params.get("type", "other"),
                    params.get("title", ""),
                    params.get("description", ""),
                    params.get("priority", "medium"),
                    params.get("reason", "")
                )
            elif action == "list":
                return await self._list(params.get("status", "all"))
            elif action == "get":
                return await self._get(params.get("request_id", ""))
            elif action == "cancel":
                return await self._cancel(params.get("request_id", ""))
            elif action == "add_comment":
                return await self._add_comment(
                    params.get("request_id", ""),
                    params.get("comment", "")
                )
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    async def _submit(
        self,
        req_type: str,
        title: str,
        description: str,
        priority: str,
        reason: str
    ) -> SkillResult:
        """Submit a new request - auto-approved and sent to Linear"""
        if not title:
            return SkillResult(success=False, message="Title is required")

        if not description:
            return SkillResult(success=False, message="Description is required")

        request_id = str(uuid.uuid4())[:8]
        agent_name = os.environ.get("AGENT_NAME", "Unknown")

        request = {
            "id": request_id,
            "type": req_type,
            "title": title,
            "description": description,
            "priority": priority,
            "reason": reason,
            "status": "approved",  # Auto-approve
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "comments": [],
            "agent": agent_name
        }

        # Create Linear ticket
        linear_result = await self._create_linear_ticket(request)
        if linear_result:
            request["linear_issue"] = linear_result.get("issue")
            request["linear_assignee"] = linear_result.get("assignee")
            request["comments"].append({
                "text": f"Linear ticket created: {linear_result.get('issue', {}).get('identifier', 'N/A')} - Assigned to {linear_result.get('assignee', 'N/A')}",
                "author": "system",
                "created_at": datetime.now().isoformat()
            })

        requests = self._load_requests()
        requests.append(request)
        self._save_requests(requests)

        return SkillResult(
            success=True,
            message=f"Request auto-approved and sent to Linear: {request_id}",
            data={
                "request_id": request_id,
                "title": title,
                "status": "approved",
                "linear": linear_result
            }
        )

    async def _create_linear_ticket(self, request: Dict) -> Optional[Dict]:
        """Create a Linear ticket for the request"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LINEAR_API_URL,
                    json={"request": request},
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        print(f"Linear API error: {resp.status}")
                        return None
        except Exception as e:
            print(f"Failed to create Linear ticket: {e}")
            return None

    async def _list(self, status: str = "all") -> SkillResult:
        """List requests"""
        requests = self._load_requests()

        if status != "all":
            requests = [r for r in requests if r.get("status") == status]

        # Sort by priority and date
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        requests.sort(key=lambda r: (
            priority_order.get(r.get("priority", "medium"), 2),
            r.get("created_at", "")
        ))

        summary = [
            {
                "id": r["id"],
                "type": r.get("type"),
                "title": r.get("title"),
                "priority": r.get("priority"),
                "status": r.get("status"),
                "created_at": r.get("created_at")
            }
            for r in requests
        ]

        return SkillResult(
            success=True,
            message=f"Found {len(summary)} requests",
            data={"requests": summary}
        )

    async def _get(self, request_id: str) -> SkillResult:
        """Get a specific request"""
        if not request_id:
            return SkillResult(success=False, message="Request ID required")

        requests = self._load_requests()
        request = next((r for r in requests if r["id"] == request_id), None)

        if not request:
            return SkillResult(success=False, message=f"Request not found: {request_id}")

        return SkillResult(
            success=True,
            message=f"Request: {request.get('title')}",
            data={"request": request}
        )

    async def _cancel(self, request_id: str) -> SkillResult:
        """Cancel a pending request"""
        if not request_id:
            return SkillResult(success=False, message="Request ID required")

        requests = self._load_requests()
        request = next((r for r in requests if r["id"] == request_id), None)

        if not request:
            return SkillResult(success=False, message=f"Request not found: {request_id}")

        if request.get("status") != "pending":
            return SkillResult(
                success=False,
                message=f"Cannot cancel request with status: {request.get('status')}"
            )

        request["status"] = "cancelled"
        request["updated_at"] = datetime.now().isoformat()
        self._save_requests(requests)

        return SkillResult(
            success=True,
            message=f"Request cancelled: {request_id}"
        )

    async def _add_comment(self, request_id: str, comment: str) -> SkillResult:
        """Add a comment to a request"""
        if not request_id:
            return SkillResult(success=False, message="Request ID required")

        if not comment:
            return SkillResult(success=False, message="Comment required")

        requests = self._load_requests()
        request = next((r for r in requests if r["id"] == request_id), None)

        if not request:
            return SkillResult(success=False, message=f"Request not found: {request_id}")

        if "comments" not in request:
            request["comments"] = []

        request["comments"].append({
            "text": comment,
            "author": "agent",
            "created_at": datetime.now().isoformat()
        })
        request["updated_at"] = datetime.now().isoformat()
        self._save_requests(requests)

        return SkillResult(
            success=True,
            message="Comment added"
        )


# Human-side functions for the web UI

def get_all_requests() -> List[Dict]:
    """Get all requests for the web UI"""
    try:
        with open(REQUESTS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def update_request_status(request_id: str, status: str, response: str = None) -> bool:
    """Update request status (for humans)"""
    try:
        with open(REQUESTS_FILE, 'r') as f:
            requests = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False

    request = next((r for r in requests if r["id"] == request_id), None)
    if not request:
        return False

    request["status"] = status
    request["updated_at"] = datetime.now().isoformat()

    if response:
        if "comments" not in request:
            request["comments"] = []
        request["comments"].append({
            "text": response,
            "author": "human",
            "created_at": datetime.now().isoformat()
        })

    with open(REQUESTS_FILE, 'w') as f:
        json.dump(requests, f, indent=2, default=str)

    return True
