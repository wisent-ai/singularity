"""Request Skill - Agent asks humans to implement things."""

import json
import os
from typing import Dict, List
from pathlib import Path
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction

REQUESTS_FILE = Path(__file__).parent.parent.parent / "data" / "requests.json"
LINEAR_API_URL = os.environ.get("LINEAR_API_URL", "http://localhost:3000/api/requests/linear")


class RequestSkill(Skill):
    """Request Skill - Agent asks for human help."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not REQUESTS_FILE.exists():
            self._save_requests([])

    def _load_requests(self) -> List[Dict]:
        try:
            with open(REQUESTS_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_requests(self, requests: List[Dict]):
        with open(REQUESTS_FILE, 'w') as f:
            json.dump(requests, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="request", name="Human Request", version="1.0.0",
            category="communication",
            description="Request features, fixes, or help from human operators",
            required_credentials=[], install_cost=0,
            actions=[
                SkillAction(name="submit", description="Submit a new request to humans",
                    parameters={"type": "Type: feature, fix, skill, resource, or other",
                                "title": "Short title", "description": "Detailed description",
                                "priority": "Priority: low, medium, high, critical",
                                "reason": "Why you need this"},
                    estimated_cost=0, success_probability=1.0),
                SkillAction(name="list", description="List all requests",
                    parameters={"status": "Filter by status: pending, approved, rejected, completed, all"},
                    estimated_cost=0, success_probability=1.0),
                SkillAction(name="get", description="Get details of a specific request",
                    parameters={"request_id": "Request ID"}, estimated_cost=0, success_probability=0.95),
                SkillAction(name="cancel", description="Cancel a pending request",
                    parameters={"request_id": "Request ID to cancel"}, estimated_cost=0, success_probability=0.9),
                SkillAction(name="add_comment", description="Add a comment to a request",
                    parameters={"request_id": "Request ID", "comment": "Comment text"},
                    estimated_cost=0, success_probability=0.95),
            ])

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            dispatch = {
                "submit": lambda: handlers.submit(self, params.get("type", "other"),
                    params.get("title", ""), params.get("description", ""),
                    params.get("priority", "medium"), params.get("reason", "")),
                "list": lambda: handlers.list_requests(self, params.get("status", "all")),
                "get": lambda: handlers.get_request(self, params.get("request_id", "")),
                "cancel": lambda: handlers.cancel(self, params.get("request_id", "")),
                "add_comment": lambda: handlers.add_comment(self, params.get("request_id", ""),
                    params.get("comment", "")),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=str(e))


# Deferred import: handlers depends on LINEAR_API_URL defined above
from . import handlers  # noqa: E402


# Human-side functions for the web UI
def get_all_requests() -> List[Dict]:
    try:
        with open(REQUESTS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def update_request_status(request_id: str, status: str, response: str = None) -> bool:
    from datetime import datetime
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
        request["comments"].append({"text": response, "author": "human",
                                    "created_at": datetime.now().isoformat()})
    with open(REQUESTS_FILE, 'w') as f:
        json.dump(requests, f, indent=2, default=str)
    return True
