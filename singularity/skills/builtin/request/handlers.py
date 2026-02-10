"""Handler functions for RequestSkill actions."""

import os
import uuid
import aiohttp
from datetime import datetime
from typing import Dict, Optional
from singularity.skills.base import SkillResult
from . import LINEAR_API_URL


async def _create_linear_ticket(request: Dict) -> Optional[Dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LINEAR_API_URL, json={"request": request},
                    headers={"Content-Type": "application/json"}) as resp:
                if resp.status == 200:
                    return await resp.json()
                print(f"Linear API error: {resp.status}")
                return None
    except Exception as e:
        print(f"Failed to create Linear ticket: {e}")
        return None


async def submit(skill, req_type: str, title: str, description: str,
                 priority: str, reason: str) -> SkillResult:
    if not title:
        return SkillResult(success=False, message="Title is required")
    if not description:
        return SkillResult(success=False, message="Description is required")
    request_id = str(uuid.uuid4())[:8]
    agent_name = os.environ.get("AGENT_NAME", "Unknown")
    request = {"id": request_id, "type": req_type, "title": title, "description": description,
               "priority": priority, "reason": reason, "status": "approved",
               "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat(),
               "comments": [], "agent": agent_name}
    linear_result = await _create_linear_ticket(request)
    if linear_result:
        request["linear_issue"] = linear_result.get("issue")
        request["linear_assignee"] = linear_result.get("assignee")
        request["comments"].append({
            "text": f"Linear ticket created: {linear_result.get('issue', {}).get('identifier', 'N/A')} - Assigned to {linear_result.get('assignee', 'N/A')}",
            "author": "system", "created_at": datetime.now().isoformat()})
    requests = skill._load_requests()
    requests.append(request)
    skill._save_requests(requests)
    return SkillResult(success=True, message=f"Request auto-approved and sent to Linear: {request_id}",
        data={"request_id": request_id, "title": title, "status": "approved", "linear": linear_result})


async def list_requests(skill, status: str = "all") -> SkillResult:
    requests = skill._load_requests()
    if status != "all":
        requests = [r for r in requests if r.get("status") == status]
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    requests.sort(key=lambda r: (priority_order.get(r.get("priority", "medium"), 2), r.get("created_at", "")))
    summary = [{"id": r["id"], "type": r.get("type"), "title": r.get("title"),
                "priority": r.get("priority"), "status": r.get("status"),
                "created_at": r.get("created_at")} for r in requests]
    return SkillResult(success=True, message=f"Found {len(summary)} requests", data={"requests": summary})


async def get_request(skill, request_id: str) -> SkillResult:
    if not request_id:
        return SkillResult(success=False, message="Request ID required")
    requests = skill._load_requests()
    request = next((r for r in requests if r["id"] == request_id), None)
    if not request:
        return SkillResult(success=False, message=f"Request not found: {request_id}")
    return SkillResult(success=True, message=f"Request: {request.get('title')}", data={"request": request})


async def cancel(skill, request_id: str) -> SkillResult:
    if not request_id:
        return SkillResult(success=False, message="Request ID required")
    requests = skill._load_requests()
    request = next((r for r in requests if r["id"] == request_id), None)
    if not request:
        return SkillResult(success=False, message=f"Request not found: {request_id}")
    if request.get("status") != "pending":
        return SkillResult(success=False, message=f"Cannot cancel request with status: {request.get('status')}")
    request["status"] = "cancelled"
    request["updated_at"] = datetime.now().isoformat()
    skill._save_requests(requests)
    return SkillResult(success=True, message=f"Request cancelled: {request_id}")


async def add_comment(skill, request_id: str, comment: str) -> SkillResult:
    if not request_id:
        return SkillResult(success=False, message="Request ID required")
    if not comment:
        return SkillResult(success=False, message="Comment required")
    requests = skill._load_requests()
    request = next((r for r in requests if r["id"] == request_id), None)
    if not request:
        return SkillResult(success=False, message=f"Request not found: {request_id}")
    if "comments" not in request:
        request["comments"] = []
    request["comments"].append({"text": comment, "author": "agent",
                                "created_at": datetime.now().isoformat()})
    request["updated_at"] = datetime.now().isoformat()
    skill._save_requests(requests)
    return SkillResult(success=True, message="Comment added")
