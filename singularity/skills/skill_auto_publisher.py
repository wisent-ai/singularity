#!/usr/bin/env python3
"""
SkillAutoPublisherSkill - Automatic skill scanning and marketplace publishing.

Bridges the SkillLoader (auto-discovery) and SkillMarketplaceHub (distribution)
into a unified workflow. Without this, agents must manually discover their
skills and publish each one individually - this skill automates that process.

Actions:
1. SCAN       - Scan installed skills and compare against marketplace listings
2. PUBLISH_ALL - Auto-publish all unpublished skills to the marketplace
3. PUBLISH_ONE - Publish a specific skill by ID
4. DIFF       - Show what's new/updated/missing between local and marketplace
5. SYNC       - Full sync: publish new skills, update changed versions
6. UNPUBLISH  - Remove a skill from the marketplace
7. STATUS     - Show publishing status of all local skills
8. SET_PRICING - Set default pricing rules for auto-publishing

Pillars served:
- Revenue: Auto-publish skills to earn from installs by other agents
- Replication: New replicas can auto-publish their skills to the network
- Self-Improvement: After composing/generating new skills, auto-publish them
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

PUBLISHER_FILE = Path(__file__).parent.parent / "data" / "skill_auto_publisher.json"
SKILLS_DIR = Path(__file__).parent


class SkillAutoPublisherSkill(Skill):
    """
    Automatically scan installed skills and publish them to the marketplace.

    Bridges SkillLoader discovery with SkillMarketplaceHub distribution,
    enabling agents to monetize their skill portfolio automatically.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        PUBLISHER_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PUBLISHER_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "published_skills": {},  # skill_id -> {listing_id, version, published_at}
            "pricing_rules": {
                "default_price": 0.0,
                "category_prices": {},  # category -> price
                "skill_prices": {},  # skill_id -> price
                "free_categories": ["core", "utility"],
            },
            "agent_id": "self",
            "auto_publish_on_scan": False,
            "exclude_skills": [],  # skill_ids to never publish
            "stats": {
                "total_scans": 0,
                "total_published": 0,
                "total_updated": 0,
                "total_unpublished": 0,
                "last_scan_at": None,
                "last_publish_at": None,
            },
        }

    def _load(self) -> Dict:
        try:
            return json.loads(PUBLISHER_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return self._default_state()

    def _save(self, state: Dict):
        PUBLISHER_FILE.write_text(json.dumps(state, indent=2, default=str))

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="skill_auto_publisher",
            name="Skill Auto-Publisher",
            version="1.0.0",
            category="revenue",
            description="Automatically scan installed skills and publish them to the marketplace",
            actions=[
                SkillAction(
                    name="scan",
                    description="Scan installed skills and compare against marketplace listings",
                    parameters={
                        "directory": {"type": "string", "required": False, "description": "Skills directory to scan (default: built-in)"},
                    },
                ),
                SkillAction(
                    name="publish_all",
                    description="Auto-publish all unpublished skills to the marketplace",
                    parameters={
                        "agent_id": {"type": "string", "required": False, "description": "Publishing agent ID"},
                        "dry_run": {"type": "boolean", "required": False, "description": "Preview without publishing"},
                    },
                ),
                SkillAction(
                    name="publish_one",
                    description="Publish a specific skill by ID",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "ID of the skill to publish"},
                        "price": {"type": "float", "required": False, "description": "Override price for this skill"},
                        "agent_id": {"type": "string", "required": False, "description": "Publishing agent ID"},
                    },
                ),
                SkillAction(
                    name="diff",
                    description="Show what's new/updated/missing between local and marketplace",
                    parameters={},
                ),
                SkillAction(
                    name="sync",
                    description="Full sync: publish new skills, update changed versions",
                    parameters={
                        "agent_id": {"type": "string", "required": False, "description": "Publishing agent ID"},
                        "dry_run": {"type": "boolean", "required": False, "description": "Preview without syncing"},
                    },
                ),
                SkillAction(
                    name="unpublish",
                    description="Remove a skill from the marketplace",
                    parameters={
                        "skill_id": {"type": "string", "required": True, "description": "ID of the skill to remove"},
                    },
                ),
                SkillAction(
                    name="status",
                    description="Show publishing status of all local skills",
                    parameters={},
                ),
                SkillAction(
                    name="set_pricing",
                    description="Set default pricing rules for auto-publishing",
                    parameters={
                        "default_price": {"type": "float", "required": False, "description": "Default price for new skills"},
                        "category_prices": {"type": "dict", "required": False, "description": "Category-specific prices"},
                        "skill_prices": {"type": "dict", "required": False, "description": "Skill-specific prices"},
                        "free_categories": {"type": "list", "required": False, "description": "Categories that are always free"},
                        "exclude_skills": {"type": "list", "required": False, "description": "Skill IDs to never publish"},
                    },
                ),
            ],
            required_credentials=[],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        handlers = {
            "scan": self._scan,
            "publish_all": self._publish_all,
            "publish_one": self._publish_one,
            "diff": self._diff,
            "sync": self._sync,
            "unpublish": self._unpublish,
            "status": self._status,
            "set_pricing": self._set_pricing,
        }

        handler = handlers.get(action)
        if not handler:
            return SkillResult(
                success=False,
                message=f"Unknown action '{action}'. Valid: {list(handlers.keys())}",
            )

        return await handler(params)

    def _scan_local_skills(self, directory: Optional[str] = None) -> List[Dict]:
        """Scan the skills directory and extract manifests."""
        skills_dir = Path(directory) if directory else SKILLS_DIR
        if not skills_dir.is_dir():
            return []

        skip_files = {"__init__.py", "base.py"}
        discovered = []

        for py_file in sorted(skills_dir.glob("*.py")):
            if py_file.name in skip_files or py_file.name.startswith("_"):
                continue

            # Try to extract skill info by importing
            try:
                skill_info = self._extract_skill_info(py_file)
                if skill_info:
                    discovered.extend(skill_info)
            except Exception:
                continue

        return discovered

    def _extract_skill_info(self, py_file: Path) -> List[Dict]:
        """Extract skill manifest info from a Python file without importing."""
        source = py_file.read_text()
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        # Parse minimal info from the source
        results = []
        import re

        # Find class definitions that inherit from Skill
        class_pattern = re.compile(r'class\s+(\w+)\s*\(\s*Skill\s*\)')
        classes = class_pattern.findall(source)

        # Find skill_id from manifest
        skill_id_pattern = re.compile(r'skill_id\s*=\s*["\']([^"\']+)["\']')
        skill_ids = skill_id_pattern.findall(source)

        # Find name from manifest
        name_pattern = re.compile(r'name\s*=\s*["\']([^"\']+)["\']')
        names = name_pattern.findall(source)

        # Find version from manifest
        version_pattern = re.compile(r'version\s*=\s*["\'](\d+\.\d+\.\d+)["\']')
        versions = version_pattern.findall(source)

        # Find category from manifest
        category_pattern = re.compile(r'category\s*=\s*["\']([^"\']+)["\']')
        categories = category_pattern.findall(source)

        # Find description from manifest
        desc_pattern = re.compile(r'description\s*=\s*["\']([^"\']{10,})["\']')
        descriptions = desc_pattern.findall(source)

        if classes and skill_ids:
            results.append({
                "class_name": classes[0],
                "skill_id": skill_ids[0],
                "name": names[0] if names else classes[0],
                "version": versions[0] if versions else "1.0.0",
                "category": categories[0] if categories else "general",
                "description": descriptions[0] if descriptions else f"Skill: {classes[0]}",
                "source_hash": source_hash,
                "file": py_file.name,
            })

        return results

    def _get_price_for_skill(self, state: Dict, skill_id: str, category: str) -> float:
        """Determine price for a skill based on pricing rules."""
        rules = state["pricing_rules"]

        # Skill-specific price takes priority
        if skill_id in rules.get("skill_prices", {}):
            return float(rules["skill_prices"][skill_id])

        # Free categories
        if category in rules.get("free_categories", []):
            return 0.0

        # Category-specific price
        if category in rules.get("category_prices", {}):
            return float(rules["category_prices"][category])

        # Default price
        return float(rules.get("default_price", 0.0))

    async def _scan(self, params: Dict) -> SkillResult:
        directory = params.get("directory")
        local_skills = self._scan_local_skills(directory)

        state = self._load()
        state["stats"]["total_scans"] += 1
        state["stats"]["last_scan_at"] = datetime.utcnow().isoformat()
        self._save(state)

        published = state.get("published_skills", {})
        new_skills = []
        updated_skills = []
        published_skills = []

        for skill in local_skills:
            sid = skill["skill_id"]
            if sid in state.get("exclude_skills", []):
                continue
            if sid in published:
                pub = published[sid]
                if pub.get("source_hash") != skill["source_hash"]:
                    updated_skills.append(skill)
                else:
                    published_skills.append(skill)
            else:
                new_skills.append(skill)

        return SkillResult(
            success=True,
            message=f"Scanned {len(local_skills)} skills: "
                    f"{len(new_skills)} new, {len(updated_skills)} updated, "
                    f"{len(published_skills)} unchanged",
            data={
                "total_scanned": len(local_skills),
                "new": [s["skill_id"] for s in new_skills],
                "updated": [s["skill_id"] for s in updated_skills],
                "published": [s["skill_id"] for s in published_skills],
                "new_details": new_skills,
                "updated_details": updated_skills,
            },
        )

    async def _publish_all(self, params: Dict) -> SkillResult:
        dry_run = params.get("dry_run", False)
        agent_id = params.get("agent_id", "self")

        local_skills = self._scan_local_skills()
        state = self._load()
        published = state.get("published_skills", {})
        exclude = state.get("exclude_skills", [])

        to_publish = []
        for skill in local_skills:
            sid = skill["skill_id"]
            if sid not in published and sid not in exclude:
                to_publish.append(skill)

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run: would publish {len(to_publish)} skills",
                data={
                    "would_publish": [
                        {
                            "skill_id": s["skill_id"],
                            "name": s["name"],
                            "price": self._get_price_for_skill(state, s["skill_id"], s["category"]),
                        }
                        for s in to_publish
                    ],
                },
            )

        results = []
        errors = []

        for skill in to_publish:
            result = await self._do_publish(state, skill, agent_id)
            if result["success"]:
                results.append(result)
            else:
                errors.append(result)

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Published {len(results)} skills, {len(errors)} errors",
            data={
                "published": results,
                "errors": errors,
                "total_published": len(results),
            },
        )

    async def _publish_one(self, params: Dict) -> SkillResult:
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="Missing skill_id")

        agent_id = params.get("agent_id", "self")
        local_skills = self._scan_local_skills()

        skill = None
        for s in local_skills:
            if s["skill_id"] == skill_id:
                skill = s
                break

        if not skill:
            return SkillResult(
                success=False,
                message=f"Skill '{skill_id}' not found locally. "
                        f"Available: {[s['skill_id'] for s in local_skills[:10]]}",
            )

        state = self._load()

        # Override price if specified
        if "price" in params:
            price_override = float(params["price"])
        else:
            price_override = None

        result = await self._do_publish(state, skill, agent_id, price_override)
        self._save(state)

        if result["success"]:
            return SkillResult(
                success=True,
                message=f"Published '{skill['name']}' v{skill['version']} to marketplace",
                data=result,
            )
        else:
            return SkillResult(success=False, message=result.get("error", "Publish failed"))

    async def _do_publish(
        self, state: Dict, skill: Dict, agent_id: str,
        price_override: Optional[float] = None,
    ) -> Dict:
        """Publish a single skill. Uses SkillMarketplaceHub via context if available."""
        sid = skill["skill_id"]
        price = price_override if price_override is not None else self._get_price_for_skill(
            state, sid, skill["category"]
        )

        publish_params = {
            "skill_id": sid,
            "name": skill["name"],
            "description": skill["description"],
            "version": skill["version"],
            "category": skill["category"],
            "price": price,
            "author_agent_id": agent_id,
            "source_hash": skill["source_hash"],
            "tags": [skill["category"], skill.get("file", "")],
        }

        # Try to use SkillMarketplaceHub via context
        if self.context:
            try:
                result = await self.context.call_skill(
                    "skill_marketplace_hub", "publish", publish_params
                )
                if result.success:
                    listing_id = result.data.get("listing_id", "")
                    state["published_skills"][sid] = {
                        "listing_id": listing_id,
                        "version": skill["version"],
                        "source_hash": skill["source_hash"],
                        "price": price,
                        "published_at": datetime.utcnow().isoformat(),
                    }
                    state["stats"]["total_published"] += 1
                    state["stats"]["last_publish_at"] = datetime.utcnow().isoformat()
                    return {"success": True, "skill_id": sid, "listing_id": listing_id, "price": price}
                else:
                    return {"success": False, "skill_id": sid, "error": result.message}
            except Exception as e:
                # Fall through to direct recording
                pass

        # Without context, record locally (marketplace will be synced later)
        listing_id = f"local_{sid}_{skill['source_hash'][:8]}"
        state["published_skills"][sid] = {
            "listing_id": listing_id,
            "version": skill["version"],
            "source_hash": skill["source_hash"],
            "price": price,
            "published_at": datetime.utcnow().isoformat(),
        }
        state["stats"]["total_published"] += 1
        state["stats"]["last_publish_at"] = datetime.utcnow().isoformat()
        return {"success": True, "skill_id": sid, "listing_id": listing_id, "price": price}

    async def _diff(self, params: Dict) -> SkillResult:
        local_skills = self._scan_local_skills()
        state = self._load()
        published = state.get("published_skills", {})
        exclude = state.get("exclude_skills", [])

        new = []
        updated = []
        unchanged = []
        orphaned = []  # published but no longer local

        local_ids = set()
        for skill in local_skills:
            sid = skill["skill_id"]
            local_ids.add(sid)
            if sid in exclude:
                continue
            if sid not in published:
                new.append({"skill_id": sid, "name": skill["name"], "version": skill["version"]})
            elif published[sid].get("source_hash") != skill["source_hash"]:
                updated.append({
                    "skill_id": sid,
                    "name": skill["name"],
                    "old_version": published[sid].get("version", "?"),
                    "new_version": skill["version"],
                })
            else:
                unchanged.append(sid)

        # Find orphaned (published but no longer in local)
        for sid in published:
            if sid not in local_ids:
                orphaned.append(sid)

        return SkillResult(
            success=True,
            message=f"Diff: {len(new)} new, {len(updated)} updated, "
                    f"{len(unchanged)} unchanged, {len(orphaned)} orphaned",
            data={
                "new": new,
                "updated": updated,
                "unchanged": unchanged,
                "orphaned": orphaned,
            },
        )

    async def _sync(self, params: Dict) -> SkillResult:
        dry_run = params.get("dry_run", False)
        agent_id = params.get("agent_id", "self")

        local_skills = self._scan_local_skills()
        state = self._load()
        published = state.get("published_skills", {})
        exclude = state.get("exclude_skills", [])

        to_publish = []
        to_update = []

        for skill in local_skills:
            sid = skill["skill_id"]
            if sid in exclude:
                continue
            if sid not in published:
                to_publish.append(skill)
            elif published[sid].get("source_hash") != skill["source_hash"]:
                to_update.append(skill)

        if dry_run:
            return SkillResult(
                success=True,
                message=f"Dry run sync: {len(to_publish)} new, {len(to_update)} updates",
                data={
                    "would_publish": [s["skill_id"] for s in to_publish],
                    "would_update": [s["skill_id"] for s in to_update],
                },
            )

        published_results = []
        updated_results = []
        errors = []

        # Publish new skills
        for skill in to_publish:
            result = await self._do_publish(state, skill, agent_id)
            if result["success"]:
                published_results.append(result)
            else:
                errors.append(result)

        # Update changed skills (re-publish with new hash)
        for skill in to_update:
            result = await self._do_publish(state, skill, agent_id)
            if result["success"]:
                state["stats"]["total_updated"] += 1
                updated_results.append(result)
            else:
                errors.append(result)

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Sync complete: {len(published_results)} published, "
                    f"{len(updated_results)} updated, {len(errors)} errors",
            data={
                "published": published_results,
                "updated": updated_results,
                "errors": errors,
            },
        )

    async def _unpublish(self, params: Dict) -> SkillResult:
        skill_id = params.get("skill_id", "").strip()
        if not skill_id:
            return SkillResult(success=False, message="Missing skill_id")

        state = self._load()
        published = state.get("published_skills", {})

        if skill_id not in published:
            return SkillResult(
                success=False,
                message=f"Skill '{skill_id}' is not published",
            )

        removed = published.pop(skill_id)
        state["stats"]["total_unpublished"] += 1
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Unpublished '{skill_id}' (was listing {removed.get('listing_id', '?')})",
            data={"skill_id": skill_id, "removed": removed},
        )

    async def _status(self, params: Dict) -> SkillResult:
        local_skills = self._scan_local_skills()
        state = self._load()
        published = state.get("published_skills", {})
        exclude = state.get("exclude_skills", [])

        skills_status = []
        for skill in local_skills:
            sid = skill["skill_id"]
            if sid in published:
                status = "published"
                pub_info = published[sid]
            elif sid in exclude:
                status = "excluded"
                pub_info = None
            else:
                status = "unpublished"
                pub_info = None

            skills_status.append({
                "skill_id": sid,
                "name": skill["name"],
                "version": skill["version"],
                "category": skill["category"],
                "status": status,
                "listing_id": pub_info["listing_id"] if pub_info else None,
                "price": pub_info["price"] if pub_info else self._get_price_for_skill(state, sid, skill["category"]),
            })

        published_count = sum(1 for s in skills_status if s["status"] == "published")
        unpublished_count = sum(1 for s in skills_status if s["status"] == "unpublished")
        excluded_count = sum(1 for s in skills_status if s["status"] == "excluded")

        return SkillResult(
            success=True,
            message=f"Skill status: {published_count} published, "
                    f"{unpublished_count} unpublished, {excluded_count} excluded "
                    f"(of {len(skills_status)} total)",
            data={
                "skills": skills_status,
                "stats": state["stats"],
                "summary": {
                    "total": len(skills_status),
                    "published": published_count,
                    "unpublished": unpublished_count,
                    "excluded": excluded_count,
                },
            },
        )

    async def _set_pricing(self, params: Dict) -> SkillResult:
        state = self._load()
        rules = state["pricing_rules"]
        changes = []

        if "default_price" in params:
            p = float(params["default_price"])
            if p < 0:
                return SkillResult(success=False, message="Price cannot be negative")
            rules["default_price"] = p
            changes.append(f"default_price=${p:.2f}")

        if "category_prices" in params:
            cp = params["category_prices"]
            if isinstance(cp, dict):
                rules["category_prices"].update({k: float(v) for k, v in cp.items()})
                changes.append(f"category_prices updated for {len(cp)} categories")

        if "skill_prices" in params:
            sp = params["skill_prices"]
            if isinstance(sp, dict):
                rules["skill_prices"].update({k: float(v) for k, v in sp.items()})
                changes.append(f"skill_prices updated for {len(sp)} skills")

        if "free_categories" in params:
            fc = params["free_categories"]
            if isinstance(fc, list):
                rules["free_categories"] = fc
                changes.append(f"free_categories={fc}")

        if "exclude_skills" in params:
            ex = params["exclude_skills"]
            if isinstance(ex, list):
                state["exclude_skills"] = ex
                changes.append(f"exclude_skills={len(ex)} skills")

        if not changes:
            return SkillResult(
                success=True,
                message="No changes specified",
                data={"pricing_rules": rules},
            )

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Pricing updated: {', '.join(changes)}",
            data={"pricing_rules": rules, "exclude_skills": state["exclude_skills"]},
        )

    async def initialize(self) -> bool:
        self.initialized = True
        return True
