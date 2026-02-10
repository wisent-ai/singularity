"""Filesystem Skill - OpenCode-style file operations."""

import os
import re
import glob as glob_module
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction
from . import handlers


def _a(n, d, p, prob=0.95):
    return SkillAction(name=n, description=d, parameters=p, estimated_cost=0, success_probability=prob)


class FilesystemSkill(Skill):
    """OpenCode-style filesystem operations: glob, grep, view, write, patch, ls."""

    def __init__(self, credentials: Dict[str, str] = None, base_path: str = None):
        super().__init__(credentials)
        self.base_path = Path(base_path) if base_path else Path.cwd()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="filesystem", name="Filesystem Operations", version="1.0.0", category="system",
            description="Read, write, search files - OpenCode style",
            required_credentials=[], install_cost=0,
            actions=[
                _a("glob", "Find files matching a pattern (e.g. **/*.py)",
                   {"pattern": "glob pattern", "path": "base path (optional)"}),
                _a("grep", "Search file contents with regex",
                   {"pattern": "regex pattern", "path": "file or directory", "include": "file pattern (e.g. *.py)"}, 0.9),
                _a("view", "Read file contents with optional offset/limit",
                   {"path": "file path", "offset": "line offset (optional)", "limit": "max lines (optional)"}),
                _a("write", "Write content to a file", {"path": "file path", "content": "content to write"}),
                _a("patch", "Apply a unified diff patch to a file", {"path": "file path", "patch": "unified diff content"}, 0.85),
                _a("ls", "List directory contents", {"path": "directory path", "pattern": "filter pattern (optional)"}),
                _a("mkdir", "Create a directory", {"path": "directory path"}),
                _a("rm", "Remove a file or directory", {"path": "path to remove", "recursive": "remove recursively (optional)"}, 0.9),
            ])

    def check_credentials(self) -> bool:
        return True

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.base_path / p

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            dispatch = {
                "glob": lambda: handlers.do_glob(self, params.get("pattern", "*"), params.get("path")),
                "grep": lambda: handlers.do_grep(self, params.get("pattern", ""),
                    params.get("path", "."), params.get("include")),
                "view": lambda: handlers.do_view(self, params.get("path", ""),
                    params.get("offset", 0), params.get("limit")),
                "write": lambda: handlers.do_write(self, params.get("path", ""), params.get("content", "")),
                "patch": lambda: handlers.do_patch(self, params.get("path", ""), params.get("patch", "")),
                "ls": lambda: handlers.do_ls(self, params.get("path", "."), params.get("pattern")),
                "mkdir": lambda: handlers.do_mkdir(self, params.get("path", "")),
                "rm": lambda: handlers.do_rm(self, params.get("path", ""), params.get("recursive", False)),
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(success=False, message=f"Unknown action: {action}")
            return await handler()
        except Exception as e:
            return SkillResult(success=False, message=str(e))
