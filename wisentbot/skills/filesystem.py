#!/usr/bin/env python3
"""
Filesystem Skill - OpenCode-style file operations

Real file access. No mocks.
"""

import os
import re
import glob as glob_module
import subprocess
from typing import Dict, List, Optional
from pathlib import Path
from .base import Skill, SkillResult, SkillManifest, SkillAction


class FilesystemSkill(Skill):
    """
    OpenCode-style filesystem operations.

    Tools: glob, grep, view, write, patch, ls
    No credentials needed.
    """

    def __init__(self, credentials: Dict[str, str] = None, base_path: str = None):
        super().__init__(credentials)
        self.base_path = Path(base_path) if base_path else Path.cwd()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="filesystem",
            name="Filesystem Operations",
            version="1.0.0",
            category="system",
            description="Read, write, search files - OpenCode style",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="glob",
                    description="Find files matching a pattern (e.g. **/*.py)",
                    parameters={"pattern": "glob pattern", "path": "base path (optional)"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="grep",
                    description="Search file contents with regex",
                    parameters={
                        "pattern": "regex pattern",
                        "path": "file or directory to search",
                        "include": "file pattern to include (e.g. *.py)"
                    },
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="view",
                    description="Read file contents with optional offset/limit",
                    parameters={
                        "path": "file path",
                        "offset": "line offset (optional)",
                        "limit": "max lines (optional)"
                    },
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="write",
                    description="Write content to a file",
                    parameters={"path": "file path", "content": "content to write"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="patch",
                    description="Apply a unified diff patch to a file",
                    parameters={"path": "file path", "patch": "unified diff content"},
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="ls",
                    description="List directory contents",
                    parameters={"path": "directory path", "pattern": "filter pattern (optional)"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="mkdir",
                    description="Create a directory",
                    parameters={"path": "directory path"},
                    estimated_cost=0,
                    success_probability=0.95
                ),
                SkillAction(
                    name="rm",
                    description="Remove a file or directory",
                    parameters={"path": "path to remove", "recursive": "remove recursively (optional)"},
                    estimated_cost=0,
                    success_probability=0.9
                ),
            ]
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base_path"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            if action == "glob":
                return await self._glob(params.get("pattern", "*"), params.get("path"))
            elif action == "grep":
                return await self._grep(
                    params.get("pattern", ""),
                    params.get("path", "."),
                    params.get("include")
                )
            elif action == "view":
                return await self._view(
                    params.get("path", ""),
                    params.get("offset", 0),
                    params.get("limit")
                )
            elif action == "write":
                return await self._write(params.get("path", ""), params.get("content", ""))
            elif action == "patch":
                return await self._patch(params.get("path", ""), params.get("patch", ""))
            elif action == "ls":
                return await self._ls(params.get("path", "."), params.get("pattern"))
            elif action == "mkdir":
                return await self._mkdir(params.get("path", ""))
            elif action == "rm":
                return await self._rm(params.get("path", ""), params.get("recursive", False))
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    async def _glob(self, pattern: str, path: Optional[str] = None) -> SkillResult:
        """Find files matching pattern"""
        base = self._resolve_path(path) if path else self.base_path
        full_pattern = str(base / pattern)

        matches = glob_module.glob(full_pattern, recursive=True)
        matches = [str(Path(m).relative_to(self.base_path)) for m in matches[:100]]

        return SkillResult(
            success=True,
            message=f"Found {len(matches)} files",
            data={"files": matches}
        )

    async def _grep(self, pattern: str, path: str, include: Optional[str] = None) -> SkillResult:
        """Search file contents with regex"""
        target = self._resolve_path(path)
        results = []

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return SkillResult(success=False, message=f"Invalid regex: {e}")

        files_to_search = []
        if target.is_file():
            files_to_search = [target]
        elif target.is_dir():
            glob_pattern = include or "*"
            files_to_search = list(target.rglob(glob_pattern))[:50]

        for file_path in files_to_search:
            if not file_path.is_file():
                continue
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append({
                                "file": str(file_path.relative_to(self.base_path)),
                                "line": i,
                                "content": line.strip()[:200]
                            })
                            if len(results) >= 50:
                                break
            except Exception:
                continue

            if len(results) >= 50:
                break

        return SkillResult(
            success=True,
            message=f"Found {len(results)} matches",
            data={"matches": results}
        )

    async def _view(self, path: str, offset: int = 0, limit: Optional[int] = None) -> SkillResult:
        """Read file contents"""
        target = self._resolve_path(path)

        if not target.exists():
            return SkillResult(success=False, message=f"File not found: {path}")

        if not target.is_file():
            return SkillResult(success=False, message=f"Not a file: {path}")

        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            return SkillResult(success=False, message=f"Read error: {e}")

        total_lines = len(lines)
        start = max(0, offset)
        end = start + limit if limit else len(lines)
        content = ''.join(lines[start:end])

        return SkillResult(
            success=True,
            message=f"Read {path} ({total_lines} lines)",
            data={
                "content": content[:10000],  # Limit size
                "total_lines": total_lines,
                "offset": start,
                "lines_returned": min(end - start, total_lines - start)
            }
        )

    async def _write(self, path: str, content: str) -> SkillResult:
        """Write content to file"""
        target = self._resolve_path(path)

        # Create parent directories
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return SkillResult(success=False, message=f"Write error: {e}")

        return SkillResult(
            success=True,
            message=f"Wrote {len(content)} bytes to {path}",
            data={"path": str(target), "bytes": len(content)}
        )

    async def _patch(self, path: str, patch_content: str) -> SkillResult:
        """Apply unified diff patch"""
        target = self._resolve_path(path)

        if not target.exists():
            return SkillResult(success=False, message=f"File not found: {path}")

        # Write patch to temp file and apply
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch_content)
            patch_file = f.name

        try:
            result = subprocess.run(
                ['patch', str(target), patch_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            os.unlink(patch_file)

            if result.returncode == 0:
                return SkillResult(
                    success=True,
                    message=f"Patched {path}",
                    data={"output": result.stdout}
                )
            else:
                return SkillResult(
                    success=False,
                    message=f"Patch failed: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            os.unlink(patch_file)
            return SkillResult(success=False, message="Patch timed out")
        except FileNotFoundError:
            os.unlink(patch_file)
            return SkillResult(success=False, message="patch command not found")

    async def _ls(self, path: str, pattern: Optional[str] = None) -> SkillResult:
        """List directory contents"""
        target = self._resolve_path(path)

        if not target.exists():
            return SkillResult(success=False, message=f"Path not found: {path}")

        if not target.is_dir():
            return SkillResult(success=False, message=f"Not a directory: {path}")

        entries = []
        for entry in sorted(target.iterdir())[:100]:
            if pattern and not glob_module.fnmatch.fnmatch(entry.name, pattern):
                continue
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None
            })

        return SkillResult(
            success=True,
            message=f"Listed {len(entries)} entries",
            data={"entries": entries}
        )

    async def _mkdir(self, path: str) -> SkillResult:
        """Create directory"""
        target = self._resolve_path(path)

        try:
            target.mkdir(parents=True, exist_ok=True)
            return SkillResult(
                success=True,
                message=f"Created directory: {path}"
            )
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    async def _rm(self, path: str, recursive: bool = False) -> SkillResult:
        """Remove file or directory"""
        target = self._resolve_path(path)

        if not target.exists():
            return SkillResult(success=False, message=f"Path not found: {path}")

        try:
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                if recursive:
                    import shutil
                    shutil.rmtree(target)
                else:
                    target.rmdir()
            return SkillResult(
                success=True,
                message=f"Removed: {path}"
            )
        except Exception as e:
            return SkillResult(success=False, message=str(e))
