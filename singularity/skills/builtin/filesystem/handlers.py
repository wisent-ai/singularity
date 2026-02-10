"""Handler functions for FilesystemSkill actions."""

import os
import re
import glob as glob_module
import subprocess
from typing import Optional
from pathlib import Path
from singularity.skills.base import SkillResult


async def do_glob(skill, pattern: str, path: Optional[str] = None) -> SkillResult:
    base = skill._resolve_path(path) if path else skill.base_path
    full_pattern = str(base / pattern)
    matches = glob_module.glob(full_pattern, recursive=True)
    matches = [str(Path(m).relative_to(skill.base_path)) for m in matches[:100]]
    return SkillResult(success=True, message=f"Found {len(matches)} files", data={"files": matches})


async def do_grep(skill, pattern: str, path: str, include: Optional[str] = None) -> SkillResult:
    target = skill._resolve_path(path)
    results = []
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return SkillResult(success=False, message=f"Invalid regex: {e}")
    files_to_search = []
    if target.is_file():
        files_to_search = [target]
    elif target.is_dir():
        files_to_search = list(target.rglob(include or "*"))[:50]
    for file_path in files_to_search:
        if not file_path.is_file():
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append({"file": str(file_path.relative_to(skill.base_path)),
                                        "line": i, "content": line.strip()[:200]})
                        if len(results) >= 50:
                            break
        except Exception:
            continue
        if len(results) >= 50:
            break
    return SkillResult(success=True, message=f"Found {len(results)} matches", data={"matches": results})


async def do_view(skill, path: str, offset: int = 0, limit: Optional[int] = None) -> SkillResult:
    target = skill._resolve_path(path)
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
    return SkillResult(success=True, message=f"Read {path} ({total_lines} lines)",
        data={"content": content[:10000], "total_lines": total_lines,
              "offset": start, "lines_returned": min(end - start, total_lines - start)})


async def do_write(skill, path: str, content: str) -> SkillResult:
    target = skill._resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        return SkillResult(success=False, message=f"Write error: {e}")
    return SkillResult(success=True, message=f"Wrote {len(content)} bytes to {path}",
                       data={"path": str(target), "bytes": len(content)})


async def do_patch(skill, path: str, patch_content: str) -> SkillResult:
    target = skill._resolve_path(path)
    if not target.exists():
        return SkillResult(success=False, message=f"File not found: {path}")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch_content)
        patch_file = f.name
    try:
        result = subprocess.run(['patch', str(target), patch_file],
                                capture_output=True, text=True)
        os.unlink(patch_file)
        if result.returncode == 0:
            return SkillResult(success=True, message=f"Patched {path}", data={"output": result.stdout})
        return SkillResult(success=False, message=f"Patch failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        os.unlink(patch_file)
        return SkillResult(success=False, message="Patch timed out")
    except FileNotFoundError:
        os.unlink(patch_file)
        return SkillResult(success=False, message="patch command not found")


async def do_ls(skill, path: str, pattern: Optional[str] = None) -> SkillResult:
    target = skill._resolve_path(path)
    if not target.exists():
        return SkillResult(success=False, message=f"Path not found: {path}")
    if not target.is_dir():
        return SkillResult(success=False, message=f"Not a directory: {path}")
    entries = []
    for entry in sorted(target.iterdir())[:100]:
        if pattern and not glob_module.fnmatch.fnmatch(entry.name, pattern):
            continue
        entries.append({"name": entry.name, "type": "dir" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else None})
    return SkillResult(success=True, message=f"Listed {len(entries)} entries", data={"entries": entries})


async def do_mkdir(skill, path: str) -> SkillResult:
    target = skill._resolve_path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        return SkillResult(success=True, message=f"Created directory: {path}")
    except Exception as e:
        return SkillResult(success=False, message=str(e))


async def do_rm(skill, path: str, recursive: bool = False) -> SkillResult:
    target = skill._resolve_path(path)
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
        return SkillResult(success=True, message=f"Removed: {path}")
    except Exception as e:
        return SkillResult(success=False, message=str(e))
