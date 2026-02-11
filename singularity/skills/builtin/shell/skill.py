"""
Shell Command Skill — execute shell commands and fetch URLs.

Provides safe command execution with timeouts, output capture, and
URL fetching. Commands run in a subprocess with configurable limits.
"""

import asyncio
import os
import shlex
import signal
from pathlib import Path
from typing import Dict, Optional

from singularity.skills.base import Skill, SkillResult, SkillAction, SkillManifest

MAX_OUTPUT_BYTES = 100_000
DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT = 600.0

BLOCKED_COMMANDS = frozenset([
    "rm -rf /",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
])


def _a(name: str, desc: str, params: Dict, prob: float = 0.8, dur: float = 10):
    """Helper to build SkillAction with defaults."""
    return SkillAction(name=name, description=desc, parameters=params, success_probability=prob, estimated_duration_seconds=dur)


class ShellSkill(Skill):
    """
    Execute shell commands and fetch URLs.

    Features:
    - Run arbitrary shell commands with timeout and output limits
    - Fetch URL content (text/HTML) via curl
    - Check if commands/binaries exist
    - Get environment info (cwd, env vars, PATH)
    - Background command execution
    """

    def __init__(self, credentials: Optional[Dict] = None, working_dir: Optional[str] = None, allowed_dirs: Optional[list] = None):
        super().__init__(credentials)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.allowed_dirs = allowed_dirs
        self._background_procs: Dict = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="shell",
            name="Shell Commands",
            version="1.0.0",
            category="system",
            description="Execute shell commands and fetch URLs. Run commands, install packages, manage processes, and retrieve web content.",
            required_credentials=[],
            actions=[
                _a("run", "Execute a shell command and return stdout/stderr",
                   {"command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (default 120, max 600)"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"}}, 0.85),
                _a("fetch", "Fetch URL content via curl",
                   {"url": {"type": "string", "description": "URL to fetch"},
                    "method": {"type": "string", "description": "HTTP method (default GET)"},
                    "headers": {"type": "object", "description": "HTTP headers"},
                    "body": {"type": "string", "description": "Request body"}}, 0.80),
                _a("which", "Check if a command exists",
                   {"command": {"type": "string", "description": "Command name to look up"}}, 0.95),
                _a("env", "Get environment info",
                   {"vars": {"type": "array", "description": "Specific env vars to retrieve"}}, 0.95),
                _a("background", "Start a command in the background",
                   {"command": {"type": "string", "description": "Command to run"},
                    "label": {"type": "string", "description": "Label for the background process"}}, 0.80),
                _a("check_background", "Check status of a background process",
                   {"label": {"type": "string", "description": "Label of the background process"}}, 0.90),
                _a("kill", "Kill a background process",
                   {"label": {"type": "string", "description": "Label of the process to kill"}}, 0.90),
                _a("pipe", "Execute a pipeline of commands",
                   {"commands": {"type": "array", "description": "List of commands to pipe"},
                    "timeout": {"type": "number", "description": "Timeout in seconds"}}, 0.80),
            ]
        )

    async def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "run": self._run,
            "fetch": self._fetch,
            "which": self._which,
            "env": self._env,
            "background": self._background,
            "check_background": self._check_background,
            "kill": self._kill,
            "pipe": self._pipe,
        }
        handler = handlers.get(action)
        if handler:
            try:
                return await handler(params)
            except Exception as e:
                return SkillResult(success=False, data=None, message=f"Shell error: {e}")
        return SkillResult(
            success=False, data=None,
            message=f"Unknown action: {action}. Available: {', '.join(handlers.keys())}"
        )

    async def _run(self, params: Dict) -> SkillResult:
        """Execute a shell command with timeout and output capture."""
        command = params.get("command", "").strip()
        if not command:
            return SkillResult(success=False, data=None, message="'command' parameter is required.")

        if self._is_blocked(command):
            return SkillResult(success=False, data=None, message=f"Command blocked for safety: {command[:80]}")

        timeout = min(float(params.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)
        cwd = str(params.get("cwd", self.working_dir))

        if self.allowed_dirs:
            cwd_path = Path(cwd).resolve()
            if not any(self._is_subpath(cwd_path, Path(d).resolve()) for d in self.allowed_dirs):
                return SkillResult(success=False, data=None, message=f"Directory {cwd} is outside allowed directories.")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=os.environ,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return SkillResult(
                    success=False, data=None,
                    message=f"Command timed out after {timeout}s: {command[:80]}",
                )

            out = self._truncate(stdout, MAX_OUTPUT_BYTES)
            err = self._truncate(stderr, MAX_OUTPUT_BYTES)
            return SkillResult(
                success=proc.returncode == 0,
                data={"stdout": out, "stderr": err, "returncode": proc.returncode},
                message=out if len(out) < 500 else out[:500],
            )
        except OSError as e:
            return SkillResult(success=False, data=None, message=str(e))

    async def _fetch(self, params: Dict) -> SkillResult:
        """Fetch URL content using curl subprocess."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, data=None, message="'url' parameter is required.")
        if not url.startswith("https://"):
            url = "https://" + url

        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        body = params.get("body", "")

        cmd_parts = ["curl", "-sS", "-L", "--max-time", "30", "-X", method]
        for k, v in headers.items():
            cmd_parts.extend(["-H", f"{k}: {v}"])
        if body:
            cmd_parts.extend(["-d", body])
        cmd_parts.append(url)

        result = await self._run({"command": shlex.join(cmd_parts)})
        if result.success and result.data:
            content = result.data.get("stdout", "")
            return SkillResult(success=True, data={"content": content, "url": url, "length": len(content)})
        return result

    async def _which(self, params: Dict) -> SkillResult:
        """Check if a command exists on the system."""
        command = params.get("command", "").strip()
        if not command:
            return SkillResult(success=False, data=None, message="'command' parameter is required.")

        result = await self._run({"command": f"which {shlex.quote(command)}", "timeout": 5})
        if result.success and result.data:
            path = result.data.get("stdout", "").strip()
            return SkillResult(success=True, data={"command": command, "path": path}, message=f"{command} found at {path}")
        return SkillResult(success=False, data={"command": command, "found": False}, message=f"{command} not found")

    async def _env(self, params: Dict) -> SkillResult:
        """Get environment info."""
        requested = params.get("vars", ["USER", "USERNAME", "PATH", "SHELL"])
        env_info = {v: os.environ.get(v, "") for v in requested}
        user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
        platform = "unknown"
        if hasattr(os, "uname"):
            platform = os.uname().sysname

        return SkillResult(
            success=True,
            data={"cwd": str(self.working_dir), "platform": platform, "user": user, "requested": env_info},
            message=f"Environment: {platform}, user={user}",
        )

    async def _background(self, params: Dict) -> SkillResult:
        """Start a command in the background."""
        command = params.get("command", "").strip()
        label = params.get("label", "")
        if not command:
            return SkillResult(success=False, data=None, message="'command' parameter is required.")
        if not label:
            label = f"bg_{len(self._background_procs)}"

        if label in self._background_procs and self._background_procs[label].returncode is None:
            return SkillResult(success=False, data=None, message=f"Background process '{label}' is already running. Kill it first.")

        if self._is_blocked(command):
            return SkillResult(success=False, data=None, message="Command blocked for safety.")

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.working_dir),
        )
        self._background_procs[label] = proc
        return SkillResult(success=True, data={"label": label, "pid": proc.pid}, message=f"Background process '{label}' started (PID {proc.pid})")

    async def _check_background(self, params: Dict) -> SkillResult:
        """Check status of a background process."""
        label = params.get("label", "").strip()
        if not label:
            # List all
            procs = {k: {"pid": p.pid, "running": p.returncode is None} for k, p in self._background_procs.items()}
            return SkillResult(success=True, data={"processes": procs}, message=f"{len(procs)} background process(es)")

        if label not in self._background_procs:
            return SkillResult(success=False, data=None, message=f"No background process with label '{label}'")

        proc = self._background_procs[label]
        if proc.returncode is None:
            return SkillResult(success=True, data={"label": label, "pid": proc.pid, "status": "running"}, message=f"Process '{label}': running")

        stdout = await proc.stdout.read() if proc.stdout else b""
        stderr = await proc.stderr.read() if proc.stderr else b""
        return SkillResult(
            success=True,
            data={"label": label, "returncode": proc.returncode, "stdout": self._truncate(stdout, MAX_OUTPUT_BYTES), "stderr": self._truncate(stderr, MAX_OUTPUT_BYTES)},
            message=f"Process '{label}': exited ({proc.returncode})",
        )

    async def _kill(self, params: Dict) -> SkillResult:
        """Kill a background process."""
        label = params.get("label", "").strip()
        if not label:
            return SkillResult(success=False, data=None, message="'label' parameter is required.")

        if label not in self._background_procs:
            return SkillResult(success=False, data=None, message=f"No background process with label '{label}'")

        proc = self._background_procs[label]
        if proc.returncode is not None:
            return SkillResult(success=True, data=None, message=f"Process '{label}' already exited ({proc.returncode})")

        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass

        return SkillResult(success=True, data={"pid": proc.pid}, message=f"Process '{label}' killed (PID {proc.pid})")

    async def _pipe(self, params: Dict) -> SkillResult:
        """Execute a pipeline of commands."""
        commands = params.get("commands", [])
        if len(commands) < 2:
            return SkillResult(success=False, data=None, message="'commands' must be a list of 2+ commands to pipe together.")

        pipeline = " | ".join(commands)
        timeout = min(float(params.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)
        return await self._run({"command": pipeline, "timeout": timeout})

    @staticmethod
    def _truncate(data: bytes, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
        """Decode and truncate output."""
        text = data.decode("utf-8", "replace")
        if len(text) <= max_bytes:
            return text
        return text[:max_bytes] + f"\n... (truncated, {len(text)} total chars)"

    @staticmethod
    def _is_blocked(command: str) -> bool:
        """Check if a command is in the blocked list."""
        cmd_lower = command.strip().lower()
        for blocked in BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return True
        return False

    @staticmethod
    def _is_subpath(path: Path, parent: Path) -> bool:
        """Check if path is under parent directory."""
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False
