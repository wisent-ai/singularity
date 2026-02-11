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

from ...base import Skill, SkillResult, SkillAction, SkillManifest


def _a(name: str, desc: str, params: Optional[Dict] = None,
       prob: float = 0.9, dur: float = 30) -> SkillAction:
    """Helper to build SkillAction with defaults."""
    return SkillAction(
        name=name,
        description=desc,
        parameters=params or {},
        estimated_cost=0,
        estimated_duration_seconds=dur,
        success_probability=prob,
    )


# Default limits
MAX_OUTPUT_BYTES = 100_000  # 100 KB
DEFAULT_TIMEOUT = 120       # 2 minutes
MAX_TIMEOUT = 600           # 10 minutes
BLOCKED_COMMANDS = frozenset({
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
    ":(){:|:&};:", "fork", "shutdown", "reboot", "halt",
    "init 0", "init 6",
})


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

    def __init__(self, credentials: Dict[str, str] = None,
                 working_dir: str = None, allowed_dirs: list = None):
        super().__init__(credentials)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.allowed_dirs = [Path(d) for d in allowed_dirs] if allowed_dirs else None
        self._background_procs: Dict[str, asyncio.subprocess.Process] = {}

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="shell",
            name="Shell Commands",
            version="1.0.0",
            category="system",
            description=(
                "Execute shell commands and fetch URLs. "
                "Run commands, install packages, manage processes, "
                "and retrieve web content."
            ),
            required_credentials=[],
            install_cost=0,
            author="system",
            actions=[
                _a("run", "Execute a shell command and return stdout/stderr",
                   {"command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (default 120, max 600)"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"}},
                   prob=0.85, dur=30),
                _a("fetch", "Fetch content from a URL via curl",
                   {"url": {"type": "string", "description": "URL to fetch"},
                    "headers": {"type": "object", "description": "Optional HTTP headers dict"},
                    "method": {"type": "string", "description": "HTTP method (default GET)"},
                    "body": {"type": "string", "description": "Request body for POST/PUT"}},
                   prob=0.9, dur=15),
                _a("which", "Check if a command/binary exists on the system",
                   {"command": {"type": "string", "description": "Command name to check"}},
                   prob=0.95, dur=2),
                _a("env", "Get environment info: cwd, user, PATH, and selected env vars",
                   {"vars": {"type": "array", "description": "List of env var names to include (optional)"}},
                   prob=1.0, dur=1),
                _a("background", "Run a command in the background (non-blocking)",
                   {"command": {"type": "string", "description": "Shell command to execute"},
                    "label": {"type": "string", "description": "Label for this background process"}},
                   prob=0.85, dur=5),
                _a("check_background", "Check status of a background process",
                   {"label": {"type": "string", "description": "Label of the background process"}},
                   prob=0.95, dur=2),
                _a("kill", "Kill a background process",
                   {"label": {"type": "string", "description": "Label of the background process to kill"}},
                   prob=0.9, dur=2),
                _a("pipe", "Execute a pipeline of commands (cmd1 | cmd2 | ...)",
                   {"commands": {"type": "array", "description": "List of commands to pipe together"},
                    "timeout": {"type": "number", "description": "Timeout in seconds"}},
                   prob=0.8, dur=30),
            ],
        )

    def check_credentials(self) -> bool:
        return True  # No credentials needed

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            dispatch = {
                "run": self._run,
                "fetch": self._fetch,
                "which": self._which,
                "env": self._env,
                "background": self._background,
                "check_background": self._check_background,
                "kill": self._kill,
                "pipe": self._pipe,
            }
            handler = dispatch.get(action)
            if not handler:
                return SkillResult(
                    success=False,
                    message=f"Unknown action: {action}. Available: {', '.join(dispatch.keys())}",
                )
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Shell error: {e}")

    # ── Action handlers ──────────────────────────────────────────────

    async def _run(self, params: Dict) -> SkillResult:
        """Execute a shell command with timeout and output capture."""
        command = params.get("command", "").strip()
        if not command:
            return SkillResult(success=False, message="'command' parameter is required.")

        # Safety check
        if self._is_blocked(command):
            return SkillResult(
                success=False,
                message=f"Command blocked for safety: {command[:80]}",
            )

        timeout = min(
            float(params.get("timeout", DEFAULT_TIMEOUT)),
            MAX_TIMEOUT,
        )
        cwd = params.get("cwd", str(self.working_dir))

        # Validate working directory if restrictions are set
        if self.allowed_dirs:
            target = Path(cwd).resolve()
            if not any(self._is_subpath(target, d) for d in self.allowed_dirs):
                return SkillResult(
                    success=False,
                    message=f"Directory {cwd} is outside allowed directories.",
                )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env={**os.environ},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return SkillResult(
                    success=False,
                    message=f"Command timed out after {timeout}s: {command[:80]}",
                    data={"exit_code": -1, "timed_out": True},
                )

            stdout_str = self._truncate(stdout)
            stderr_str = self._truncate(stderr)
            exit_code = proc.returncode

            success = exit_code == 0
            message = stdout_str if success else f"Exit {exit_code}: {stderr_str or stdout_str}"
            if len(message) > 2000:
                message = message[:2000] + f"\n... ({len(stdout_str) + len(stderr_str)} chars total)"

            return SkillResult(
                success=success,
                message=message,
                data={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": exit_code,
                    "command": command,
                },
            )

        except OSError as e:
            return SkillResult(success=False, message=f"Failed to execute: {e}")

    async def _fetch(self, params: Dict) -> SkillResult:
        """Fetch URL content using curl subprocess."""
        url = params.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="'url' parameter is required.")

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        body = params.get("body", "")

        # Build curl command
        cmd_parts = ["curl", "-sS", "-L", "--max-time", "30", "-X", method]
        for key, val in headers.items():
            cmd_parts.extend(["-H", f"{key}: {val}"])
        if body and method in ("POST", "PUT", "PATCH"):
            cmd_parts.extend(["-d", body])
        cmd_parts.append(url)

        command = shlex.join(cmd_parts)
        result = await self._run({"command": command, "timeout": 45})

        if result.success:
            content = result.data.get("stdout", "")
            return SkillResult(
                success=True,
                message=f"Fetched {url} ({len(content)} chars)",
                data={"url": url, "method": method, "content": content,
                      "length": len(content)},
            )
        return result

    async def _which(self, params: Dict) -> SkillResult:
        """Check if a command exists on the system."""
        command = params.get("command", "").strip()
        if not command:
            return SkillResult(success=False, message="'command' parameter is required.")

        result = await self._run({"command": f"which {shlex.quote(command)}", "timeout": 5})
        path = result.data.get("stdout", "").strip() if result.success else ""

        return SkillResult(
            success=bool(path),
            message=f"{command} found at {path}" if path else f"{command} not found",
            data={"command": command, "path": path, "exists": bool(path)},
        )

    async def _env(self, params: Dict) -> SkillResult:
        """Get environment info."""
        requested_vars = params.get("vars", [])
        env_data = {
            "cwd": str(self.working_dir),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "home": str(Path.home()),
            "path": os.environ.get("PATH", ""),
            "shell": os.environ.get("SHELL", "unknown"),
            "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
        }

        if requested_vars:
            env_data["requested"] = {
                v: os.environ.get(v, "<not set>") for v in requested_vars
            }

        return SkillResult(
            success=True,
            message=f"Environment: {env_data['platform']}, user={env_data['user']}, cwd={env_data['cwd']}",
            data=env_data,
        )

    async def _background(self, params: Dict) -> SkillResult:
        """Start a command in the background."""
        command = params.get("command", "").strip()
        label = params.get("label", "").strip()
        if not command:
            return SkillResult(success=False, message="'command' parameter is required.")
        if not label:
            label = f"bg_{len(self._background_procs)}"

        if label in self._background_procs:
            proc = self._background_procs[label]
            if proc.returncode is None:
                return SkillResult(
                    success=False,
                    message=f"Background process '{label}' is already running. Kill it first.",
                )

        if self._is_blocked(command):
            return SkillResult(success=False, message=f"Command blocked for safety.")

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.working_dir),
        )
        self._background_procs[label] = proc

        return SkillResult(
            success=True,
            message=f"Background process '{label}' started (PID {proc.pid})",
            data={"label": label, "pid": proc.pid, "command": command},
        )

    async def _check_background(self, params: Dict) -> SkillResult:
        """Check status of a background process."""
        label = params.get("label", "").strip()
        if not label:
            # List all background processes
            procs = {}
            for lbl, proc in self._background_procs.items():
                procs[lbl] = {
                    "pid": proc.pid,
                    "running": proc.returncode is None,
                    "exit_code": proc.returncode,
                }
            return SkillResult(
                success=True,
                message=f"{len(procs)} background process(es)",
                data={"processes": procs},
            )

        proc = self._background_procs.get(label)
        if not proc:
            return SkillResult(
                success=False,
                message=f"No background process with label '{label}'",
            )

        running = proc.returncode is None
        data = {"label": label, "pid": proc.pid, "running": running,
                "exit_code": proc.returncode}

        if not running:
            stdout = await proc.stdout.read() if proc.stdout else b""
            stderr = await proc.stderr.read() if proc.stderr else b""
            data["stdout"] = self._truncate(stdout)
            data["stderr"] = self._truncate(stderr)

        return SkillResult(
            success=True,
            message=f"Process '{label}': {'running' if running else f'exited ({proc.returncode})'}",
            data=data,
        )

    async def _kill(self, params: Dict) -> SkillResult:
        """Kill a background process."""
        label = params.get("label", "").strip()
        if not label:
            return SkillResult(success=False, message="'label' parameter is required.")

        proc = self._background_procs.get(label)
        if not proc:
            return SkillResult(
                success=False,
                message=f"No background process with label '{label}'",
            )

        if proc.returncode is not None:
            return SkillResult(
                success=True,
                message=f"Process '{label}' already exited ({proc.returncode})",
            )

        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass

        return SkillResult(
            success=True,
            message=f"Process '{label}' killed (PID {proc.pid})",
            data={"label": label, "pid": proc.pid},
        )

    async def _pipe(self, params: Dict) -> SkillResult:
        """Execute a pipeline of commands."""
        commands = params.get("commands", [])
        if not commands or len(commands) < 2:
            return SkillResult(
                success=False,
                message="'commands' must be a list of 2+ commands to pipe together.",
            )

        # Join with pipes and execute as single shell command
        pipeline = " | ".join(commands)
        timeout = min(float(params.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)

        return await self._run({"command": pipeline, "timeout": timeout})

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _truncate(data: bytes, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
        """Decode and truncate output."""
        text = data.decode("utf-8", errors="replace")
        if len(text) > max_bytes:
            return text[:max_bytes] + f"\n... (truncated, {len(text)} total chars)"
        return text

    @staticmethod
    def _is_blocked(command: str) -> bool:
        """Check if a command is in the blocked list."""
        normalized = command.strip().lower()
        for blocked in BLOCKED_COMMANDS:
            if blocked in normalized:
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
