#!/usr/bin/env python3
"""
Shell Skill - Command execution

Real shell commands. No mocks.
"""

import asyncio
import subprocess
import os
from typing import Dict, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction


class ShellSkill(Skill):
    """
    Shell command execution.

    Tools: bash, fetch
    No credentials needed.
    """

    def __init__(self, credentials: Dict[str, str] = None, cwd: str = None):
        super().__init__(credentials)
        self.cwd = cwd or os.getcwd()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="shell",
            name="Shell Commands",
            version="1.0.0",
            category="system",
            description="Execute shell commands and fetch URLs",
            required_credentials=[],
            install_cost=0,
            actions=[
                SkillAction(
                    name="bash",
                    description="Execute a bash command",
                    parameters={
                        "command": "command to execute",
                        "timeout": "timeout in seconds (default 30)",
                        "cwd": "working directory (optional)"
                    },
                    estimated_cost=0,
                    success_probability=0.85
                ),
                SkillAction(
                    name="fetch",
                    description="Fetch content from a URL",
                    parameters={
                        "url": "URL to fetch",
                        "method": "HTTP method (default GET)",
                        "headers": "optional headers dict",
                        "body": "optional request body"
                    },
                    estimated_cost=0,
                    success_probability=0.9
                ),
                SkillAction(
                    name="spawn",
                    description="Spawn a background process",
                    parameters={
                        "command": "command to run in background",
                        "cwd": "working directory (optional)"
                    },
                    estimated_cost=0,
                    success_probability=0.85
                ),
            ]
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            if action == "bash":
                return await self._bash(
                    params.get("command", ""),
                    params.get("timeout", 30),
                    params.get("cwd")
                )
            elif action == "fetch":
                return await self._fetch(
                    params.get("url", ""),
                    params.get("method", "GET"),
                    params.get("headers"),
                    params.get("body")
                )
            elif action == "spawn":
                return await self._spawn(
                    params.get("command", ""),
                    params.get("cwd")
                )
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=str(e))

    def _check_dangerous_command(self, command: str) -> Optional[str]:
        """Check if a command matches known dangerous patterns.

        Returns an error message if the command is dangerous, None if safe.
        Applied to both interactive (_bash) and background (_spawn) execution
        to prevent bypassing safety checks via background processes.
        """
        dangerous = ['rm -rf /', 'mkfs', ':(){:|:&};:', 'dd if=/dev/zero']
        for d in dangerous:
            if d in command:
                return f"Blocked dangerous command: matches '{d}'"
        return None

    async def _bash(self, command: str, timeout: int = 30, cwd: Optional[str] = None) -> SkillResult:
        """Execute bash command"""
        if not command:
            return SkillResult(success=False, message="No command provided")

        # Security: block dangerous commands
        blocked = self._check_dangerous_command(command)
        if blocked:
            return SkillResult(success=False, message=blocked)

        work_dir = cwd or self.cwd

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return SkillResult(
                    success=False,
                    message=f"Command timed out after {timeout}s"
                )

            stdout_str = stdout.decode('utf-8', errors='ignore')[:5000]
            stderr_str = stderr.decode('utf-8', errors='ignore')[:2000]

            return SkillResult(
                success=proc.returncode == 0,
                message=f"Exit code: {proc.returncode}",
                data={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": proc.returncode
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Execution error: {e}")

    async def _fetch(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict] = None,
        body: Optional[str] = None
    ) -> SkillResult:
        """Fetch URL content"""
        if not url:
            return SkillResult(success=False, message="No URL provided")

        import httpx

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body
                )

                content = response.text[:10000]  # Limit response size

                return SkillResult(
                    success=response.status_code < 400,
                    message=f"HTTP {response.status_code}",
                    data={
                        "status": response.status_code,
                        "headers": dict(response.headers),
                        "content": content,
                        "url": str(response.url)
                    }
                )
        except httpx.TimeoutException:
            return SkillResult(success=False, message="Request timed out")
        except Exception as e:
            return SkillResult(success=False, message=f"Fetch error: {e}")

    async def _spawn(self, command: str, cwd: Optional[str] = None) -> SkillResult:
        """Spawn background process"""
        if not command:
            return SkillResult(success=False, message="No command provided")

        # Security: block dangerous commands (same checks as _bash)
        blocked = self._check_dangerous_command(command)
        if blocked:
            return SkillResult(success=False, message=blocked)

        work_dir = cwd or self.cwd

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=work_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            return SkillResult(
                success=True,
                message=f"Spawned process {proc.pid}",
                data={"pid": proc.pid, "command": command}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Spawn error: {e}")
