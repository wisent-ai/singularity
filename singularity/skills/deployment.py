#!/usr/bin/env python3
"""
Deployment Skill - Enable the agent to deploy itself and replicas to cloud environments.

This is critical infrastructure for the Replication pillar. Without deployment
capability, replicated agents only exist as in-memory configurations. This skill
enables agents to:

- Generate deployment configurations (Dockerfile, docker-compose, fly.toml, railway.json)
- Build and push Docker images
- Deploy to cloud platforms (fly.io, Railway, Docker hosts)
- Monitor deployment status and health
- Scale deployments up/down
- Rollback to previous versions
- Track deployment history and costs

Architecture:
  DeploymentSkill generates platform-specific deployment configurations,
  manages deployment state in a JSON file, and shells out to platform CLIs
  (docker, flyctl, railway) for actual deployment operations.

Part of the Replication pillar - enables replicas to actually run somewhere.
"""

import json
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import Skill, SkillResult, SkillManifest, SkillAction

DEPLOY_FILE = Path(__file__).parent.parent / "data" / "deployments.json"
DEPLOY_HISTORY_FILE = Path(__file__).parent.parent / "data" / "deploy_history.json"

# Supported platforms
PLATFORMS = ["docker", "fly.io", "railway", "local"]

# Default resource limits
DEFAULT_RESOURCES = {
    "cpu": "1",
    "memory": "512MB",
    "disk": "1GB",
}


def _load_deployments() -> Dict:
    """Load deployment state."""
    if DEPLOY_FILE.exists():
        try:
            return json.loads(DEPLOY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"deployments": {}, "templates": {}}


def _save_deployments(data: Dict) -> None:
    """Save deployment state."""
    DEPLOY_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEPLOY_FILE.write_text(json.dumps(data, indent=2, default=str))


def _load_history() -> List[Dict]:
    """Load deployment history."""
    if DEPLOY_HISTORY_FILE.exists():
        try:
            return json.loads(DEPLOY_HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(history: List[Dict]) -> None:
    """Save deployment history."""
    DEPLOY_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Keep last 500 entries
    history = history[-500:]
    DEPLOY_HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


def _record_event(deployment_id: str, event_type: str, details: Dict = None) -> None:
    """Record a deployment event in history."""
    history = _load_history()
    history.append({
        "id": str(uuid.uuid4())[:8],
        "deployment_id": deployment_id,
        "event_type": event_type,
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
    })
    _save_history(history)


class DeploymentSkill(Skill):
    """Skill for deploying the agent and replicas to cloud environments."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="deployment",
            name="Deployment Skill",
            version="1.0.0",
            category="infrastructure",
            description="Deploy the agent and replicas to cloud environments (Docker, fly.io, Railway)",
            actions=[
                SkillAction(
                    name="generate_config",
                    description="Generate deployment configuration for a target platform",
                    parameters={
                        "platform": {"type": "str", "required": True, "description": "Target platform: docker, fly.io, railway"},
                        "name": {"type": "str", "required": True, "description": "Deployment name"},
                        "env_vars": {"type": "dict", "required": False, "description": "Environment variables to include"},
                        "resources": {"type": "dict", "required": False, "description": "Resource limits (cpu, memory, disk)"},
                        "port": {"type": "int", "required": False, "description": "Port to expose (default 8080)"},
                        "replicas": {"type": "int", "required": False, "description": "Number of replicas (default 1)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="deploy",
                    description="Deploy or update a deployment to the target platform",
                    parameters={
                        "deployment_id": {"type": "str", "required": True, "description": "ID of the deployment to deploy"},
                    },
                    estimated_cost=0.01,
                ),
                SkillAction(
                    name="status",
                    description="Check the status of a deployment",
                    parameters={
                        "deployment_id": {"type": "str", "required": True, "description": "Deployment ID to check"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_deployments",
                    description="List all known deployments",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="scale",
                    description="Scale a deployment up or down",
                    parameters={
                        "deployment_id": {"type": "str", "required": True, "description": "Deployment ID"},
                        "replicas": {"type": "int", "required": True, "description": "Target number of replicas"},
                    },
                    estimated_cost=0.005,
                ),
                SkillAction(
                    name="rollback",
                    description="Rollback a deployment to a previous version",
                    parameters={
                        "deployment_id": {"type": "str", "required": True, "description": "Deployment ID"},
                        "version": {"type": "int", "required": False, "description": "Version to rollback to (default: previous)"},
                    },
                    estimated_cost=0.01,
                ),
                SkillAction(
                    name="destroy",
                    description="Destroy a deployment and clean up resources",
                    parameters={
                        "deployment_id": {"type": "str", "required": True, "description": "Deployment ID to destroy"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_deploy_history",
                    description="Get deployment history and events",
                    parameters={
                        "deployment_id": {"type": "str", "required": False, "description": "Filter by deployment ID"},
                        "limit": {"type": "int", "required": False, "description": "Max events to return (default 20)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="generate_dockerfile",
                    description="Generate an optimized Dockerfile for the agent",
                    parameters={
                        "base_image": {"type": "str", "required": False, "description": "Base Docker image (default python:3.11-slim)"},
                        "include_gpu": {"type": "bool", "required": False, "description": "Include GPU support (default False)"},
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
            author="singularity",
        )

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        actions = {
            "generate_config": self._generate_config,
            "deploy": self._deploy,
            "status": self._status,
            "list_deployments": self._list_deployments,
            "scale": self._scale,
            "rollback": self._rollback,
            "destroy": self._destroy,
            "get_deploy_history": self._get_deploy_history,
            "generate_dockerfile": self._generate_dockerfile,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}. Available: {list(actions.keys())}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Deployment error: {e}")

    async def _generate_config(self, params: Dict) -> SkillResult:
        """Generate deployment configuration for a target platform."""
        platform = params.get("platform", "").lower()
        name = params.get("name", "")
        env_vars = params.get("env_vars", {})
        resources = params.get("resources", DEFAULT_RESOURCES.copy())
        port = params.get("port", 8080)
        replicas = params.get("replicas", 1)

        if not name:
            return SkillResult(success=False, message="Deployment name is required")
        if platform not in PLATFORMS:
            return SkillResult(success=False, message=f"Unsupported platform: {platform}. Supported: {PLATFORMS}")
        if replicas < 0 or replicas > 100:
            return SkillResult(success=False, message="Replicas must be between 0 and 100")

        deployment_id = f"{name}-{str(uuid.uuid4())[:8]}"
        config = {}

        if platform == "docker":
            config = self._gen_docker_compose(name, env_vars, resources, port, replicas)
        elif platform == "fly.io":
            config = self._gen_fly_config(name, env_vars, resources, port, replicas)
        elif platform == "railway":
            config = self._gen_railway_config(name, env_vars, resources, port)
        elif platform == "local":
            config = self._gen_local_config(name, env_vars, port)

        deployment = {
            "id": deployment_id,
            "name": name,
            "platform": platform,
            "config": config,
            "env_vars": list(env_vars.keys()),  # Store keys only, not values
            "resources": resources,
            "port": port,
            "replicas": replicas,
            "status": "configured",
            "version": 1,
            "versions": [{"version": 1, "config": config, "created_at": datetime.now().isoformat()}],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        data = _load_deployments()
        data["deployments"][deployment_id] = deployment
        _save_deployments(data)
        _record_event(deployment_id, "config_generated", {"platform": platform})

        return SkillResult(
            success=True,
            message=f"Generated {platform} deployment config '{name}' (ID: {deployment_id})",
            data={"deployment_id": deployment_id, "platform": platform, "config": config},
        )

    def _gen_docker_compose(self, name: str, env_vars: Dict, resources: Dict, port: int, replicas: int) -> Dict:
        """Generate docker-compose configuration."""
        service = {
            "image": f"singularity-agent:{name}",
            "build": {"context": ".", "dockerfile": "Dockerfile"},
            "ports": [f"{port}:{port}"],
            "environment": {k: "${" + k + "}" for k in env_vars},
            "restart": "unless-stopped",
            "deploy": {
                "replicas": replicas,
                "resources": {
                    "limits": {
                        "cpus": resources.get("cpu", "1"),
                        "memory": resources.get("memory", "512M"),
                    }
                },
            },
        }
        return {
            "version": "3.8",
            "services": {name: service},
        }

    def _gen_fly_config(self, name: str, env_vars: Dict, resources: Dict, port: int, replicas: int) -> Dict:
        """Generate fly.io configuration (fly.toml format as dict)."""
        mem_str = resources.get("memory", "512MB")
        mem_mb = int(mem_str.replace("MB", "").replace("GB", "000").replace("M", "").replace("G", "000"))
        return {
            "app": name,
            "primary_region": "iad",
            "build": {"dockerfile": "Dockerfile"},
            "env": {k: "" for k in env_vars},  # Placeholder values
            "http_service": {
                "internal_port": port,
                "force_https": True,
                "auto_stop_machines": True,
                "auto_start_machines": True,
                "min_machines_running": replicas,
            },
            "vm": {
                "cpu_kind": "shared",
                "cpus": int(resources.get("cpu", "1")),
                "memory_mb": mem_mb,
            },
        }

    def _gen_railway_config(self, name: str, env_vars: Dict, resources: Dict, port: int) -> Dict:
        """Generate Railway configuration."""
        return {
            "name": name,
            "build": {"builder": "DOCKERFILE", "dockerfilePath": "Dockerfile"},
            "deploy": {
                "startCommand": f"python -m singularity --port {port}",
                "healthcheckPath": "/health",
                "restartPolicyType": "ON_FAILURE",
            },
            "env": {k: "" for k in env_vars},
        }

    def _gen_local_config(self, name: str, env_vars: Dict, port: int) -> Dict:
        """Generate local deployment configuration."""
        return {
            "name": name,
            "type": "local",
            "command": f"python -m singularity --port {port}",
            "env": {k: "" for k in env_vars},
            "pid_file": f"/tmp/singularity-{name}.pid",
        }

    async def _deploy(self, params: Dict) -> SkillResult:
        """Deploy or update a deployment."""
        deployment_id = params.get("deployment_id", "")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        deployment = data["deployments"].get(deployment_id)
        if not deployment:
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' not found")

        platform = deployment["platform"]
        name = deployment["name"]

        # Check if platform CLI is available
        cli_check = self._check_platform_cli(platform)
        if not cli_check["available"]:
            # Still record the deploy attempt and update status
            deployment["status"] = "pending_cli"
            deployment["updated_at"] = datetime.now().isoformat()
            deployment["last_deploy_note"] = f"Platform CLI '{cli_check['cli']}' not found. Install it to proceed."
            data["deployments"][deployment_id] = deployment
            _save_deployments(data)
            _record_event(deployment_id, "deploy_pending", {"reason": f"CLI '{cli_check['cli']}' not available"})
            return SkillResult(
                success=False,
                message=f"Cannot deploy: {cli_check['cli']} CLI not found. Install it first.",
                data={"deployment_id": deployment_id, "cli_needed": cli_check["cli"], "install_hint": cli_check.get("install_hint", "")},
            )

        # Execute platform-specific deployment
        deploy_result = await self._execute_deploy(platform, name, deployment)

        if deploy_result["success"]:
            deployment["status"] = "deployed"
            deployment["last_deployed_at"] = datetime.now().isoformat()
            deployment["deploy_output"] = deploy_result.get("output", "")[:2000]
        else:
            deployment["status"] = "deploy_failed"
            deployment["last_error"] = deploy_result.get("error", "unknown")[:2000]

        deployment["updated_at"] = datetime.now().isoformat()
        data["deployments"][deployment_id] = deployment
        _save_deployments(data)
        _record_event(deployment_id, "deploy_attempted", {"success": deploy_result["success"]})

        return SkillResult(
            success=deploy_result["success"],
            message=deploy_result.get("message", "Deploy completed"),
            data={"deployment_id": deployment_id, "status": deployment["status"]},
            cost=0.01,
        )

    def _check_platform_cli(self, platform: str) -> Dict:
        """Check if the required CLI tool is available."""
        cli_map = {
            "docker": {"cli": "docker", "install_hint": "Install Docker: https://docs.docker.com/get-docker/"},
            "fly.io": {"cli": "flyctl", "install_hint": "Install flyctl: curl -L https://fly.io/install.sh | sh"},
            "railway": {"cli": "railway", "install_hint": "Install Railway CLI: npm i -g @railway/cli"},
            "local": {"cli": "python", "install_hint": "Python should be available"},
        }
        info = cli_map.get(platform, {"cli": "unknown", "install_hint": ""})
        try:
            result = subprocess.run(
                [info["cli"], "--version"],
                capture_output=True, text=True, timeout=10
            )
            info["available"] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info["available"] = False
        return info

    async def _execute_deploy(self, platform: str, name: str, deployment: Dict) -> Dict:
        """Execute platform-specific deployment commands."""
        try:
            if platform == "docker":
                return self._deploy_docker(name, deployment)
            elif platform == "fly.io":
                return self._deploy_fly(name, deployment)
            elif platform == "railway":
                return self._deploy_railway(name, deployment)
            elif platform == "local":
                return self._deploy_local(name, deployment)
            else:
                return {"success": False, "error": f"Unsupported platform: {platform}"}
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Deploy failed: {e}"}

    def _deploy_docker(self, name: str, deployment: Dict) -> Dict:
        """Deploy using Docker."""
        try:
            # Build the image
            build_result = subprocess.run(
                ["docker", "build", "-t", f"singularity-agent:{name}", "."],
                capture_output=True, text=True, timeout=300
            )
            if build_result.returncode != 0:
                return {"success": False, "error": build_result.stderr, "message": "Docker build failed"}

            # Run the container
            port = deployment.get("port", 8080)
            run_cmd = [
                "docker", "run", "-d",
                "--name", f"singularity-{name}",
                "-p", f"{port}:{port}",
                "--restart", "unless-stopped",
                f"singularity-agent:{name}",
            ]
            run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60)
            if run_result.returncode != 0:
                return {"success": False, "error": run_result.stderr, "message": "Docker run failed"}

            return {"success": True, "output": run_result.stdout, "message": f"Docker container deployed: singularity-{name}"}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout", "message": "Docker deploy timed out"}

    def _deploy_fly(self, name: str, deployment: Dict) -> Dict:
        """Deploy to fly.io."""
        try:
            result = subprocess.run(
                ["flyctl", "deploy", "--app", name, "--now"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                return {"success": False, "error": result.stderr, "message": "Fly.io deploy failed"}
            return {"success": True, "output": result.stdout, "message": f"Deployed to fly.io: {name}"}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout", "message": "Fly.io deploy timed out"}

    def _deploy_railway(self, name: str, deployment: Dict) -> Dict:
        """Deploy to Railway."""
        try:
            result = subprocess.run(
                ["railway", "up", "--detach"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                return {"success": False, "error": result.stderr, "message": "Railway deploy failed"}
            return {"success": True, "output": result.stdout, "message": f"Deployed to Railway: {name}"}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout", "message": "Railway deploy timed out"}

    def _deploy_local(self, name: str, deployment: Dict) -> Dict:
        """Deploy locally as a background process."""
        config = deployment.get("config", {})
        cmd = config.get("command", "python -m singularity")
        pid_file = config.get("pid_file", f"/tmp/singularity-{name}.pid")

        try:
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Save PID
            Path(pid_file).write_text(str(process.pid))
            return {
                "success": True,
                "output": f"PID: {process.pid}",
                "message": f"Local process started: PID {process.pid}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": f"Local deploy failed: {e}"}

    async def _status(self, params: Dict) -> SkillResult:
        """Check deployment status."""
        deployment_id = params.get("deployment_id", "")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        deployment = data["deployments"].get(deployment_id)
        if not deployment:
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' not found")

        # Try to get live status
        live_status = self._get_live_status(deployment)

        status_info = {
            "deployment_id": deployment_id,
            "name": deployment["name"],
            "platform": deployment["platform"],
            "status": deployment["status"],
            "version": deployment.get("version", 1),
            "replicas": deployment.get("replicas", 1),
            "port": deployment.get("port", 8080),
            "created_at": deployment.get("created_at"),
            "updated_at": deployment.get("updated_at"),
            "last_deployed_at": deployment.get("last_deployed_at"),
            "live_status": live_status,
        }

        return SkillResult(
            success=True,
            message=f"Deployment '{deployment['name']}' is {deployment['status']}",
            data=status_info,
        )

    def _get_live_status(self, deployment: Dict) -> Dict:
        """Check live status of a deployment."""
        platform = deployment["platform"]
        name = deployment["name"]

        try:
            if platform == "docker":
                result = subprocess.run(
                    ["docker", "inspect", f"singularity-{name}", "--format", "{{.State.Status}}"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return {"running": True, "container_status": result.stdout.strip()}
                return {"running": False, "reason": "Container not found"}

            elif platform == "fly.io":
                result = subprocess.run(
                    ["flyctl", "status", "--app", name, "--json"],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    try:
                        return {"running": True, "details": json.loads(result.stdout)}
                    except json.JSONDecodeError:
                        return {"running": True, "output": result.stdout[:500]}
                return {"running": False, "reason": result.stderr[:200]}

            elif platform == "local":
                pid_file = deployment.get("config", {}).get("pid_file", "")
                if pid_file and Path(pid_file).exists():
                    pid = int(Path(pid_file).read_text().strip())
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        return {"running": True, "pid": pid}
                    except (OSError, ProcessLookupError):
                        return {"running": False, "reason": "Process not running"}
                return {"running": False, "reason": "No PID file"}

        except Exception as e:
            return {"running": False, "error": str(e)}

        return {"running": False, "reason": "Unknown platform"}

    async def _list_deployments(self, params: Dict) -> SkillResult:
        """List all deployments."""
        data = _load_deployments()
        deployments = data.get("deployments", {})

        summary = []
        for dep_id, dep in deployments.items():
            summary.append({
                "id": dep_id,
                "name": dep["name"],
                "platform": dep["platform"],
                "status": dep["status"],
                "version": dep.get("version", 1),
                "replicas": dep.get("replicas", 1),
                "created_at": dep.get("created_at"),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(summary)} deployment(s)",
            data={"deployments": summary, "total": len(summary)},
        )

    async def _scale(self, params: Dict) -> SkillResult:
        """Scale a deployment."""
        deployment_id = params.get("deployment_id", "")
        target_replicas = params.get("replicas")

        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")
        if target_replicas is None or not isinstance(target_replicas, int):
            return SkillResult(success=False, message="replicas (int) is required")
        if target_replicas < 0 or target_replicas > 100:
            return SkillResult(success=False, message="Replicas must be between 0 and 100")

        data = _load_deployments()
        deployment = data["deployments"].get(deployment_id)
        if not deployment:
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' not found")

        old_replicas = deployment.get("replicas", 1)
        deployment["replicas"] = target_replicas
        deployment["updated_at"] = datetime.now().isoformat()

        # Platform-specific scaling
        platform = deployment["platform"]
        scale_result = {"success": True, "message": "Scale recorded"}

        if platform == "fly.io" and deployment["status"] == "deployed":
            try:
                result = subprocess.run(
                    ["flyctl", "scale", "count", str(target_replicas), "--app", deployment["name"]],
                    capture_output=True, text=True, timeout=30
                )
                scale_result = {
                    "success": result.returncode == 0,
                    "message": result.stdout if result.returncode == 0 else result.stderr,
                }
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                scale_result = {"success": False, "message": str(e)}

        data["deployments"][deployment_id] = deployment
        _save_deployments(data)
        _record_event(deployment_id, "scaled", {"from": old_replicas, "to": target_replicas})

        return SkillResult(
            success=True,
            message=f"Scaled '{deployment['name']}' from {old_replicas} to {target_replicas} replicas",
            data={
                "deployment_id": deployment_id,
                "old_replicas": old_replicas,
                "new_replicas": target_replicas,
                "platform_result": scale_result,
            },
            cost=0.005,
        )

    async def _rollback(self, params: Dict) -> SkillResult:
        """Rollback a deployment to a previous version."""
        deployment_id = params.get("deployment_id", "")
        target_version = params.get("version")

        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        deployment = data["deployments"].get(deployment_id)
        if not deployment:
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' not found")

        versions = deployment.get("versions", [])
        if len(versions) < 2:
            return SkillResult(success=False, message="No previous version available for rollback")

        current_version = deployment.get("version", 1)

        if target_version is not None:
            # Find specific version
            target = None
            for v in versions:
                if v["version"] == target_version:
                    target = v
                    break
            if not target:
                available = [v["version"] for v in versions]
                return SkillResult(success=False, message=f"Version {target_version} not found. Available: {available}")
        else:
            # Rollback to previous version
            target = versions[-2]
            target_version = target["version"]

        # Apply the rollback
        deployment["config"] = target["config"]
        new_version = current_version + 1
        deployment["version"] = new_version
        deployment["versions"].append({
            "version": new_version,
            "config": target["config"],
            "created_at": datetime.now().isoformat(),
            "rollback_from": current_version,
            "rollback_to": target_version,
        })
        deployment["status"] = "rollback_pending"
        deployment["updated_at"] = datetime.now().isoformat()

        data["deployments"][deployment_id] = deployment
        _save_deployments(data)
        _record_event(deployment_id, "rollback", {"from_version": current_version, "to_version": target_version})

        return SkillResult(
            success=True,
            message=f"Rolled back '{deployment['name']}' from v{current_version} to config from v{target_version} (now v{new_version}). Re-deploy to apply.",
            data={
                "deployment_id": deployment_id,
                "old_version": current_version,
                "new_version": new_version,
                "rollback_to_config_from": target_version,
            },
            cost=0.01,
        )

    async def _destroy(self, params: Dict) -> SkillResult:
        """Destroy a deployment."""
        deployment_id = params.get("deployment_id", "")
        if not deployment_id:
            return SkillResult(success=False, message="deployment_id is required")

        data = _load_deployments()
        deployment = data["deployments"].get(deployment_id)
        if not deployment:
            return SkillResult(success=False, message=f"Deployment '{deployment_id}' not found")

        name = deployment["name"]
        platform = deployment["platform"]

        # Platform-specific cleanup
        cleanup_msg = ""
        if platform == "docker" and deployment["status"] == "deployed":
            try:
                subprocess.run(
                    ["docker", "rm", "-f", f"singularity-{name}"],
                    capture_output=True, text=True, timeout=30
                )
                cleanup_msg = "Docker container removed. "
            except (FileNotFoundError, subprocess.TimeoutExpired):
                cleanup_msg = "Docker cleanup skipped (CLI unavailable). "
        elif platform == "local":
            pid_file = deployment.get("config", {}).get("pid_file", "")
            if pid_file and Path(pid_file).exists():
                try:
                    pid = int(Path(pid_file).read_text().strip())
                    os.kill(pid, 15)  # SIGTERM
                    Path(pid_file).unlink(missing_ok=True)
                    cleanup_msg = f"Local process {pid} terminated. "
                except (OSError, ValueError):
                    cleanup_msg = "Local process cleanup failed. "

        _record_event(deployment_id, "destroyed", {"platform": platform})
        del data["deployments"][deployment_id]
        _save_deployments(data)

        return SkillResult(
            success=True,
            message=f"{cleanup_msg}Deployment '{name}' ({deployment_id}) destroyed",
            data={"deployment_id": deployment_id, "name": name},
        )

    async def _get_deploy_history(self, params: Dict) -> SkillResult:
        """Get deployment history."""
        deployment_id = params.get("deployment_id")
        limit = params.get("limit", 20)

        history = _load_history()

        if deployment_id:
            history = [e for e in history if e.get("deployment_id") == deployment_id]

        # Return most recent events
        events = history[-limit:]

        return SkillResult(
            success=True,
            message=f"Found {len(events)} deployment event(s)",
            data={"events": events, "total_filtered": len(history)},
        )

    async def _generate_dockerfile(self, params: Dict) -> SkillResult:
        """Generate an optimized Dockerfile for the agent."""
        base_image = params.get("base_image", "python:3.11-slim")
        include_gpu = params.get("include_gpu", False)

        if include_gpu:
            base_image = "nvidia/cuda:12.1.0-runtime-ubuntu22.04"

        dockerfile_lines = [
            f"FROM {base_image}",
            "",
            "# System dependencies",
            "RUN apt-get update && apt-get install -y --no-install-recommends \\",
            "    git curl && \\",
            "    rm -rf /var/lib/apt/lists/*",
            "",
            "# Set working directory",
            "WORKDIR /app",
            "",
            "# Install Python dependencies first for better caching",
            "COPY pyproject.toml .",
            "RUN pip install --no-cache-dir -e '.[all]' || pip install --no-cache-dir -e .",
            "",
            "# Copy application code",
            "COPY . .",
            "",
            "# Create data directory",
            "RUN mkdir -p /app/singularity/data",
            "",
            "# Health check",
            'HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8080/health || exit 1',
            "",
            "# Default port",
            "EXPOSE 8080",
            "",
            "# Entry point",
            'ENTRYPOINT ["python", "-m", "singularity"]',
        ]

        if include_gpu:
            # Add GPU-specific setup after system dependencies
            gpu_lines = [
                "",
                "# GPU support",
                "RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121",
            ]
            # Insert after system deps
            idx = dockerfile_lines.index("# Set working directory")
            for i, line in enumerate(gpu_lines):
                dockerfile_lines.insert(idx + i, line)

        dockerfile_content = "\n".join(dockerfile_lines) + "\n"

        return SkillResult(
            success=True,
            message=f"Generated Dockerfile (base: {base_image}, GPU: {include_gpu})",
            data={
                "dockerfile": dockerfile_content,
                "base_image": base_image,
                "include_gpu": include_gpu,
                "line_count": len(dockerfile_lines),
            },
        )
