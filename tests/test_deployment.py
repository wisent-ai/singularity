#!/usr/bin/env python3
"""Tests for DeploymentSkill - cloud deployment management."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

from singularity.skills.deployment import (
    DeploymentSkill, DEPLOY_FILE, DEPLOY_HISTORY_FILE, PLATFORMS,
)


@pytest.fixture
def skill(tmp_path):
    deploy_f = tmp_path / "deployments.json"
    history_f = tmp_path / "deploy_history.json"
    with patch("singularity.skills.deployment.DEPLOY_FILE", deploy_f), \
         patch("singularity.skills.deployment.DEPLOY_HISTORY_FILE", history_f):
        s = DeploymentSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestGenerateConfig:
    def test_docker_config(self, skill):
        result = run(skill.execute("generate_config", {
            "platform": "docker", "name": "test-agent",
            "env_vars": {"API_KEY": "xxx"}, "port": 9000,
        }))
        assert result.success
        assert result.data["platform"] == "docker"
        config = result.data["config"]
        assert "services" in config
        assert "test-agent" in config["services"]

    def test_fly_config(self, skill):
        result = run(skill.execute("generate_config", {
            "platform": "fly.io", "name": "my-fly-app",
        }))
        assert result.success
        assert result.data["config"]["app"] == "my-fly-app"

    def test_railway_config(self, skill):
        result = run(skill.execute("generate_config", {
            "platform": "railway", "name": "rail-app",
        }))
        assert result.success
        assert result.data["config"]["name"] == "rail-app"

    def test_local_config(self, skill):
        result = run(skill.execute("generate_config", {
            "platform": "local", "name": "local-agent",
        }))
        assert result.success
        assert result.data["config"]["type"] == "local"

    def test_invalid_platform(self, skill):
        result = run(skill.execute("generate_config", {
            "platform": "heroku", "name": "x",
        }))
        assert not result.success

    def test_missing_name(self, skill):
        result = run(skill.execute("generate_config", {"platform": "docker"}))
        assert not result.success

    def test_invalid_replicas(self, skill):
        result = run(skill.execute("generate_config", {
            "platform": "docker", "name": "x", "replicas": 999,
        }))
        assert not result.success


class TestListAndStatus:
    def test_list_empty(self, skill):
        result = run(skill.execute("list_deployments", {}))
        assert result.success
        assert result.data["total"] == 0

    def test_list_after_create(self, skill):
        run(skill.execute("generate_config", {"platform": "docker", "name": "a"}))
        run(skill.execute("generate_config", {"platform": "fly.io", "name": "b"}))
        result = run(skill.execute("list_deployments", {}))
        assert result.success
        assert result.data["total"] == 2

    def test_status(self, skill):
        gen = run(skill.execute("generate_config", {"platform": "local", "name": "s"}))
        dep_id = gen.data["deployment_id"]
        result = run(skill.execute("status", {"deployment_id": dep_id}))
        assert result.success
        assert result.data["status"] == "configured"

    def test_status_missing(self, skill):
        result = run(skill.execute("status", {"deployment_id": "nope"}))
        assert not result.success


class TestScaleAndRollback:
    def test_scale(self, skill):
        gen = run(skill.execute("generate_config", {"platform": "docker", "name": "sc"}))
        dep_id = gen.data["deployment_id"]
        result = run(skill.execute("scale", {"deployment_id": dep_id, "replicas": 3}))
        assert result.success
        assert result.data["new_replicas"] == 3

    def test_scale_invalid(self, skill):
        result = run(skill.execute("scale", {"deployment_id": "x", "replicas": 5}))
        assert not result.success

    def test_rollback_no_versions(self, skill):
        gen = run(skill.execute("generate_config", {"platform": "docker", "name": "rb"}))
        dep_id = gen.data["deployment_id"]
        result = run(skill.execute("rollback", {"deployment_id": dep_id}))
        assert not result.success  # Only 1 version


class TestDestroy:
    def test_destroy(self, skill):
        gen = run(skill.execute("generate_config", {"platform": "local", "name": "del"}))
        dep_id = gen.data["deployment_id"]
        result = run(skill.execute("destroy", {"deployment_id": dep_id}))
        assert result.success
        # Verify it's gone
        lst = run(skill.execute("list_deployments", {}))
        assert lst.data["total"] == 0


class TestDockerfile:
    def test_generate_default(self, skill):
        result = run(skill.execute("generate_dockerfile", {}))
        assert result.success
        assert "python:3.11-slim" in result.data["dockerfile"]
        assert "EXPOSE 8080" in result.data["dockerfile"]

    def test_generate_gpu(self, skill):
        result = run(skill.execute("generate_dockerfile", {"include_gpu": True}))
        assert result.success
        assert "nvidia/cuda" in result.data["dockerfile"]

    def test_custom_base(self, skill):
        result = run(skill.execute("generate_dockerfile", {"base_image": "python:3.12"}))
        assert result.success
        assert "python:3.12" in result.data["dockerfile"]


class TestHistory:
    def test_history_recorded(self, skill):
        run(skill.execute("generate_config", {"platform": "docker", "name": "hist"}))
        result = run(skill.execute("get_deploy_history", {}))
        assert result.success
        assert len(result.data["events"]) >= 1

    def test_unknown_action(self, skill):
        result = run(skill.execute("bogus_action", {}))
        assert not result.success
