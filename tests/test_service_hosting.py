#!/usr/bin/env python3
"""Tests for ServiceHostingSkill - HTTP service hosting for agents."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.service_hosting import (
    ServiceHostingSkill, SERVICES_FILE, SERVICE_LOGS_FILE,
)


@pytest.fixture
def skill(tmp_path):
    svc_f = tmp_path / "hosted_services.json"
    logs_f = tmp_path / "service_logs.json"
    with patch("singularity.skills.service_hosting.SERVICES_FILE", svc_f), \
         patch("singularity.skills.service_hosting.SERVICE_LOGS_FILE", logs_f):
        s = ServiceHostingSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


ADAM_ENDPOINTS = [
    {"path": "/code_review", "method": "POST", "price": 0.10, "description": "Code analysis"},
    {"path": "/summarize", "method": "POST", "price": 0.05, "description": "Text summarization"},
    {"path": "/seo_audit", "method": "POST", "price": 0.05, "description": "SEO optimization"},
]


class TestRegisterService:
    def test_register_success(self, skill):
        result = run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "api-services",
            "endpoints": ADAM_ENDPOINTS, "docker_image": "adam-services:latest",
        }))
        assert result.success
        assert result.data["service_id"] == "adam-api-services"
        assert result.data["domain"] == "adam.singularity.wisent.ai"
        assert len(result.data["endpoints"]) == 3

    def test_register_duplicate_fails(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "api",
            "endpoints": [{"path": "/test"}],
        }))
        result = run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "api",
            "endpoints": [{"path": "/test2"}],
        }))
        assert not result.success
        assert "already registered" in result.message

    def test_register_missing_fields(self, skill):
        result = run(skill.execute("register_service", {"agent_name": "X"}))
        assert not result.success

    def test_register_no_endpoints(self, skill):
        result = run(skill.execute("register_service", {
            "agent_name": "X", "service_name": "Y", "endpoints": [],
        }))
        assert not result.success


class TestListAndGetService:
    def test_list_empty(self, skill):
        result = run(skill.execute("list_services", {}))
        assert result.success
        assert result.data["total"] == 0

    def test_list_after_register(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "svc",
            "endpoints": [{"path": "/test"}],
        }))
        result = run(skill.execute("list_services", {}))
        assert result.success
        assert result.data["total"] == 1

    def test_get_service(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "svc",
            "endpoints": [{"path": "/test", "price": 0.10}],
        }))
        result = run(skill.execute("get_service", {"service_id": "adam-svc"}))
        assert result.success
        assert result.data["service"]["agent_name"] == "Adam"


class TestDeregisterService:
    def test_deregister(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Bot", "service_name": "api",
            "endpoints": [{"path": "/run"}],
        }))
        result = run(skill.execute("deregister_service", {"service_id": "bot-api"}))
        assert result.success
        # Verify gone
        result2 = run(skill.execute("get_service", {"service_id": "bot-api"}))
        assert not result2.success


class TestRouteRequest:
    def test_route_success(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "svc",
            "endpoints": [{"path": "/code_review", "method": "POST", "price": 0.10}],
        }))
        result = run(skill.execute("route_request", {
            "service_id": "adam-svc", "path": "/code_review",
            "caller_id": "user-123", "payload": {"code": "print('hi')"},
        }))
        assert result.success
        assert result.data["price_charged"] == 0.10
        assert result.revenue == 0.10

    def test_route_nonexistent_path(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "svc",
            "endpoints": [{"path": "/summarize"}],
        }))
        result = run(skill.execute("route_request", {
            "service_id": "adam-svc", "path": "/nonexistent",
        }))
        assert not result.success


class TestProxyConfig:
    def test_nginx_config(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "api",
            "endpoints": ADAM_ENDPOINTS, "port": 9000,
        }))
        result = run(skill.execute("generate_proxy_config", {"proxy_type": "nginx"}))
        assert result.success
        assert "server_name adam.singularity.wisent.ai" in result.data["config"]
        assert "proxy_pass http://127.0.0.1:9000" in result.data["config"]

    def test_caddy_config(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "Adam", "service_name": "api",
            "endpoints": [{"path": "/test"}],
        }))
        result = run(skill.execute("generate_proxy_config", {"proxy_type": "caddy"}))
        assert result.success
        assert "reverse_proxy" in result.data["config"]


class TestAnalytics:
    def test_analytics_empty(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "A", "service_name": "s",
            "endpoints": [{"path": "/x"}],
        }))
        result = run(skill.execute("service_analytics", {"service_id": "a-s"}))
        assert result.success
        assert result.data["total_requests"] == 0

    def test_analytics_after_requests(self, skill):
        run(skill.execute("register_service", {
            "agent_name": "A", "service_name": "s",
            "endpoints": [{"path": "/x", "price": 0.05}],
        }))
        run(skill.execute("route_request", {"service_id": "a-s", "path": "/x", "caller_id": "c1"}))
        run(skill.execute("route_request", {"service_id": "a-s", "path": "/x", "caller_id": "c2"}))
        result = run(skill.execute("service_analytics", {"service_id": "a-s"}))
        assert result.success
        assert result.data["total_requests"] == 2
        assert result.data["total_revenue"] == 0.10


class TestManifest:
    def test_manifest(self, skill):
        m = skill.manifest
        assert m.skill_id == "service_hosting"
        assert len(m.actions) == 9
        assert m.category == "infrastructure"
