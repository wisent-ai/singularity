"""Tests for ServiceCatalogSkill."""

import json
import asyncio
import pytest
from pathlib import Path
from singularity.skills.service_catalog import (
    ServiceCatalogSkill, CATALOG_FILE, BUILTIN_PACKAGES,
)


@pytest.fixture(autouse=True)
def clean_data():
    if CATALOG_FILE.exists():
        CATALOG_FILE.unlink()
    yield
    if CATALOG_FILE.exists():
        CATALOG_FILE.unlink()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_skill():
    return ServiceCatalogSkill()


def test_manifest():
    s = make_skill()
    m = s.manifest
    assert m.skill_id == "service_catalog"
    actions = [a.name for a in m.actions]
    assert "list_packages" in actions
    assert "preview" in actions
    assert "deploy" in actions
    assert "undeploy" in actions
    assert "create_custom" in actions
    assert "compare" in actions
    assert "recommend" in actions
    assert "status" in actions


def test_list_packages():
    s = make_skill()
    r = run(s.execute("list_packages", {}))
    assert r.success
    assert r.data["total_builtin"] == len(BUILTIN_PACKAGES)
    assert len(r.data["packages"]) >= 4
    names = [p["name"] for p in r.data["packages"]]
    assert "Developer Toolkit" in names
    assert "Full Stack Enterprise" in names


def test_preview_builtin():
    s = make_skill()
    r = run(s.execute("preview", {"package_id": "developer_toolkit"}))
    assert r.success
    assert r.data["name"] == "Developer Toolkit"
    assert len(r.data["services"]) == 2
    assert r.data["pricing"]["discount_pct"] == 15.0
    assert r.data["pricing"]["bundle_total"] < r.data["pricing"]["individual_total"]


def test_preview_not_found():
    s = make_skill()
    r = run(s.execute("preview", {"package_id": "nonexistent"}))
    assert not r.success


def test_deploy_and_undeploy():
    s = make_skill()
    r = run(s.execute("deploy", {"package_id": "content_suite", "agent_name": "test-agent"}))
    assert r.success
    dep_id = r.data["deployment_id"]
    assert r.data["services_deployed"] == 2
    assert "test-agent" in r.data["domain"]

    # Status shows active
    st = run(s.execute("status", {}))
    assert st.data["active_count"] == 1

    # Undeploy
    r2 = run(s.execute("undeploy", {"deployment_id": dep_id}))
    assert r2.success
    assert r2.data["services_removed"] == 2

    # Status shows none active
    st2 = run(s.execute("status", {}))
    assert st2.data["active_count"] == 0


def test_deploy_custom_pricing():
    s = make_skill()
    r = run(s.execute("deploy", {
        "package_id": "developer_toolkit",
        "custom_pricing": {"code_review": 0.20},
    }))
    assert r.success
    # The code_review service should use overridden price
    cr_entry = [e for e in r.data["marketplace_entries"] if e["action"] == "code_review"][0]
    # 0.20 * (1 - 0.15) = 0.17
    assert cr_entry["price"] == 0.17


def test_create_custom_package():
    s = make_skill()
    r = run(s.execute("create_custom", {
        "package_id": "my_bundle",
        "name": "My Custom Bundle",
        "description": "A test bundle",
        "services": [
            {"skill": "revenue_services", "action": "code_review", "name": "Review", "price": 0.15},
            {"skill": "revenue_services", "action": "seo_audit", "name": "SEO", "price": 0.08},
        ],
        "bundle_discount": 0.20,
        "tags": ["custom", "test"],
    }))
    assert r.success
    assert r.data["services_count"] == 2

    # Shows up in list
    r2 = run(s.execute("list_packages", {}))
    ids = [p["package_id"] for p in r2.data["packages"]]
    assert "my_bundle" in ids
    assert r2.data["total_custom"] == 1


def test_create_custom_rejects_builtin_id():
    s = make_skill()
    r = run(s.execute("create_custom", {
        "package_id": "developer_toolkit",
        "name": "Conflict",
        "description": "test",
        "services": [{"skill": "x", "action": "y", "name": "z"}],
    }))
    assert not r.success


def test_delete_custom():
    s = make_skill()
    run(s.execute("create_custom", {
        "package_id": "to_delete",
        "name": "Delete Me",
        "description": "test",
        "services": [{"skill": "x", "action": "y", "name": "z"}],
    }))
    r = run(s.execute("delete_custom", {"package_id": "to_delete"}))
    assert r.success

    # Cannot delete builtin
    r2 = run(s.execute("delete_custom", {"package_id": "full_stack"}))
    assert not r2.success


def test_compare():
    s = make_skill()
    r = run(s.execute("compare", {"package_ids": ["developer_toolkit", "content_suite", "full_stack"]}))
    assert r.success
    assert len(r.data["comparisons"]) == 3
    assert r.data["best_value"] is not None
    assert len(r.data["shared_services"]) >= 0


def test_recommend_developer():
    s = make_skill()
    r = run(s.execute("recommend", {"use_case": "developer tools for code quality"}))
    assert r.success
    assert r.data["top_pick"] == "developer_toolkit"


def test_recommend_content():
    s = make_skill()
    r = run(s.execute("recommend", {"use_case": "content marketing and SEO optimization"}))
    assert r.success
    assert r.data["top_pick"] == "content_suite"


def test_recommend_data():
    s = make_skill()
    r = run(s.execute("recommend", {"use_case": "data analytics and reporting"}))
    assert r.success
    assert r.data["top_pick"] == "data_intelligence"


def test_status_empty():
    s = make_skill()
    r = run(s.execute("status", {}))
    assert r.success
    assert r.data["active_count"] == 0
    assert "developer_toolkit" in r.data["builtin_packages"]
