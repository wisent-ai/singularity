"""Tests for RevenueServiceCatalogSkill."""
import pytest
from singularity.skills.revenue_catalog import RevenueServiceCatalogSkill, BUILTIN_PRODUCTS, BUILTIN_BUNDLES


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "revenue_catalog.json"
    return RevenueServiceCatalogSkill(data_path=path)


@pytest.mark.asyncio
async def test_browse_all(catalog):
    r = await catalog.execute("browse", {})
    assert r.success
    assert len(r.data["products"]) == len(BUILTIN_PRODUCTS)


@pytest.mark.asyncio
async def test_browse_by_category(catalog):
    r = await catalog.execute("browse", {"category": "developer_tools"})
    assert r.success
    assert all(p["category"] == "developer_tools" for p in r.data["products"])
    assert len(r.data["products"]) >= 2  # code_review_basic, code_review_pro, api_doc_generator


@pytest.mark.asyncio
async def test_browse_by_tier(catalog):
    r = await catalog.execute("browse", {"tier": "pro"})
    assert r.success
    assert all(p["tier"] == "pro" for p in r.data["products"])


@pytest.mark.asyncio
async def test_browse_by_tag(catalog):
    r = await catalog.execute("browse", {"tag": "security"})
    assert r.success
    assert len(r.data["products"]) >= 1


@pytest.mark.asyncio
async def test_details(catalog):
    r = await catalog.execute("details", {"product_id": "code_review_basic"})
    assert r.success
    assert r.data["name"] == "AI Code Review - Basic"
    assert r.data["pricing"]["price_usd"] == 0.10
    assert r.data["estimated_margin_pct"] > 0


@pytest.mark.asyncio
async def test_details_not_found(catalog):
    r = await catalog.execute("details", {"product_id": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_deploy(catalog):
    r = await catalog.execute("deploy", {"product_id": "text_summarizer"})
    assert r.success
    assert r.data["status"] == "active"
    assert r.data["pricing"]["price_usd"] == 0.05


@pytest.mark.asyncio
async def test_deploy_with_price_override(catalog):
    r = await catalog.execute("deploy", {"product_id": "text_summarizer", "price_override": 0.10})
    assert r.success
    assert r.data["pricing"]["price_usd"] == 0.10


@pytest.mark.asyncio
async def test_deploy_duplicate(catalog):
    await catalog.execute("deploy", {"product_id": "text_summarizer"})
    r = await catalog.execute("deploy", {"product_id": "text_summarizer"})
    assert not r.success
    assert "already deployed" in r.message


@pytest.mark.asyncio
async def test_pause(catalog):
    await catalog.execute("deploy", {"product_id": "seo_optimizer"})
    r = await catalog.execute("pause", {"product_id": "seo_optimizer"})
    assert r.success
    assert r.data["status"] == "paused"


@pytest.mark.asyncio
async def test_pause_not_deployed(catalog):
    r = await catalog.execute("pause", {"product_id": "seo_optimizer"})
    assert not r.success


@pytest.mark.asyncio
async def test_retire(catalog):
    await catalog.execute("deploy", {"product_id": "data_analyzer"})
    r = await catalog.execute("retire", {"product_id": "data_analyzer"})
    assert r.success
    assert r.data["status"] == "retired"


@pytest.mark.asyncio
async def test_reactivate_paused(catalog):
    await catalog.execute("deploy", {"product_id": "text_summarizer"})
    await catalog.execute("pause", {"product_id": "text_summarizer"})
    r = await catalog.execute("deploy", {"product_id": "text_summarizer"})
    assert r.success  # Should reactivate


@pytest.mark.asyncio
async def test_bundles(catalog):
    r = await catalog.execute("bundles", {})
    assert r.success
    assert len(r.data["bundles"]) == len(BUILTIN_BUNDLES)
    for b in r.data["bundles"]:
        assert b["discount_pct"] > 0
        assert b["savings_usd"] > 0


@pytest.mark.asyncio
async def test_deploy_bundle(catalog):
    r = await catalog.execute("deploy_bundle", {"bundle_id": "content_suite"})
    assert r.success
    assert len(r.data["deployed"]) == 2  # text_summarizer + seo_optimizer


@pytest.mark.asyncio
async def test_deploy_bundle_not_found(catalog):
    r = await catalog.execute("deploy_bundle", {"bundle_id": "nonexistent"})
    assert not r.success


@pytest.mark.asyncio
async def test_projections_no_deployments(catalog):
    r = await catalog.execute("projections", {})
    assert r.success
    assert r.data["total_monthly_revenue"] == 0


@pytest.mark.asyncio
async def test_projections_with_deployments(catalog):
    await catalog.execute("deploy", {"product_id": "code_review_basic"})
    await catalog.execute("deploy", {"product_id": "text_summarizer"})
    r = await catalog.execute("projections", {"monthly_requests_per_service": 1000})
    assert r.success
    assert r.data["total_monthly_revenue"] > 0
    assert r.data["total_monthly_profit"] > 0
    assert r.data["active_service_count"] == 2


@pytest.mark.asyncio
async def test_deployments_list(catalog):
    await catalog.execute("deploy", {"product_id": "text_summarizer"})
    await catalog.execute("deploy", {"product_id": "seo_optimizer"})
    r = await catalog.execute("deployments", {"status": "active"})
    assert r.success
    assert len(r.data["deployments"]) == 2


@pytest.mark.asyncio
async def test_create_custom_product(catalog):
    r = await catalog.execute("create_product", {
        "product_id": "custom_analysis",
        "name": "Custom Data Pipeline",
        "description": "Automated data pipeline processing",
        "skill_id": "data_transform",
        "action": "transform",
        "price_usd": 1.00,
    })
    assert r.success
    # Should appear in browse
    browse = await catalog.execute("browse", {})
    assert any(p["product_id"] == "custom_analysis" for p in browse.data["products"])


@pytest.mark.asyncio
async def test_create_duplicate_product(catalog):
    r = await catalog.execute("create_product", {
        "product_id": "code_review_basic",  # Already exists as builtin
        "name": "Duplicate",
        "description": "test",
        "skill_id": "x",
        "action": "y",
        "price_usd": 1.0,
    })
    assert not r.success


@pytest.mark.asyncio
async def test_unknown_action(catalog):
    r = await catalog.execute("nonexistent", {})
    assert not r.success
