"""Tests for BillingPipelineSkill."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch
from singularity.skills.billing_pipeline import BillingPipelineSkill, BILLING_FILE


@pytest.fixture(autouse=True)
def isolated_billing_file(tmp_path, monkeypatch):
    """Each test gets its own billing file."""
    bf = tmp_path / "billing.json"
    monkeypatch.setattr("singularity.skills.billing_pipeline.BILLING_FILE", bf)
    return bf


@pytest.fixture
def skill():
    return BillingPipelineSkill()


@pytest.fixture
def skill_with_customer(skill):
    """Skill with one registered customer that has usage records."""
    skill.register_customer("cust-001", "Acme Corp", "basic")
    skill.record_usage("cust-001", "revenue_services", "code_review", cost=0.10)
    skill.record_usage("cust-001", "revenue_services", "code_review", cost=0.10)
    skill.record_usage("cust-001", "revenue_services", "summarize_text", cost=0.05)
    return skill


@pytest.mark.asyncio
async def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "billing_pipeline"
    assert m.category == "revenue"
    assert len(m.actions) == 8


@pytest.mark.asyncio
async def test_register_customer(skill):
    profile = skill.register_customer("c1", "Test User", "premium")
    assert profile["name"] == "Test User"
    assert profile["tier"] == "premium"


@pytest.mark.asyncio
async def test_record_usage(skill):
    skill.register_customer("c1", "Test", "basic")
    skill.record_usage("c1", "revenue_services", "code_review", cost=0.10)
    result = await skill.execute("billing_status", {})
    assert result.success
    assert result.data["active_customers"] == 1


@pytest.mark.asyncio
async def test_bill_customer(skill_with_customer):
    result = await skill_with_customer.execute("bill_customer", {"customer_id": "cust-001"})
    assert result.success
    assert "INV-" in result.data["invoice"]["invoice_id"]
    assert result.data["invoice"]["total"] == 0.25
    assert result.revenue == 0.25


@pytest.mark.asyncio
async def test_run_billing_cycle_dry_run(skill_with_customer):
    result = await skill_with_customer.execute("run_billing_cycle", {"dry_run": True})
    assert result.success
    assert "[DRY RUN]" in result.message
    assert result.data["customers_billed"] == 1
    assert result.data["total_revenue"] == 0.25
    assert result.revenue == 0  # No revenue on dry run


@pytest.mark.asyncio
async def test_run_billing_cycle(skill_with_customer):
    result = await skill_with_customer.execute("run_billing_cycle", {})
    assert result.success
    assert result.data["customers_billed"] == 1
    assert result.revenue == 0.25
    # Usage should be cleared after billing
    status = await skill_with_customer.execute("billing_status", {})
    assert status.data["active_customers"] == 0


@pytest.mark.asyncio
async def test_apply_credit(skill):
    skill.register_customer("c1", "Test", "basic")
    result = await skill.execute("apply_credit", {
        "customer_id": "c1", "amount": 5.00, "reason": "Signup bonus"
    })
    assert result.success
    assert result.data["new_balance"] == 5.00


@pytest.mark.asyncio
async def test_credit_applied_to_invoice(skill):
    skill.register_customer("c1", "Test", "basic")
    await skill.execute("apply_credit", {"customer_id": "c1", "amount": 0.20})
    skill.record_usage("c1", "revenue_services", "code_review", cost=0.25)
    result = await skill.execute("bill_customer", {"customer_id": "c1"})
    assert result.success
    inv = result.data["invoice"]
    assert inv["credits_applied"] == 0.20
    assert inv["total"] == 0.05


@pytest.mark.asyncio
async def test_apply_discount_percentage(skill):
    skill.register_customer("c1", "Test", "basic")
    result = await skill.execute("apply_discount", {
        "customer_id": "c1", "discount_type": "percentage", "value": 20
    })
    assert result.success
    assert "20%" in result.message


@pytest.mark.asyncio
async def test_discount_applied_to_invoice(skill):
    skill.register_customer("c1", "Test", "basic")
    await skill.execute("apply_discount", {
        "customer_id": "c1", "discount_type": "percentage", "value": 50
    })
    skill.record_usage("c1", "revenue_services", "code_review", cost=1.00)
    result = await skill.execute("bill_customer", {"customer_id": "c1"})
    assert result.success
    inv = result.data["invoice"]
    assert inv["discount"] == 0.50
    assert inv["total"] == 0.50


@pytest.mark.asyncio
async def test_configure(skill):
    result = await skill.execute("configure", {
        "billing_period": "weekly", "minimum_charge": 1.00
    })
    assert result.success
    assert result.data["config"]["billing_period"] == "weekly"


@pytest.mark.asyncio
async def test_billing_history_empty(skill):
    result = await skill.execute("billing_history", {})
    assert result.success
    assert result.data["total_cycles"] == 0


@pytest.mark.asyncio
async def test_forecast_no_data(skill):
    result = await skill.execute("forecast", {})
    assert result.success
    assert "low confidence" in result.message


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
