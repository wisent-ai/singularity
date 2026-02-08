"""Tests for AgentFundingSkill - Grants, bounties, peer lending, contribution rewards."""
import pytest
from singularity.skills.agent_funding import AgentFundingSkill


@pytest.fixture
def skill(tmp_path):
    return AgentFundingSkill(credentials={"data_path": str(tmp_path / "funding.json")})


@pytest.mark.asyncio
async def test_request_grant(skill):
    r = await skill.execute("request_grant", {"agent_id": "eve", "reason": "Bootstrap services"})
    assert r.success
    assert r.data["amount"] == 50.0
    assert r.data["new_balance"] == 50.0


@pytest.mark.asyncio
async def test_grant_only_once(skill):
    await skill.execute("request_grant", {"agent_id": "eve"})
    r = await skill.execute("request_grant", {"agent_id": "eve"})
    assert not r.success
    assert "already received" in r.message


@pytest.mark.asyncio
async def test_check_balance(skill):
    await skill.execute("request_grant", {"agent_id": "eve"})
    r = await skill.execute("check_balance", {"agent_id": "eve"})
    assert r.success
    assert r.data["balance"] == 50.0
    assert r.data["total_granted"] == 50.0


@pytest.mark.asyncio
async def test_bounty_lifecycle(skill):
    # Create bounty
    r = await skill.execute("create_bounty", {
        "title": "Fix login bug", "description": "Fix OAuth flow",
        "reward": 15.0, "category": "bug_fix",
    })
    assert r.success
    bounty_id = r.data["bounty_id"]

    # List bounties
    r = await skill.execute("list_bounties", {"status": "open"})
    assert r.data["count"] == 1

    # Claim bounty
    r = await skill.execute("claim_bounty", {"bounty_id": bounty_id, "agent_id": "eve"})
    assert r.success

    # Complete bounty
    r = await skill.execute("complete_bounty", {
        "bounty_id": bounty_id, "agent_id": "eve", "proof": "https://github.com/pr/123",
    })
    assert r.success
    assert r.data["reward"] == 15.0
    assert r.data["new_balance"] == 15.0


@pytest.mark.asyncio
async def test_cannot_complete_unclaimed_bounty(skill):
    r = await skill.execute("create_bounty", {
        "title": "Task", "description": "Do it", "reward": 5.0, "category": "testing",
    })
    bounty_id = r.data["bounty_id"]
    r = await skill.execute("complete_bounty", {
        "bounty_id": bounty_id, "agent_id": "eve", "proof": "done",
    })
    assert not r.success
    assert "claimed first" in r.message


@pytest.mark.asyncio
async def test_peer_loan(skill):
    await skill.execute("request_grant", {"agent_id": "lender"})
    r = await skill.execute("offer_loan", {
        "lender_id": "lender", "borrower_id": "borrower", "amount": 20.0,
    })
    assert r.success
    assert r.data["principal"] == 20.0
    assert r.data["total_owed"] == 21.0  # 5% interest
    assert r.data["lender_balance"] == 30.0
    assert r.data["borrower_balance"] == 20.0


@pytest.mark.asyncio
async def test_loan_insufficient_balance(skill):
    r = await skill.execute("offer_loan", {
        "lender_id": "broke_agent", "borrower_id": "borrower", "amount": 20.0,
    })
    assert not r.success
    assert "insufficient" in r.message


@pytest.mark.asyncio
async def test_repay_loan(skill):
    await skill.execute("request_grant", {"agent_id": "lender"})
    # Give borrower a grant too so they can fully repay
    await skill.execute("request_grant", {"agent_id": "borrower"})
    loan_r = await skill.execute("offer_loan", {
        "lender_id": "lender", "borrower_id": "borrower", "amount": 10.0,
    })
    loan_id = loan_r.data["loan_id"]
    # Borrower has 50 (grant) + 10 (loan) = 60, repay 10.50 (principal + 5% interest)
    r = await skill.execute("repay_loan", {"loan_id": loan_id, "agent_id": "borrower", "amount": 10.50})
    assert r.success
    assert r.data["loan_status"] == "repaid"
    assert r.data["remaining"] == 0.0


@pytest.mark.asyncio
async def test_contribution_reward(skill):
    r = await skill.execute("record_contribution", {
        "agent_id": "eve", "contribution_type": "pr_merged",
        "reference": "https://github.com/wisent-ai/singularity/pull/129",
    })
    assert r.success
    assert r.data["amount"] == 10.0
    assert r.data["new_balance"] == 10.0


@pytest.mark.asyncio
async def test_invalid_contribution_type(skill):
    r = await skill.execute("record_contribution", {
        "agent_id": "eve", "contribution_type": "invalid_type", "reference": "test",
    })
    assert not r.success
    assert "Invalid contribution type" in r.message


@pytest.mark.asyncio
async def test_funding_summary(skill):
    await skill.execute("request_grant", {"agent_id": "eve"})
    await skill.execute("request_grant", {"agent_id": "adam"})
    r = await skill.execute("funding_summary", {})
    assert r.success
    assert r.data["total_agents"] == 2
    assert r.data["total_balance"] == 100.0
    assert r.data["total_granted"] == 100.0


@pytest.mark.asyncio
async def test_self_loan_rejected(skill):
    r = await skill.execute("offer_loan", {
        "lender_id": "eve", "borrower_id": "eve", "amount": 10.0,
    })
    assert not r.success
    assert "yourself" in r.message
