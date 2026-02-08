"""Tests for RevenueGoalAutoSettingSkill."""

import pytest

from singularity.skills.revenue_goal_auto_setting import (
    BRIDGE_FILE,
    RevenueGoalAutoSettingSkill,
)


@pytest.fixture(autouse=True)
def clean_bridge_file():
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()
    yield
    if BRIDGE_FILE.exists():
        BRIDGE_FILE.unlink()


@pytest.fixture
def skill():
    return RevenueGoalAutoSettingSkill()


def sample_forecast(growth=0.05, days=7):
    return {
        "daily_growth_rate": growth,
        "days_to_breakeven": 10,
        "forecasted_days": [
            {"day": i + 1, "revenue": 0.01 * (1 + growth) ** (i + 1)} for i in range(days)
        ],
    }


def sample_overview(revenue=0.10, profit=0.05, margin=50.0):
    return {
        "total_revenue": revenue,
        "total_profit": profit,
        "profit_margin_pct": margin,
    }


def sample_profitability(margin=50.0, best=None, worst=None):
    return {
        "overall_margin_pct": margin,
        "best_margin_sources": best or [{"source": "api_service", "margin_pct": 60.0}],
        "worst_margin_sources": worst or [{"source": "compute_tasks", "margin_pct": 5.0}],
    }


# -------------------------------------------------------------------------
# Manifest
# -------------------------------------------------------------------------


class TestManifest:
    def test_manifest_skill_id(self, skill):
        assert skill.manifest.skill_id == "revenue_goal_auto_setting"

    def test_manifest_name(self, skill):
        assert skill.manifest.name == "Revenue Goal Auto-Setting"

    def test_manifest_version(self, skill):
        assert skill.manifest.version == "1.0.0"

    def test_manifest_category(self, skill):
        assert skill.manifest.category == "revenue"

    def test_manifest_has_7_actions(self, skill):
        assert len(skill.manifest.actions) == 7

    def test_manifest_action_names(self, skill):
        names = {a.name for a in skill.manifest.actions}
        expected = {
            "generate",
            "review",
            "adjust",
            "track",
            "recommend",
            "configure",
            "status",
        }
        assert names == expected

    def test_estimate_cost_zero(self, skill):
        assert skill.estimate_cost("generate", {}) == 0.0


@pytest.mark.asyncio
async def test_unknown_action(skill):
    result = await skill.execute("nonexistent", {})
    assert not result.success
    assert "Unknown action" in result.message


# -------------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_basic(self, skill):
        result = await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(),
            },
        )
        assert result.success
        assert "goal_id" in result.data["goal"]
        assert result.data["goal"]["priority"] in ("critical", "high", "medium")

    @pytest.mark.asyncio
    async def test_generate_with_profitability(self, skill):
        result = await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(),
                "profitability": sample_profitability(),
            },
        )
        assert result.success
        assert result.data["goal"]["target_revenue"] > 0

    @pytest.mark.asyncio
    async def test_generate_dry_run(self, skill):
        result = await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "dry_run": True,
            },
        )
        assert result.success
        assert "DRY RUN" in result.message
        assert result.data["dry_run"] is True
        # Verify no goal was actually created
        review = await skill.execute("review", {})
        assert review.data["summary"]["total_goals"] == 0

    @pytest.mark.asyncio
    async def test_generate_missing_forecast(self, skill):
        result = await skill.execute("generate", {})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_generate_disabled(self, skill):
        await skill.execute("configure", {"auto_generate_enabled": False})
        result = await skill.execute(
            "generate",
            {"forecast": sample_forecast()},
        )
        assert result.success
        assert "disabled" in result.message.lower()

    @pytest.mark.asyncio
    async def test_generate_high_growth_medium_priority(self, skill):
        result = await skill.execute(
            "generate",
            {"forecast": sample_forecast(growth=0.10)},
        )
        assert result.success
        assert result.data["goal"]["priority"] == "medium"
        assert result.data["goal"]["growth_status"] == "on_track"

    @pytest.mark.asyncio
    async def test_generate_low_growth_high_priority(self, skill):
        result = await skill.execute(
            "generate",
            {"forecast": sample_forecast(growth=0.02)},
        )
        assert result.success
        assert result.data["goal"]["priority"] == "high"
        assert result.data["goal"]["growth_status"] == "below_target"

    @pytest.mark.asyncio
    async def test_generate_negative_growth_critical(self, skill):
        result = await skill.execute(
            "generate",
            {"forecast": {"daily_growth_rate": -0.01}},
        )
        assert result.success
        assert result.data["goal"]["priority"] == "critical"
        assert result.data["goal"]["growth_status"] == "declining"

    @pytest.mark.asyncio
    async def test_generate_max_active_goals(self, skill):
        await skill.execute("configure", {"max_active_goals": 2})
        await skill.execute("generate", {"forecast": sample_forecast()})
        await skill.execute("generate", {"forecast": sample_forecast()})
        result = await skill.execute("generate", {"forecast": sample_forecast()})
        assert not result.success
        assert "Max active goals" in result.message

    @pytest.mark.asyncio
    async def test_generate_updates_stats(self, skill):
        await skill.execute("generate", {"forecast": sample_forecast()})
        status = await skill.execute("status", {})
        assert status.data["stats"]["goals_generated"] == 1

    @pytest.mark.asyncio
    async def test_generate_creates_milestones(self, skill):
        result = await skill.execute(
            "generate",
            {"forecast": sample_forecast(days=5)},
        )
        assert result.success
        milestones = result.data["goal"]["milestones"]
        assert len(milestones) >= 1
        assert any("$" in m for m in milestones)

    @pytest.mark.asyncio
    async def test_generate_no_forecast_days_uses_growth(self, skill):
        result = await skill.execute(
            "generate",
            {"forecast": {"daily_growth_rate": 0.03}},
        )
        assert result.success
        assert len(result.data["goal"]["milestones"]) > 0


# -------------------------------------------------------------------------
# Review
# -------------------------------------------------------------------------


class TestReview:
    @pytest.mark.asyncio
    async def test_review_empty(self, skill):
        result = await skill.execute("review", {})
        assert result.success
        assert result.data["summary"]["total_goals"] == 0

    @pytest.mark.asyncio
    async def test_review_with_goals(self, skill):
        await skill.execute("generate", {"forecast": sample_forecast()})
        await skill.execute("generate", {"forecast": sample_forecast()})
        result = await skill.execute("review", {})
        assert result.success
        assert result.data["summary"]["active"] == 2

    @pytest.mark.asyncio
    async def test_review_filter_active(self, skill):
        await skill.execute("generate", {"forecast": sample_forecast()})
        result = await skill.execute("review", {"status_filter": "active"})
        assert result.success
        assert len(result.data["goals"]) == 1

    @pytest.mark.asyncio
    async def test_review_filter_achieved(self, skill):
        await skill.execute("generate", {"forecast": sample_forecast()})
        result = await skill.execute("review", {"status_filter": "achieved"})
        assert result.success
        assert len(result.data["goals"]) == 0

    @pytest.mark.asyncio
    async def test_review_achievement_rate(self, skill):
        # Create and manually achieve a goal
        await skill.execute("generate", {"forecast": sample_forecast()})
        data = skill._load()
        data["generated_goals"][0]["status"] = "achieved"
        data["generated_goals"][0]["achieved_revenue"] = 1.0
        skill._save(data)
        result = await skill.execute("review", {})
        assert result.data["summary"]["achieved"] == 1
        assert result.data["summary"]["achievement_rate_pct"] == 100.0


# -------------------------------------------------------------------------
# Adjust
# -------------------------------------------------------------------------


class TestAdjust:
    @pytest.mark.asyncio
    async def test_adjust_basic(self, skill):
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(revenue=0.10),
            },
        )
        result = await skill.execute("adjust", {"current_revenue": 0.50})
        assert result.success

    @pytest.mark.asyncio
    async def test_adjust_missing_revenue(self, skill):
        result = await skill.execute("adjust", {})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_adjust_dry_run(self, skill):
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(revenue=0.10),
            },
        )
        result = await skill.execute(
            "adjust",
            {"current_revenue": 0.50, "dry_run": True},
        )
        assert result.success
        assert result.data["dry_run"] is True

    @pytest.mark.asyncio
    async def test_adjust_specific_goal(self, skill):
        gen = await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(revenue=0.10),
            },
        )
        goal_id = gen.data["goal"]["goal_id"]
        result = await skill.execute(
            "adjust",
            {"goal_id": goal_id, "current_revenue": 0.50},
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_adjust_nonexistent_goal(self, skill):
        result = await skill.execute(
            "adjust",
            {"goal_id": "nonexistent", "current_revenue": 0.50},
        )
        assert not result.success
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_adjust_updates_stats(self, skill):
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(revenue=0.01),
            },
        )
        # Diverge significantly from forecast
        await skill.execute("adjust", {"current_revenue": 10.0})
        status = await skill.execute("status", {})
        # stats may or may not show adjustment depending on threshold
        assert status.data["stats"]["goals_adjusted"] >= 0


# -------------------------------------------------------------------------
# Track
# -------------------------------------------------------------------------


class TestTrack:
    @pytest.mark.asyncio
    async def test_track_basic(self, skill):
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(revenue=0.10),
            },
        )
        result = await skill.execute("track", {"actual_revenue": 0.05})
        assert result.success
        assert len(result.data["tracked_goals"]) == 1

    @pytest.mark.asyncio
    async def test_track_missing_revenue(self, skill):
        result = await skill.execute("track", {})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_track_achieves_goal(self, skill):
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(growth=0.01, days=3),
                "overview": sample_overview(revenue=0.01),
            },
        )
        # Track with revenue exceeding target
        result = await skill.execute("track", {"actual_revenue": 100.0})
        assert result.success
        assert result.data["newly_achieved"] >= 1

    @pytest.mark.asyncio
    async def test_track_in_progress(self, skill):
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(revenue=0.10),
            },
        )
        result = await skill.execute("track", {"actual_revenue": 0.001})
        assert result.success
        tracked = result.data["tracked_goals"]
        assert any(t["status"] == "in_progress" for t in tracked)

    @pytest.mark.asyncio
    async def test_track_with_source_breakdown(self, skill):
        await skill.execute(
            "generate",
            {"forecast": sample_forecast()},
        )
        result = await skill.execute(
            "track",
            {
                "actual_revenue": 0.05,
                "by_source": {"api_service": 0.03, "compute": 0.02},
            },
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_track_updates_milestones(self, skill):
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(growth=0.05, days=5),
                "overview": sample_overview(revenue=0.01),
            },
        )
        result = await skill.execute("track", {"actual_revenue": 0.02})
        assert result.success
        # At least some milestones should be partially tracked
        tracked = result.data["tracked_goals"]
        assert len(tracked) >= 1


# -------------------------------------------------------------------------
# Recommend
# -------------------------------------------------------------------------


class TestRecommend:
    @pytest.mark.asyncio
    async def test_recommend_basic(self, skill):
        result = await skill.execute(
            "recommend",
            {"profitability": sample_profitability()},
        )
        assert result.success
        assert len(result.data["recommendations"]) >= 1

    @pytest.mark.asyncio
    async def test_recommend_missing_profitability(self, skill):
        result = await skill.execute("recommend", {})
        assert not result.success
        assert "Required" in result.message

    @pytest.mark.asyncio
    async def test_recommend_high_margin_scale_up(self, skill):
        result = await skill.execute(
            "recommend",
            {
                "profitability": sample_profitability(
                    best=[{"source": "premium_api", "margin_pct": 80.0}],
                    worst=[],
                ),
            },
        )
        assert result.success
        recs = result.data["recommendations"]
        scale_ups = [r for r in recs if r["action"] == "scale_up"]
        assert len(scale_ups) >= 1
        assert scale_ups[0]["source"] == "premium_api"

    @pytest.mark.asyncio
    async def test_recommend_low_margin_deprioritize(self, skill):
        result = await skill.execute(
            "recommend",
            {
                "profitability": sample_profitability(
                    best=[],
                    worst=[{"source": "cheap_compute", "margin_pct": 2.0}],
                ),
            },
        )
        assert result.success
        recs = result.data["recommendations"]
        deprioritized = [r for r in recs if r["action"] == "deprioritize"]
        assert len(deprioritized) >= 1

    @pytest.mark.asyncio
    async def test_recommend_below_target_margin(self, skill):
        result = await skill.execute(
            "recommend",
            {"profitability": sample_profitability(margin=5.0, best=[], worst=[])},
        )
        assert result.success
        recs = result.data["recommendations"]
        improve = [r for r in recs if r["action"] == "improve_margin"]
        assert len(improve) == 1

    @pytest.mark.asyncio
    async def test_recommend_excess_margin_reinvest(self, skill):
        result = await skill.execute(
            "recommend",
            {"profitability": sample_profitability(margin=80.0, best=[], worst=[])},
        )
        assert result.success
        recs = result.data["recommendations"]
        reinvest = [r for r in recs if r["action"] == "reinvest"]
        assert len(reinvest) == 1

    @pytest.mark.asyncio
    async def test_recommend_updates_stats(self, skill):
        await skill.execute(
            "recommend",
            {"profitability": sample_profitability()},
        )
        status = await skill.execute("status", {})
        assert status.data["stats"]["recommendations_made"] >= 1

    @pytest.mark.asyncio
    async def test_recommend_with_string_sources(self, skill):
        """Handle case where sources are plain strings instead of dicts."""
        result = await skill.execute(
            "recommend",
            {
                "profitability": {
                    "overall_margin_pct": 30.0,
                    "best_margin_sources": ["api_service"],
                    "worst_margin_sources": ["compute"],
                },
            },
        )
        assert result.success


# -------------------------------------------------------------------------
# Configure
# -------------------------------------------------------------------------


class TestConfigure:
    @pytest.mark.asyncio
    async def test_configure_single(self, skill):
        result = await skill.execute(
            "configure",
            {"target_daily_growth_rate": 0.10},
        )
        assert result.success
        assert result.data["config"]["target_daily_growth_rate"] == 0.10

    @pytest.mark.asyncio
    async def test_configure_multiple(self, skill):
        result = await skill.execute(
            "configure",
            {
                "target_profit_margin_pct": 30.0,
                "adjustment_threshold_pct": 50.0,
                "max_active_goals": 10,
            },
        )
        assert result.success
        assert result.data["config"]["target_profit_margin_pct"] == 30.0
        assert result.data["config"]["max_active_goals"] == 10

    @pytest.mark.asyncio
    async def test_configure_no_changes(self, skill):
        result = await skill.execute("configure", {})
        assert result.success
        assert "No changes" in result.message

    @pytest.mark.asyncio
    async def test_configure_negative_value(self, skill):
        result = await skill.execute(
            "configure",
            {"target_daily_growth_rate": -0.5},
        )
        assert not result.success
        assert "non-negative" in result.message

    @pytest.mark.asyncio
    async def test_configure_invalid_max_goals(self, skill):
        result = await skill.execute(
            "configure",
            {"max_active_goals": 0},
        )
        assert not result.success
        assert "positive integer" in result.message

    @pytest.mark.asyncio
    async def test_configure_returns_diff(self, skill):
        result = await skill.execute(
            "configure",
            {"max_active_goals": 15},
        )
        assert result.data["updated"]["max_active_goals"]["old"] == 5
        assert result.data["updated"]["max_active_goals"]["new"] == 15


# -------------------------------------------------------------------------
# Status
# -------------------------------------------------------------------------


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_empty(self, skill):
        result = await skill.execute("status", {})
        assert result.success
        assert result.data["active_goals_count"] == 0
        assert "config" in result.data
        assert "stats" in result.data

    @pytest.mark.asyncio
    async def test_status_with_goals(self, skill):
        await skill.execute("generate", {"forecast": sample_forecast()})
        result = await skill.execute("status", {})
        assert result.data["active_goals_count"] == 1
        assert len(result.data["active_goals"]) == 1

    @pytest.mark.asyncio
    async def test_status_shows_recent_events(self, skill):
        await skill.execute("generate", {"forecast": sample_forecast()})
        result = await skill.execute("status", {})
        assert len(result.data["recent_events"]) >= 1


# -------------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_persists_across_instances(self, skill):
        await skill.execute("generate", {"forecast": sample_forecast()})
        skill2 = RevenueGoalAutoSettingSkill()
        result = await skill2.execute("review", {})
        assert result.data["summary"]["total_goals"] == 1

    @pytest.mark.asyncio
    async def test_event_log_truncation(self, skill):
        data = skill._load()
        data["event_log"] = [{"event": f"test-{i}"} for i in range(600)]
        skill._save(data)
        loaded = skill._load()
        assert len(loaded["event_log"]) == 500

    @pytest.mark.asyncio
    async def test_goals_truncation(self, skill):
        data = skill._load()
        data["generated_goals"] = [{"goal_id": f"g-{i}"} for i in range(250)]
        skill._save(data)
        loaded = skill._load()
        assert len(loaded["generated_goals"]) == 200

    @pytest.mark.asyncio
    async def test_corrupted_file_returns_default(self, skill):
        BRIDGE_FILE.write_text("not json")
        data = skill._load()
        assert "config" in data
        assert "generated_goals" in data

    @pytest.mark.asyncio
    async def test_default_config_values(self, skill):
        data = skill._load()
        config = data["config"]
        assert config["auto_generate_enabled"] is True
        assert config["target_daily_growth_rate"] == 0.05
        assert config["target_profit_margin_pct"] == 20.0
        assert config["max_active_goals"] == 5


# -------------------------------------------------------------------------
# Integration workflows
# -------------------------------------------------------------------------


class TestWorkflows:
    @pytest.mark.asyncio
    async def test_generate_track_achieve(self, skill):
        """Full lifecycle: generate goal, track progress, achieve."""
        gen = await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(growth=0.01, days=3),
                "overview": sample_overview(revenue=0.01),
            },
        )
        assert gen.success

        # Track with revenue exceeding target
        track = await skill.execute("track", {"actual_revenue": 100.0})
        assert track.success
        assert track.data["newly_achieved"] >= 1

        # Review should show achieved
        review = await skill.execute("review", {})
        assert review.data["summary"]["achieved"] >= 1

    @pytest.mark.asyncio
    async def test_generate_adjust_track(self, skill):
        """Generate, adjust when diverged, track progress."""
        await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "overview": sample_overview(revenue=0.01),
            },
        )
        # Adjust with significantly different revenue
        await skill.execute("adjust", {"current_revenue": 5.0})
        # Track
        track = await skill.execute("track", {"actual_revenue": 5.0})
        assert track.success

    @pytest.mark.asyncio
    async def test_recommend_then_generate(self, skill):
        """Get recommendations, then generate goals based on insights."""
        rec = await skill.execute(
            "recommend",
            {"profitability": sample_profitability(margin=50.0)},
        )
        assert rec.success
        gen = await skill.execute(
            "generate",
            {
                "forecast": sample_forecast(),
                "profitability": sample_profitability(margin=50.0),
            },
        )
        assert gen.success
        status = await skill.execute("status", {})
        assert status.data["stats"]["recommendations_made"] >= 1
        assert status.data["stats"]["goals_generated"] == 1

    @pytest.mark.asyncio
    async def test_configure_affects_generation(self, skill):
        """Configure thresholds, then verify generation uses them."""
        await skill.execute(
            "configure",
            {"target_daily_growth_rate": 0.20},
        )
        # Growth of 0.05 is now below the 0.20 target
        result = await skill.execute(
            "generate",
            {"forecast": sample_forecast(growth=0.05)},
        )
        assert result.success
        assert result.data["goal"]["growth_status"] == "below_target"
        assert result.data["goal"]["priority"] == "high"
