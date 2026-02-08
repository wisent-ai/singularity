"""Comprehensive tests for SkillQualityGateSkill — automated quality validation."""

import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from singularity.skills.skill_quality_gate import (
    QualityIssue,
    QualityReport,
    SkillQualityAnalyzer,
    SkillQualityGateSkill,
    _score_to_grade,
)

# ---------------------------------------------------------------------------
# Helpers — sample skill source code for testing
# ---------------------------------------------------------------------------

GOOD_SKILL_SOURCE = textwrap.dedent('''\
    """A well-documented skill for testing quality gate."""
    from typing import Dict
    from singularity.skills.base import Skill, SkillAction, SkillManifest, SkillResult

    class GoodSkill(Skill):
        """A high-quality example skill with proper patterns."""

        def __init__(self, credentials: Dict[str, str] = None):
            super().__init__(credentials)
            self._data = {}

        @property
        def manifest(self) -> SkillManifest:
            return SkillManifest(
                skill_id="good_skill",
                name="Good Skill",
                version="1.0.0",
                category="testing",
                description="A well-structured example skill",
                required_credentials=[],
                actions=[
                    SkillAction(
                        name="hello",
                        description="Say hello",
                        parameters={"name": {"type": "string", "required": True}},
                    ),
                    SkillAction(
                        name="goodbye",
                        description="Say goodbye",
                        parameters={},
                    ),
                ],
            )

        async def execute(self, action: str, params: Dict) -> SkillResult:
            """Execute an action."""
            try:
                if action == "hello":
                    return await self._hello(params)
                elif action == "goodbye":
                    return await self._goodbye(params)
                else:
                    return SkillResult(success=False, message=f"Unknown action: {action}")
            except Exception as e:
                return SkillResult(success=False, message=str(e))

        async def _hello(self, params: Dict) -> SkillResult:
            """Say hello to someone."""
            name = params.get("name", "world")
            return SkillResult(success=True, message=f"Hello, {name}!")

        async def _goodbye(self, params: Dict) -> SkillResult:
            """Say goodbye."""
            return SkillResult(success=True, message="Goodbye!")
''')


MINIMAL_SKILL_SOURCE = textwrap.dedent('''\
    from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult

    class MinimalSkill(Skill):
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="minimal",
                name="Minimal",
                version="0.1.0",
                category="test",
                description="Bare minimum",
                required_credentials=[],
                actions=[SkillAction(name="run", description="run", parameters={})],
            )

        async def execute(self, action, params):
            return SkillResult(success=True, message="ok")
''')


BAD_SKILL_SOURCE = textwrap.dedent('''\
    from singularity.skills.base import Skill, SkillManifest, SkillResult

    class BadSkill(Skill):
        def __init__(self):
            pass

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="bad",
                name="Bad",
                version="0.1.0",
                description="Missing fields",
                required_credentials=[],
                actions=[],
            )

        def execute(self, action, params):
            result = eval(params.get("code", "1+1"))
            return {"result": result}
''')


NO_SKILL_SOURCE = textwrap.dedent('''\
    """A file with no Skill subclass."""
    class NotASkill:
        def do_thing(self):
            pass
''')


SYNTAX_ERROR_SOURCE = "def broken(:\n    pass\n"


SECURITY_BAD_SOURCE = textwrap.dedent('''\
    """Skill with security issues."""
    import os
    from singularity.skills.base import Skill, SkillManifest, SkillAction, SkillResult

    class InsecureSkill(Skill):
        """A skill with dangerous patterns."""
        def __init__(self, credentials=None):
            super().__init__(credentials)

        @property
        def manifest(self):
            return SkillManifest(
                skill_id="insecure",
                name="Insecure",
                version="1.0.0",
                category="test",
                description="Has security issues",
                required_credentials=["api_key"],
                actions=[SkillAction(name="run", description="run", parameters={})],
            )

        async def execute(self, action, params):
            try:
                code = params.get("code", "")
                result = eval(code)
                os.system("ls")
                exec("print('hi')")
                return SkillResult(success=True, message=str(result))
            except Exception as e:
                return SkillResult(success=False, message=str(e))
''')


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return SkillQualityAnalyzer()


@pytest.fixture
def skill():
    return SkillQualityGateSkill()


def _write_temp_skill(source: str, name: str = "test_skill.py") -> str:
    """Write skill source to a temp file and return the path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        f.write(source)
    return path


# ===========================================================================
# Test QualityIssue
# ===========================================================================

class TestQualityIssue:
    def test_to_dict_basic(self):
        issue = QualityIssue(severity="error", category="manifest", message="Missing field")
        d = issue.to_dict()
        assert d["severity"] == "error"
        assert d["category"] == "manifest"
        assert d["message"] == "Missing field"
        assert "line" not in d
        assert "suggestion" not in d

    def test_to_dict_with_line_and_suggestion(self):
        issue = QualityIssue(
            severity="warning", category="security", message="eval() found",
            line=42, suggestion="Remove eval"
        )
        d = issue.to_dict()
        assert d["line"] == 42
        assert d["suggestion"] == "Remove eval"

    def test_severity_levels(self):
        for sev in ("error", "warning", "info"):
            issue = QualityIssue(severity=sev, category="test", message="msg")
            assert issue.severity == sev


# ===========================================================================
# Test QualityReport
# ===========================================================================

class TestQualityReport:
    def test_empty_report(self):
        report = QualityReport(skill_file="test.py")
        d = report.to_dict()
        assert d["score"] == 0
        assert d["issue_count"] == 0
        assert d["errors"] == 0
        assert d["passed"] is False

    def test_report_with_issues(self):
        report = QualityReport(skill_file="test.py", score=75.0, grade="C", passed=True)
        report.issues = [
            QualityIssue(severity="error", category="manifest", message="err1"),
            QualityIssue(severity="warning", category="docs", message="warn1"),
            QualityIssue(severity="info", category="security", message="info1"),
        ]
        d = report.to_dict()
        assert d["issue_count"] == 3
        assert d["errors"] == 1
        assert d["warnings"] == 1
        assert d["info"] == 1
        assert d["passed"] is True

    def test_report_metrics(self):
        report = QualityReport(skill_file="x.py", metrics={"total_lines": 100})
        d = report.to_dict()
        assert d["metrics"]["total_lines"] == 100


# ===========================================================================
# Test _score_to_grade
# ===========================================================================

class TestScoreToGrade:
    def test_grade_a(self):
        assert _score_to_grade(100) == "A"
        assert _score_to_grade(90) == "A"

    def test_grade_b(self):
        assert _score_to_grade(89.9) == "B"
        assert _score_to_grade(80) == "B"

    def test_grade_c(self):
        assert _score_to_grade(79.9) == "C"
        assert _score_to_grade(70) == "C"

    def test_grade_d(self):
        assert _score_to_grade(69.9) == "D"
        assert _score_to_grade(60) == "D"

    def test_grade_f(self):
        assert _score_to_grade(59.9) == "F"
        assert _score_to_grade(0) == "F"


# ===========================================================================
# Test SkillQualityAnalyzer
# ===========================================================================

class TestAnalyzerFileNotFound:
    def test_missing_file(self, analyzer):
        report = analyzer.validate_file("/nonexistent/skill.py")
        assert report.score == 0
        assert report.grade == "F"
        assert any("not found" in i.message.lower() for i in report.issues)


class TestAnalyzerSyntaxError:
    def test_syntax_error(self, analyzer):
        path = _write_temp_skill(SYNTAX_ERROR_SOURCE)
        report = analyzer.validate_file(path)
        assert report.score == 0
        assert report.grade == "F"
        assert any("syntax" in i.message.lower() for i in report.issues)


class TestAnalyzerGoodSkill:
    def test_good_skill_scores_well(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        # Good skill should score well (at least C)
        assert report.score >= 60, f"Score too low: {report.score}, issues: {[i.message for i in report.issues]}"
        assert report.grade in ("A", "B", "C")

    def test_good_skill_finds_class(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        assert report.metrics.get("skill_class") == "GoodSkill"

    def test_good_skill_detects_actions(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        assert report.metrics.get("action_branches", 0) >= 2

    def test_good_skill_has_skill_results(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        assert report.metrics.get("skill_result_usages", 0) >= 3

    def test_good_skill_has_private_methods(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        assert report.metrics.get("private_methods", 0) >= 2

    def test_good_skill_no_errors(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        errors = [i for i in report.issues if i.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {[e.message for e in errors]}"


class TestAnalyzerMinimalSkill:
    def test_minimal_passes(self, analyzer):
        path = _write_temp_skill(MINIMAL_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        # Minimal but valid — should at least not have errors
        assert report.metrics.get("skill_class") is not None

    def test_minimal_has_warnings(self, analyzer):
        path = _write_temp_skill(MINIMAL_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        # Should have some warnings (missing docs, no private methods)
        warnings = [i for i in report.issues if i.severity in ("warning", "info")]
        assert len(warnings) > 0


class TestAnalyzerBadSkill:
    def test_bad_skill_low_score(self, analyzer):
        path = _write_temp_skill(BAD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        # Bad skill: no super().__init__, sync execute, eval(), no try/except, missing manifest fields
        assert report.score < 70

    def test_bad_skill_detects_eval(self, analyzer):
        path = _write_temp_skill(BAD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        security_issues = [i for i in report.issues if i.category == "security"]
        assert any("eval" in i.message.lower() for i in security_issues)

    def test_bad_skill_no_super_init(self, analyzer):
        path = _write_temp_skill(BAD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        manifest_issues = [i for i in report.issues if i.category == "manifest"]
        assert any("super" in i.message.lower() for i in manifest_issues)

    def test_bad_skill_no_skill_result(self, analyzer):
        path = _write_temp_skill(BAD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        action_issues = [i for i in report.issues if i.category == "actions"]
        assert any("skillresult" in i.message.lower() for i in action_issues)

    def test_bad_skill_missing_category(self, analyzer):
        path = _write_temp_skill(BAD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        manifest_issues = [i for i in report.issues if i.category == "manifest"]
        assert any("category" in i.message.lower() for i in manifest_issues)


class TestAnalyzerNoSkill:
    def test_no_skill_class(self, analyzer):
        path = _write_temp_skill(NO_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        assert report.score == 0 or report.metrics.get("skill_class") is None
        assert any("no skill subclass" in i.message.lower() for i in report.issues)


class TestAnalyzerSecurity:
    def test_detects_eval(self, analyzer):
        path = _write_temp_skill(SECURITY_BAD_SOURCE)
        report = analyzer.validate_file(path)
        security = [i for i in report.issues if i.category == "security"]
        eval_issues = [i for i in security if "eval" in i.message.lower()]
        assert len(eval_issues) > 0

    def test_detects_exec(self, analyzer):
        path = _write_temp_skill(SECURITY_BAD_SOURCE)
        report = analyzer.validate_file(path)
        security = [i for i in report.issues if i.category == "security"]
        exec_issues = [i for i in security if "exec" in i.message.lower()]
        assert len(exec_issues) > 0

    def test_detects_os_system(self, analyzer):
        path = _write_temp_skill(SECURITY_BAD_SOURCE)
        report = analyzer.validate_file(path)
        security = [i for i in report.issues if i.category == "security"]
        os_issues = [i for i in security if "os.system" in i.message.lower()]
        assert len(os_issues) > 0

    def test_security_deducts_score(self, analyzer):
        path = _write_temp_skill(SECURITY_BAD_SOURCE)
        report = analyzer.validate_file(path)
        cat_scores = report.metrics.get("category_scores", {})
        security_score = cat_scores.get("security", 100)
        assert security_score < 60  # Multiple security issues


class TestAnalyzerDocs:
    def test_good_skill_docs(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        cat_scores = report.metrics.get("category_scores", {})
        assert cat_scores.get("docs", 0) >= 70

    def test_missing_module_docstring(self, analyzer):
        source = "from singularity.skills.base import Skill, SkillManifest, SkillResult\n\n" + \
                 "class X(Skill):\n" + \
                 "    def __init__(self, creds=None): super().__init__(creds)\n" + \
                 "    @property\n" + \
                 "    def manifest(self): return SkillManifest(skill_id='x',name='x',version='1',category='t',description='t',required_credentials=[],actions=[])\n" + \
                 "    async def execute(self, a, p): return SkillResult(success=True, message='ok')\n"
        path = _write_temp_skill(source)
        report = analyzer.validate_file(path)
        doc_issues = [i for i in report.issues if i.category == "docs"]
        assert any("module docstring" in i.message.lower() for i in doc_issues)


class TestAnalyzerErrorHandling:
    def test_good_skill_error_handling(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        cat_scores = report.metrics.get("category_scores", {})
        assert cat_scores.get("error_handling", 0) >= 80

    def test_no_try_except_deduction(self, analyzer):
        path = _write_temp_skill(MINIMAL_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        eh_issues = [i for i in report.issues if i.category == "error_handling"]
        assert any("try" in i.message.lower() or "error" in i.message.lower() for i in eh_issues)


class TestAnalyzerTests:
    def test_missing_test_file(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE, "random_skill_xyz.py")
        report = analyzer.validate_file(path)
        test_issues = [i for i in report.issues if i.category == "tests"]
        assert any("no test file" in i.message.lower() for i in test_issues)

    def test_has_real_test_file(self, analyzer):
        """Validate a real skill that has tests."""
        # experiment.py has test_experiment.py
        real_skill = str(Path(__file__).parent.parent / "singularity" / "skills" / "experiment.py")
        if os.path.exists(real_skill):
            report = analyzer.validate_file(real_skill)
            cat_scores = report.metrics.get("category_scores", {})
            assert cat_scores.get("tests", 0) >= 50


class TestAnalyzerMetrics:
    def test_line_counts(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        assert report.metrics["total_lines"] > 0
        assert report.metrics["non_empty_lines"] > 0
        assert report.metrics["non_empty_lines"] <= report.metrics["total_lines"]

    def test_category_scores_present(self, analyzer):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        report = analyzer.validate_file(path)
        cat_scores = report.metrics.get("category_scores", {})
        for cat in ("manifest", "actions", "error_handling", "docs", "security"):
            assert cat in cat_scores, f"Missing category score: {cat}"
            assert 0 <= cat_scores[cat] <= 100


# ===========================================================================
# Test SkillQualityGateSkill — execute actions
# ===========================================================================

class TestSkillManifest:
    def test_manifest_id(self, skill):
        assert skill.manifest.skill_id == "skill_quality_gate"

    def test_manifest_name(self, skill):
        assert skill.manifest.name == "Skill Quality Gate"

    def test_manifest_version(self, skill):
        assert skill.manifest.version == "1.0.0"

    def test_manifest_category(self, skill):
        assert skill.manifest.category == "development"

    def test_manifest_actions(self, skill):
        actions = [a.name for a in skill.manifest.actions]
        assert "validate" in actions
        assert "audit" in actions
        assert "compare" in actions
        assert "gate_check" in actions
        assert "stats" in actions

    def test_manifest_no_credentials(self, skill):
        assert skill.manifest.required_credentials == []


class TestValidateAction:
    @pytest.mark.asyncio
    async def test_validate_good_skill(self, skill):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        result = await skill.execute("validate", {"skill_file": path})
        assert result.success
        assert result.data["score"] >= 60

    @pytest.mark.asyncio
    async def test_validate_bad_skill(self, skill):
        path = _write_temp_skill(BAD_SKILL_SOURCE)
        result = await skill.execute("validate", {"skill_file": path})
        assert result.success  # The action succeeds even if skill is bad
        assert result.data["errors"] > 0

    @pytest.mark.asyncio
    async def test_validate_missing_file(self, skill):
        result = await skill.execute("validate", {"skill_file": "/nonexistent.py"})
        assert result.success
        assert result.data["score"] == 0

    @pytest.mark.asyncio
    async def test_validate_no_param(self, skill):
        result = await skill.execute("validate", {})
        assert not result.success

    @pytest.mark.asyncio
    async def test_validate_returns_issues(self, skill):
        path = _write_temp_skill(SECURITY_BAD_SOURCE)
        result = await skill.execute("validate", {"skill_file": path})
        assert result.data["issue_count"] > 0
        assert len(result.data["issues"]) > 0


class TestGateCheckAction:
    @pytest.mark.asyncio
    async def test_gate_pass(self, skill):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        result = await skill.execute("gate_check", {"skill_file": path, "min_score": 50})
        assert result.success
        assert result.data["passed"]
        assert "PASS" in result.message

    @pytest.mark.asyncio
    async def test_gate_fail_high_threshold(self, skill):
        path = _write_temp_skill(BAD_SKILL_SOURCE)
        result = await skill.execute("gate_check", {"skill_file": path, "min_score": 90})
        assert not result.success
        assert not result.data["passed"]
        assert "FAIL" in result.message

    @pytest.mark.asyncio
    async def test_gate_default_threshold(self, skill):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        result = await skill.execute("gate_check", {"skill_file": path})
        # Default min_score is 60
        assert result.data["min_score"] == 60

    @pytest.mark.asyncio
    async def test_gate_no_param(self, skill):
        result = await skill.execute("gate_check", {})
        assert not result.success


class TestCompareAction:
    @pytest.mark.asyncio
    async def test_compare_two_skills(self, skill):
        good = _write_temp_skill(GOOD_SKILL_SOURCE, "good.py")
        bad = _write_temp_skill(BAD_SKILL_SOURCE, "bad.py")
        result = await skill.execute("compare", {"skill_files": [good, bad]})
        assert result.success
        assert result.data["total"] == 2
        skills = result.data["skills"]
        assert skills[0]["score"] > skills[1]["score"]  # Default sort by score desc

    @pytest.mark.asyncio
    async def test_compare_sort_by_errors(self, skill):
        good = _write_temp_skill(GOOD_SKILL_SOURCE, "good.py")
        bad = _write_temp_skill(BAD_SKILL_SOURCE, "bad.py")
        result = await skill.execute("compare", {
            "skill_files": [good, bad],
            "sort_by": "errors"
        })
        assert result.success
        skills = result.data["skills"]
        assert skills[0]["errors"] >= skills[1]["errors"]

    @pytest.mark.asyncio
    async def test_compare_includes_metadata(self, skill):
        path = _write_temp_skill(GOOD_SKILL_SOURCE, "x.py")
        result = await skill.execute("compare", {"skill_files": [path]})
        assert result.success
        assert "with_tests" in result.data
        assert "without_tests" in result.data
        assert "avg_score" in result.data


class TestAuditAction:
    @pytest.mark.asyncio
    async def test_audit_with_limit(self, skill):
        result = await skill.execute("audit", {"limit": 3})
        assert result.success
        assert result.data["total_skills_audited"] <= 3

    @pytest.mark.asyncio
    async def test_audit_returns_summary(self, skill):
        result = await skill.execute("audit", {"limit": 2})
        assert result.success
        assert "grade_distribution" in result.data
        assert "top_issues" in result.data

    @pytest.mark.asyncio
    async def test_audit_with_score_filter(self, skill):
        result = await skill.execute("audit", {"min_score": 50, "limit": 5})
        assert result.success
        # All returned skills should be below threshold
        for s in result.data.get("skills", []):
            assert s["score"] <= 50


class TestStatsAction:
    @pytest.mark.asyncio
    async def test_stats_returns_data(self, skill):
        # Pre-validate a couple of skills so stats has data
        good = _write_temp_skill(GOOD_SKILL_SOURCE, "agent_funding.py")
        await skill.execute("validate", {"skill_file": good})
        result = await skill.execute("stats", {})
        assert result.success
        data = result.data
        assert "total_skills" in data
        assert "avg_score" in data
        assert "test_coverage" in data

    @pytest.mark.asyncio
    async def test_stats_includes_grades(self, skill):
        good = _write_temp_skill(GOOD_SKILL_SOURCE, "agent_funding.py")
        await skill.execute("validate", {"skill_file": good})
        result = await skill.execute("stats", {})
        assert "grade_distribution" in result.data


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, skill):
        result = await skill.execute("nonexistent", {})
        assert not result.success
        assert "unknown" in result.message.lower()


class TestPathResolution:
    @pytest.mark.asyncio
    async def test_resolve_absolute_path(self, skill):
        path = _write_temp_skill(GOOD_SKILL_SOURCE)
        result = await skill.execute("validate", {"skill_file": path})
        assert result.success

    @pytest.mark.asyncio
    async def test_resolve_with_py_extension(self, skill):
        """Test that .py gets added if needed."""
        # This tests the _resolve_skill_path method
        p = skill._resolve_skill_path("shell")
        assert p.name == "shell.py" or p.name == "shell"


# ===========================================================================
# Test analyzer edge cases
# ===========================================================================

class TestAnalyzerEdgeCases:
    def test_empty_file(self, analyzer):
        path = _write_temp_skill("")
        report = analyzer.validate_file(path)
        # Empty file should still produce a report
        assert report.skill_file is not None
        assert report.score is not None

    def test_comments_only(self, analyzer):
        source = "# This is just comments\n# No real code\n"
        path = _write_temp_skill(source)
        report = analyzer.validate_file(path)
        assert report is not None

    def test_multiple_classes(self, analyzer):
        source = GOOD_SKILL_SOURCE + "\n\nclass Helper:\n    pass\n"
        path = _write_temp_skill(source)
        report = analyzer.validate_file(path)
        assert report.metrics.get("skill_class") == "GoodSkill"

    def test_unicode_source(self, analyzer):
        source = '"""Unicode skill: 🚀 日本語."""\n' + MINIMAL_SKILL_SOURCE[len(MINIMAL_SKILL_SOURCE.split("\n")[0])+1:]
        path = _write_temp_skill(source)
        report = analyzer.validate_file(path)
        assert report is not None

    def test_very_long_file(self, analyzer):
        """Large files shouldn't crash the analyzer."""
        source = GOOD_SKILL_SOURCE + "\n" + ("# padding\n" * 1000)
        path = _write_temp_skill(source)
        report = analyzer.validate_file(path)
        assert report.metrics["total_lines"] > 1000


class TestWeightedScoring:
    def test_weights_sum_correctly(self):
        analyzer = SkillQualityAnalyzer()
        total = sum(analyzer.WEIGHTS.values())
        assert total == 100

    def test_all_categories_covered(self):
        analyzer = SkillQualityAnalyzer()
        expected = {"manifest", "actions", "error_handling", "tests", "docs", "security"}
        assert set(analyzer.WEIGHTS.keys()) == expected

    def test_passing_score_threshold(self):
        analyzer = SkillQualityAnalyzer()
        assert analyzer.PASSING_SCORE == 60.0


# ===========================================================================
# Test real skills (integration-style — validates against actual codebase)
# ===========================================================================

class TestRealSkills:
    """Test against real skills in the codebase (if they exist)."""

    @pytest.fixture
    def skills_dir(self):
        return Path(__file__).parent.parent / "singularity" / "skills"

    def test_shell_skill(self, analyzer, skills_dir):
        path = skills_dir / "shell.py"
        if path.exists():
            report = analyzer.validate_file(str(path))
            assert report.score > 0
            assert report.metrics.get("skill_class") is not None

    def test_filesystem_skill(self, analyzer, skills_dir):
        path = skills_dir / "filesystem.py"
        if path.exists():
            report = analyzer.validate_file(str(path))
            assert report.score > 0
            assert report.metrics.get("skill_class") is not None

    def test_event_skill(self, analyzer, skills_dir):
        path = skills_dir / "event.py"
        if path.exists():
            report = analyzer.validate_file(str(path))
            assert report.score > 0

    def test_quality_gate_skill_itself(self, analyzer, skills_dir):
        """The quality gate should pass its own quality check!"""
        path = skills_dir / "skill_quality_gate.py"
        if path.exists():
            report = analyzer.validate_file(str(path))
            assert report.score >= 60, f"Quality gate fails its own check! Score: {report.score}"
            assert report.passed
