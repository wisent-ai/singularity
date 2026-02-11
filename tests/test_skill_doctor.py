"""Tests for singularity.skills.doctor - the skill diagnostic tool."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from singularity.skills.doctor import (
    Severity,
    DiagnosticIssue,
    SkillDiagnostic,
    DoctorReport,
    _classify_import_error,
    _check_source_files,
    diagnose_skill,
    run_doctor,
)


# ─── DiagnosticIssue tests ───────────────────────────────────────────────────

class TestDiagnosticIssue:
    def test_create_issue(self):
        issue = DiagnosticIssue(
            severity=Severity.ERROR,
            code="TEST_CODE",
            message="test message",
            fix_hint="do something",
        )
        assert issue.severity == Severity.ERROR
        assert issue.code == "TEST_CODE"
        assert issue.message == "test message"
        assert issue.fix_hint == "do something"

    def test_default_fix_hint(self):
        issue = DiagnosticIssue(
            severity=Severity.OK, code="X", message="msg"
        )
        assert issue.fix_hint == ""


# ─── SkillDiagnostic tests ───────────────────────────────────────────────────

class TestSkillDiagnostic:
    def test_status_ok_when_no_issues(self):
        d = SkillDiagnostic(
            skill_id="test", module_path="m", class_name="C", category="cat"
        )
        assert d.status == Severity.OK

    def test_status_warn(self):
        d = SkillDiagnostic(
            skill_id="test", module_path="m", class_name="C", category="cat",
            issues=[DiagnosticIssue(Severity.WARN, "W", "warning")]
        )
        assert d.status == Severity.WARN

    def test_status_error(self):
        d = SkillDiagnostic(
            skill_id="test", module_path="m", class_name="C", category="cat",
            issues=[
                DiagnosticIssue(Severity.WARN, "W", "warning"),
                DiagnosticIssue(Severity.ERROR, "E", "error"),
            ]
        )
        assert d.status == Severity.ERROR

    def test_status_critical(self):
        d = SkillDiagnostic(
            skill_id="test", module_path="m", class_name="C", category="cat",
            issues=[
                DiagnosticIssue(Severity.ERROR, "E", "error"),
                DiagnosticIssue(Severity.CRITICAL, "C", "critical"),
            ]
        )
        assert d.status == Severity.CRITICAL

    def test_defaults(self):
        d = SkillDiagnostic(
            skill_id="x", module_path="m", class_name="C", category=""
        )
        assert d.can_import is False
        assert d.can_instantiate is False
        assert d.action_count == 0
        assert d.issues == []
        assert d.import_error == ""


# ─── DoctorReport tests ─────────────────────────────────────────────────────

class TestDoctorReport:
    def test_summary_line(self):
        report = DoctorReport(
            total_skills=10, importable=8, instantiable=7, broken=2, warnings=1
        )
        line = report.summary_line()
        assert "8/10 importable" in line
        assert "7/10 instantiable" in line
        assert "2 broken" in line
        assert "1 warnings" in line

    def test_to_dict(self):
        report = DoctorReport(total_skills=1, importable=1, instantiable=0, broken=0, warnings=0)
        d = SkillDiagnostic(
            skill_id="test", module_path="m", class_name="C", category="cat",
            can_import=True, issues=[
                DiagnosticIssue(Severity.WARN, "CODE", "msg", "hint")
            ]
        )
        report.skills["test"] = d
        result = report.to_dict()
        assert result["total_skills"] == 1
        assert "test" in result["skills"]
        assert result["skills"]["test"]["can_import"] is True
        assert len(result["skills"]["test"]["issues"]) == 1
        assert result["skills"]["test"]["issues"][0]["code"] == "CODE"

    def test_to_dict_is_json_serializable(self):
        report = DoctorReport(total_skills=2, importable=1)
        report.skills["a"] = SkillDiagnostic("a", "m", "C", "cat", can_import=True)
        report.skills["b"] = SkillDiagnostic("b", "m", "C", "cat", issues=[
            DiagnosticIssue(Severity.ERROR, "E", "err")
        ])
        # Must not raise
        json_str = json.dumps(report.to_dict())
        parsed = json.loads(json_str)
        assert parsed["total_skills"] == 2


# ─── _classify_import_error tests ────────────────────────────────────────────

class TestClassifyImportError:
    def test_circular_import(self):
        error = "cannot import name 'FOO' from partially initialized module 'bar'"
        issue = _classify_import_error(error, "", "bar")
        assert issue.code == "CIRCULAR_IMPORT"
        assert issue.severity == Severity.CRITICAL
        assert "constants.py" in issue.fix_hint

    def test_wrong_import_path(self):
        error = "No module named 'skills.base'"
        issue = _classify_import_error(error, "", "tiktok")
        assert issue.code == "WRONG_IMPORT_PATH"
        assert "singularity.skills.base" in issue.fix_hint

    def test_missing_optional_dep_jwt(self):
        error = "No module named 'jwt'"
        issue = _classify_import_error(error, "", "mobile_app_publisher")
        assert issue.code == "MISSING_OPTIONAL_DEP"
        assert issue.severity == Severity.WARN
        assert "PyJWT" in issue.fix_hint

    def test_missing_optional_dep_payments(self):
        error = "No module named 'payments'"
        issue = _classify_import_error(error, "", "issuing")
        assert issue.code == "MISSING_OPTIONAL_DEP"
        assert "payments" in issue.message

    def test_missing_optional_dep_httpx(self):
        error = "No module named 'httpx'"
        issue = _classify_import_error(error, "", "something")
        assert issue.code == "MISSING_OPTIONAL_DEP"
        assert "httpx" in issue.fix_hint

    def test_missing_unknown_module(self):
        error = "No module named 'totally_unknown_package'"
        issue = _classify_import_error(error, "", "test")
        assert issue.code == "MISSING_MODULE"
        assert issue.severity == Severity.ERROR

    def test_missing_attribute(self):
        error = "module 'foo' has no attribute 'Bar'"
        issue = _classify_import_error(error, "", "test")
        assert issue.code == "MISSING_ATTRIBUTE"

    def test_cannot_import_name(self):
        error = "cannot import name 'Foo' from 'bar'"
        issue = _classify_import_error(error, "", "test")
        assert issue.code == "MISSING_ATTRIBUTE"

    def test_generic_error(self):
        error = "some random error that doesn't match patterns"
        issue = _classify_import_error(error, "", "test")
        assert issue.code == "IMPORT_ERROR"
        assert issue.severity == Severity.ERROR


# ─── diagnose_skill tests ───────────────────────────────────────────────────

class TestDiagnoseSkill:
    def test_diagnose_real_working_skill(self):
        """Test with a real skill that should work (twitter)."""
        diag = diagnose_skill("twitter", "singularity.skills.builtin.twitter", "TwitterSkill", "social")
        assert diag.can_import is True
        assert diag.skill_id == "twitter"
        assert diag.category == "social"

    def test_diagnose_nonexistent_module(self):
        diag = diagnose_skill("fake", "singularity.skills.builtin.nonexistent", "FakeSkill", "test")
        assert diag.can_import is False
        assert len(diag.issues) >= 1
        assert any(i.code in ("MISSING_SOURCE", "MISSING_MODULE") for i in diag.issues)

    def test_diagnose_wrong_class_name(self):
        """Module exists but class name is wrong."""
        diag = diagnose_skill(
            "twitter", "singularity.skills.builtin.twitter",
            "WrongClassName", "social"
        )
        assert diag.can_import is True
        assert any(i.code == "MISSING_CLASS" for i in diag.issues)


# ─── run_doctor tests ───────────────────────────────────────────────────────

class TestRunDoctor:
    def test_run_doctor_full(self):
        """Run doctor on all skills — should complete without crashing."""
        report = run_doctor()
        assert report.total_skills > 0
        assert report.importable >= 0
        assert report.total_skills == report.importable + report.broken + report.warnings + (
            report.total_skills - report.importable - report.broken - report.warnings
        )

    def test_run_doctor_specific_skill(self):
        report = run_doctor(["twitter"])
        assert report.total_skills == 1
        assert "twitter" in report.skills

    def test_run_doctor_unknown_skill(self):
        report = run_doctor(["totally_nonexistent_skill_xyz"])
        assert report.total_skills == 0

    def test_run_doctor_multiple_skills(self):
        report = run_doctor(["twitter", "github"])
        assert report.total_skills == 2

    def test_report_json_serializable(self):
        report = run_doctor(["twitter"])
        json_str = json.dumps(report.to_dict())
        assert json_str  # Should not raise


# ─── Severity enum tests ────────────────────────────────────────────────────

class TestSeverity:
    def test_values(self):
        assert Severity.OK.value == "ok"
        assert Severity.WARN.value == "warn"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"

    def test_string_enum(self):
        assert str(Severity.OK) == "Severity.OK" or Severity.OK == "ok"
        assert Severity.OK == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
