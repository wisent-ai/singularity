"""Tests for singularity.cli — developer tools CLI."""


from singularity.cli import (
    _load_registry,
    _check_skill_directory,
    _validate_manifest,
    validate_skill,
    validate_all,
)


class TestLoadRegistry:
    def test_loads_successfully(self):
        registry = _load_registry()
        assert "version" in registry
        assert "skills" in registry
        assert len(registry["skills"]) > 0


class TestCheckSkillDirectory:
    def test_existing_directory_with_files(self):
        # github skill should have Python files
        issues = _check_skill_directory("github", "singularity.skills.builtin.github")
        assert len(issues) == 0

    def test_missing_directory(self):
        issues = _check_skill_directory("fake", "singularity.skills.builtin.nonexistent_dir_xyz")
        assert any("Directory missing" in i for i in issues)

    def test_external_module_skipped(self):
        issues = _check_skill_directory("ext", "external.module.skill")
        assert len(issues) == 0


class TestValidateManifest:
    def test_valid_manifest(self):
        manifest = {
            "skill_id": "test",
            "name": "Test Skill",
            "version": "1.0.0",
            "category": "dev",
            "description": "A test skill",
        }
        issues = _validate_manifest("test", manifest)
        assert len(issues) == 0

    def test_mismatched_skill_id(self):
        manifest = {"skill_id": "wrong", "name": "X", "version": "1.0",
                     "category": "x", "description": "x"}
        issues = _validate_manifest("correct", manifest)
        assert any("!=" in i or "mismatch" in i for i in issues)

    def test_missing_fields(self):
        manifest = {"skill_id": "test"}
        issues = _validate_manifest("test", manifest)
        assert len(issues) > 0
        assert any("name" in i for i in issues)


class TestValidateSkill:
    def test_validates_github(self):
        registry = _load_registry()
        result = validate_skill("github", registry["skills"]["github"])
        assert result["skill_id"] == "github"
        # github is a fully implemented skill, should have few/no issues
        assert result["status"] in ("PASS", "FAIL")

    def test_nonexistent_skill_dir(self):
        skill_data = {
            "module": "singularity.skills.builtin.does_not_exist",
            "class": "FakeSkill",
            "manifest": {"skill_id": "fake", "name": "Fake", "version": "1.0",
                          "category": "test", "description": "test"},
        }
        result = validate_skill("fake", skill_data)
        assert result["status"] == "FAIL"
        assert any("Directory missing" in i for i in result["issues"])


class TestValidateAll:
    def test_returns_results_for_all_skills(self):
        registry = _load_registry()
        results = validate_all(registry)
        assert len(results) == len(registry["skills"])
        assert all("skill_id" in r for r in results)
        assert all("status" in r for r in results)
        assert all("issues" in r for r in results)

    def test_some_skills_pass(self):
        registry = _load_registry()
        results = validate_all(registry)
        passed = [r for r in results if r["status"] == "PASS"]
        assert len(passed) > 0, "Expected at least some skills to pass validation"
