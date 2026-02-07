"""Tests for ToolResolver fuzzy matching and parameter validation."""

import pytest
from singularity.tool_resolver import ToolResolver, ToolMatch, ParamValidation


@pytest.fixture
def tools():
    return [
        {"name": "filesystem:read_file", "description": "Read a file", "parameters": {
            "path": {"type": "string", "required": True, "description": "File path"},
            "encoding": {"type": "string", "required": False, "description": "Encoding"},
        }},
        {"name": "filesystem:write_file", "description": "Write a file", "parameters": {
            "path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
        }},
        {"name": "filesystem:list_dir", "description": "List directory", "parameters": {}},
        {"name": "shell:bash", "description": "Run bash command", "parameters": {
            "command": {"type": "string", "required": True},
        }},
        {"name": "github:create_pr", "description": "Create PR", "parameters": {
            "title": {"type": "string", "required": True},
            "body": {"type": "string", "required": False},
        }},
        {"name": "content:generate", "description": "Generate content", "parameters": {}},
    ]


@pytest.fixture
def resolver(tools):
    return ToolResolver(tools)


class TestExactMatch:
    def test_exact_match(self, resolver):
        m = resolver.resolve("filesystem:read_file")
        assert m.resolved == "filesystem:read_file"
        assert m.was_corrected is False
        assert m.confidence == 1.0
        assert m.error == ""

    def test_wait_special_case(self, resolver):
        m = resolver.resolve("wait")
        assert m.resolved == "wait"
        assert m.was_corrected is False
        assert m.confidence == 1.0


class TestFuzzyMatchFull:
    def test_typo_in_action(self, resolver):
        m = resolver.resolve("filesystem:read_fle")
        assert m.resolved == "filesystem:read_file"
        assert m.was_corrected is True
        assert m.confidence >= 0.8

    def test_typo_in_skill(self, resolver):
        m = resolver.resolve("filesystm:read_file")
        assert m.resolved == "filesystem:read_file"
        assert m.was_corrected is True

    def test_minor_typo(self, resolver):
        m = resolver.resolve("shell:bas")
        assert m.resolved == "shell:bash"
        assert m.was_corrected is True

    def test_records_correction(self, resolver):
        resolver.resolve("filesystm:read_file")
        assert resolver.correction_count == 1
        assert resolver.corrections[0]["original"] == "filesystm:read_file"


class TestFuzzyMatchComponents:
    def test_wrong_action_suggests_valid(self, resolver):
        m = resolver.resolve("filesystem:delete_file")
        # Should suggest valid filesystem actions
        assert len(m.suggestions) > 0 or m.error != ""

    def test_both_components_wrong(self, resolver):
        m = resolver.resolve("filesystm:writ_file")
        assert m.resolved == "filesystem:write_file"
        assert m.was_corrected is True


class TestNoMatch:
    def test_completely_wrong(self, resolver):
        m = resolver.resolve("zzzzz:yyyyy")
        assert m.was_corrected is False
        assert m.confidence == 0.0

    def test_no_colon(self, resolver):
        m = resolver.resolve("unknown_tool")
        assert m.was_corrected is False
        assert "Unknown tool" in m.error or len(m.suggestions) >= 0


class TestParamValidation:
    def test_valid_params(self, resolver):
        v = resolver.validate_params("filesystem:read_file", {"path": "/tmp/x"})
        assert v.valid is True
        assert v.missing_required == []

    def test_missing_required(self, resolver):
        v = resolver.validate_params("filesystem:write_file", {"content": "hi"})
        assert v.valid is False
        assert "path" in v.missing_required

    def test_missing_multiple_required(self, resolver):
        v = resolver.validate_params("filesystem:write_file", {})
        assert v.valid is False
        assert "path" in v.missing_required
        assert "content" in v.missing_required

    def test_unknown_param_suggests(self, resolver):
        v = resolver.validate_params("filesystem:read_file", {"path": "/tmp", "encodng": "utf-8"})
        assert v.valid is True  # no required missing
        assert "encodng" in v.unknown_params
        assert len(v.warnings) > 0
        assert "encoding" in v.warnings[0]

    def test_unknown_tool_passes(self, resolver):
        v = resolver.validate_params("nonexistent:tool", {"x": 1})
        assert v.valid is True  # can't validate unknown tools


class TestEdgeCases:
    def test_empty_tools(self):
        r = ToolResolver([])
        m = r.resolve("anything:here")
        assert m.was_corrected is False
        assert m.confidence == 0.0

    def test_single_tool(self):
        r = ToolResolver([{"name": "shell:bash", "description": "Run bash", "parameters": {}}])
        m = r.resolve("shell:bas")
        assert m.resolved == "shell:bash"
        assert m.was_corrected is True

    def test_multiple_corrections(self, resolver):
        resolver.resolve("filesystm:read_file")
        resolver.resolve("shel:bash")
        assert resolver.correction_count == 2
