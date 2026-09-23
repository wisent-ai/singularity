"""Tests for native tool calling support."""
import pytest
from singularity.tool_calling import (
    _param_description_to_schema,
    tools_to_anthropic_format,
    tools_to_openai_format,
    format_tools_for_text_prompt,
    _build_name_map,
)
from singularity.cognition import Action


# -- _param_description_to_schema --

def test_simple_string_description():
    schema, required = _param_description_to_schema("pattern", "glob pattern")
    assert schema["type"] == "string"
    assert schema["description"] == "glob pattern"
    assert required is True


def test_optional_param():
    schema, required = _param_description_to_schema("limit", "max lines (optional)")
    assert required is False


def test_numeric_param_by_name():
    schema, _ = _param_description_to_schema("limit", "max results")
    assert schema["type"] == "number"


def test_boolean_param_by_name():
    schema, _ = _param_description_to_schema("recursive", "recurse into dirs")
    assert schema["type"] == "boolean"


def test_dict_description():
    schema, required = _param_description_to_schema("name", {
        "type": "string", "required": False, "description": "user name"
    })
    assert schema["type"] == "string"
    assert schema["description"] == "user name"
    assert required is False


# -- tools_to_anthropic_format --

SAMPLE_TOOLS = [
    {
        "name": "fs:glob",
        "description": "Find files matching a pattern",
        "parameters": {"pattern": "glob pattern", "path": "base path (optional)"},
    },
    {
        "name": "shell:bash",
        "description": "Run a shell command",
        "parameters": {"command": "shell command to run"},
    },
    {
        "name": "wait",
        "description": "Wait",
        "parameters": {},
    },
]


def test_anthropic_format_count():
    defs = tools_to_anthropic_format(SAMPLE_TOOLS)
    assert len(defs) == 2  # 'wait' is excluded


def test_anthropic_format_structure():
    defs = tools_to_anthropic_format(SAMPLE_TOOLS)
    fs_def = defs[0]
    assert fs_def["name"] == "fs_glob"
    assert fs_def["description"] == "Find files matching a pattern"
    assert "input_schema" in fs_def
    assert fs_def["input_schema"]["type"] == "object"
    props = fs_def["input_schema"]["properties"]
    assert "pattern" in props
    assert "path" in props


def test_anthropic_required_params():
    defs = tools_to_anthropic_format(SAMPLE_TOOLS)
    fs_def = defs[0]
    # 'pattern' is required, 'path' is optional
    assert "pattern" in fs_def["input_schema"].get("required", [])
    assert "path" not in fs_def["input_schema"].get("required", [])


def test_anthropic_shell_tool():
    defs = tools_to_anthropic_format(SAMPLE_TOOLS)
    shell_def = defs[1]
    assert shell_def["name"] == "shell_bash"
    assert "command" in shell_def["input_schema"]["properties"]


# -- tools_to_openai_format --

def test_openai_format_structure():
    defs = tools_to_openai_format(SAMPLE_TOOLS)
    assert len(defs) == 2
    assert defs[0]["type"] == "function"
    assert "function" in defs[0]
    func = defs[0]["function"]
    assert func["name"] == "fs_glob"
    assert "parameters" in func


# -- _build_name_map --

def test_name_map():
    nmap = _build_name_map(SAMPLE_TOOLS)
    assert nmap["fs_glob"] == "fs:glob"
    assert nmap["shell_bash"] == "shell:bash"


# -- format_tools_for_text_prompt --

def test_text_format_includes_params():
    text = format_tools_for_text_prompt(SAMPLE_TOOLS)
    assert "fs:glob" in text
    assert "pattern" in text
    assert "path" in text


def test_text_format_marks_optional():
    text = format_tools_for_text_prompt(SAMPLE_TOOLS)
    assert "path?" in text or "optional" in text.lower()


def test_empty_tools():
    defs = tools_to_anthropic_format([])
    assert defs == []


def test_tool_with_no_params():
    tools = [{"name": "system:status", "description": "Get status", "parameters": {}}]
    defs = tools_to_anthropic_format(tools)
    assert len(defs) == 1
    assert defs[0]["input_schema"]["properties"] == {}
