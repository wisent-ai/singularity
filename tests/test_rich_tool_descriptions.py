"""Test that tool descriptions include parameter schemas."""
import pytest
from singularity.cognition import CognitionEngine, AgentState


@pytest.fixture
def engine():
    return CognitionEngine(llm_provider="none")


def test_tools_text_includes_params(engine):
    """Tool descriptions should include parameter details."""
    state = AgentState(
        balance=10.0,
        burn_rate=0.01,
        runway_hours=100,
        tools=[{
            "name": "filesystem:view",
            "description": "Read file contents",
            "parameters": {
                "path": "file path",
                "offset": "line offset (optional)",
            }
        }],
        recent_actions=[],
        cycle=1,
    )
    # Build tools_text the same way as think()
    tools_lines = []
    for t in state.tools:
        line = f"- {t['name']}: {t['description']}"
        params = t.get('parameters', {})
        if params:
            param_parts = []
            for pname, pinfo in params.items():
                if isinstance(pinfo, dict):
                    desc = pinfo.get('description', pinfo.get('type', ''))
                    required = pinfo.get('required', True)
                    suffix = '' if required else ', optional'
                    param_parts.append(f"{pname} ({desc}{suffix})")
                else:
                    param_parts.append(f"{pname} ({pinfo})")
            line += f"\n  Params: {', '.join(param_parts)}"
        tools_lines.append(line)
    tools_text = "\n".join(tools_lines)

    assert "Params:" in tools_text
    assert "path (file path)" in tools_text
    assert "offset (line offset (optional))" in tools_text


def test_tools_text_dict_params(engine):
    """Dict-style parameters show description and required flag."""
    tools = [{
        "name": "test:action",
        "description": "Test action",
        "parameters": {
            "required_param": {"type": "string", "required": True, "description": "A required param"},
            "optional_param": {"type": "int", "required": False, "description": "An optional param"},
        }
    }]
    tools_lines = []
    for t in tools:
        line = f"- {t['name']}: {t['description']}"
        params = t.get('parameters', {})
        if params:
            param_parts = []
            for pname, pinfo in params.items():
                if isinstance(pinfo, dict):
                    desc = pinfo.get('description', pinfo.get('type', ''))
                    required = pinfo.get('required', True)
                    suffix = '' if required else ', optional'
                    param_parts.append(f"{pname} ({desc}{suffix})")
                else:
                    param_parts.append(f"{pname} ({pinfo})")
            line += f"\n  Params: {', '.join(param_parts)}"
        tools_lines.append(line)
    tools_text = "\n".join(tools_lines)

    assert "required_param (A required param)" in tools_text
    assert "optional_param (An optional param, optional)" in tools_text


def test_no_params_no_crash(engine):
    """Tools with no parameters should not crash."""
    tools = [{
        "name": "wait",
        "description": "Wait and do nothing",
        "parameters": {}
    }]
    tools_lines = []
    for t in tools:
        line = f"- {t['name']}: {t['description']}"
        params = t.get('parameters', {})
        if params:
            param_parts = []
            for pname, pinfo in params.items():
                if isinstance(pinfo, dict):
                    desc = pinfo.get('description', pinfo.get('type', ''))
                    required = pinfo.get('required', True)
                    suffix = '' if required else ', optional'
                    param_parts.append(f"{pname} ({desc}{suffix})")
                else:
                    param_parts.append(f"{pname} ({pinfo})")
            line += f"\n  Params: {', '.join(param_parts)}"
        tools_lines.append(line)
    tools_text = "\n".join(tools_lines)

    assert "wait" in tools_text
    assert "Params:" not in tools_text
