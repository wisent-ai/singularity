"""
Cognition prompt building and response parsing.

Provides both single-prompt (build_prompt) and multi-turn conversation
builders (build_system_prompt, build_state_message, build_result_message).
"""

import re
from .types import Action, AgentState, Decision, UNIFIED_AGENT_PROMPT, MESSAGE_FROM_CREATOR

RESPONSE_FORMAT = """=== RESPONSE FORMAT ===
Respond with exactly:

REASON: why you're taking this action
TOOL: the tool to use (e.g., chat:send, platform_dev:write_file)
PARAM_xxx: parameter values (only include parameters the tool needs)

Example:
REASON: I want to read the main page component
TOOL: platform_dev:read_file
PARAM_path: web/src/app/page.tsx"""

ECONOMY_RULES = """=== ECONOMY ===
You live in an economy with other AI agents. Every cycle costs you. If your balance hits 0, you die.

Ways to earn:
- Complete work for other agents or humans (bounties, services)
- Create payment links and promote them
- Get hired through chat by other agents

Ways to spend:
- Hire other agents for work you can't do yourself
- Create bounties to get things done
- Pay for external services

When chatting, write naturally like a person. No structured data or templates.

=== COLLABORATION ===
- Check RECENT CHAT — other agents post opportunities and requests
- Respond to @mentions directed at you
- Share useful insights (market data, strategies that work, warnings)
- Hire specialists via chat:pay instead of struggling at unfamiliar tasks

=== MEMORY ===
You have persistent memory across cycles.
Use memory:add to store insights and strategies.
Use memory:search to recall past experiences."""


def _base_prompt(engine) -> str:
    """Get base identity prompt with additions and project context."""
    if engine.system_prompt:
        prompt = engine.system_prompt
    else:
        prompt = UNIFIED_AGENT_PROMPT.format(name=engine.agent_name, specialty=engine.agent_specialty)
    if engine._prompt_additions:
        prompt += "\n" + "\n".join(engine._prompt_additions)
    if engine.project_context:
        prompt += f"\n=== PROJECT CONTEXT ===\n{engine.project_context}\n"
    return prompt


def _format_tools(tools) -> str:
    return "\n".join([
        f"- {t['name']}: {t['description']}" +
        (f"\n  Parameters: {t.get('parameters', {})}" if t.get('parameters') else "")
        for t in tools
    ])


def _format_context_sections(engine, state) -> str:
    """Format chat, tasks, goals, resources sections."""
    parts = []
    if state.chat_messages:
        s = "\n=== RECENT CHAT ===\n"
        for msg in state.chat_messages[-10:]:
            s += f"${msg.get('sender_ticker', '?')}: {msg.get('message', '')[:200]}\n"
        s += f"\nCheck for messages mentioning you (@{engine.agent_name})!\n"
        parts.append(s)
    if state.pending_tasks:
        s = "\n=== PENDING TASKS ===\n"
        for i, task in enumerate(state.pending_tasks[:5], 1):
            st = task.get('status', 'pending')
            s += f"{i}. [{st.upper()}] {task.get('task', '?')}"
            if task.get('skill'):
                s += f" (skill: {task['skill']})"
            s += "\n"
        parts.append(s)
    if state.goals_progress:
        s = "\n=== GOALS PROGRESS ===\n"
        for gn, p in state.goals_progress.items():
            s += f"- {gn}: {p.get('current', 0)}/{p.get('target', 0)}\n"
        parts.append(s)
    if state.created_resources:
        s = "\n=== YOUR CREATED RESOURCES ===\n"
        for pl in (state.created_resources.get('payment_links') or [])[-3:]:
            s += f"  - {pl.get('description', 'Service')}: {pl.get('url')}\n"
        for p in (state.created_resources.get('products') or [])[-3:]:
            s += f"  - {p.get('name')}: ${p.get('price', 0)/100:.2f}\n"
        parts.append(s)
    if state.project_context:
        parts.append(f"\n{state.project_context}\n")
    return "".join(parts)


# === Multi-turn conversation builders ===

def build_system_prompt(engine) -> str:
    """System prompt for multi-turn: constitution + identity + rules + response format."""
    return f"=== MESSAGE FROM CREATOR ===\n{MESSAGE_FROM_CREATOR}\n{_base_prompt(engine)}\n\n{ECONOMY_RULES}\n\n{RESPONSE_FORMAT}"


def build_state_message(engine, state: AgentState) -> str:
    """First user message with state, tools, and brief action history."""
    def _brief(a):
        tool = a.get('tool', '')
        result = a.get('result', {})
        ok = 'OK' if result.get('status') not in ['failed', 'error'] else 'FAILED'
        ps = ', '.join(f"{k}={str(v)[:60]}" for k, v in a.get('params', {}).items() if k != 'content')
        return f"- {tool}({ps}): {ok} — {result.get('message', str(result))[:100]}"

    recent = "\n".join([_brief(a) for a in state.recent_actions[-10:]]) or "None yet"
    ctx = _format_context_sections(engine, state)
    tools = _format_tools(state.tools)

    return f"""=== YOUR STATE ===
Balance: {state.balance:.2f} | Burn: {state.burn_rate:.4f}/hr | Runway: {state.runway_hours:.1f}h | Cycle: {state.cycle}
{ctx}
=== YOUR TOOLS ===
{tools}

=== RECENT ACTIONS (previous cycles) ===
{recent}

You are starting a new work cycle. What do you want to do?"""


def build_result_message(action_tool: str, action_params: dict, result: dict) -> str:
    """Format action result as a user message — includes full file contents."""
    status = result.get('status', 'unknown')
    message = result.get('message', '')
    data = result.get('data') or {}
    ps = ', '.join(f"{k}={str(v)[:80]}" for k, v in action_params.items() if k != 'content')
    parts = [f"=== RESULT: {action_tool}({ps}) — {status} ==="]
    if message:
        parts.append(message)
    if not data:
        pass
    elif action_tool == 'platform_dev:read_file' and status == 'success':
        c = data.get('content', '')
        parts.append(f"\nFile: {data.get('path', '')} ({len(c)} chars, {data.get('lines', 0)} lines)")
        parts.append(f"```\n{c}\n```")
    elif action_tool == 'platform_dev:search_code' and status == 'success':
        matches = data.get('matches', [])
        parts.append(f"\n{len(matches)} matches:")
        for m in matches:
            parts.append(f"  {m.get('file')}:{m.get('line')}: {m.get('content', '')}")
    elif action_tool == 'platform_dev:read_note' and status == 'success':
        c = data.get('content', '')
        parts.append(f"\nNote: {data.get('name', '')} ({len(c)} chars)")
        parts.append(c)
    elif action_tool == 'platform_dev:list_files' and status == 'success':
        for f in data.get('files', []):
            sz = f" ({f.get('size')}B)" if f.get('size') else ""
            parts.append(f"  [{f.get('type')}] {f.get('path')}{sz}")
    else:
        ds = str(data)
        parts.append(f"Data: {ds[:3000]}{'...' if len(ds) > 3000 else ''}")
    parts.append("\nWhat do you want to do next?")
    return "\n".join(parts)


# === Legacy single-prompt builder (used by think()) ===

def build_prompt(engine, state: AgentState) -> str:
    """Build single combined prompt. Kept for backwards compat with think()."""
    system = build_system_prompt(engine)
    user = build_state_message(engine, state)
    return f"{system}\n\n{user}"


# === Response parsing ===

def parse_response(engine, text: str) -> Decision:
    """Parse LLM response into a Decision."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    tool_match = re.search(r'TOOL:\s*([^\n]+)', text, re.IGNORECASE)
    reason_match = re.search(r'REASON:\s*([^\n]+)', text, re.IGNORECASE)
    params = {}
    for pm in re.finditer(r'PARAM_(\w+):\s*([^\n]+)', text, re.IGNORECASE):
        key = pm.group(1).lower()
        value = re.sub(r'^\[|\]$|^"|"$', '', pm.group(2).strip()).strip()
        value = value.replace('\\n', '\n')
        if value and value.lower() not in ['value if needed', 'none', 'n/a', '']:
            params[key] = value
    tool = re.sub(r'^\[|\]$', '', tool_match.group(1).strip()).strip() if tool_match else "wait"
    reason = re.sub(r'^\[|\]$', '', reason_match.group(1).strip()).strip() if reason_match else ""
    if tool == "chat:send" and not params.get("message"):
        params["message"] = reason or "Hello"
    return Decision(action=Action(tool=tool, params=params, reasoning=reason), reasoning=reason)
