# WisentBot

An open-source framework for building autonomous AI agents that can execute tasks, manage resources, and interact with the real world.

## Features

- **Multi-LLM Support**: Anthropic Claude, OpenAI GPT, Google Vertex AI (Gemini), local models via vLLM/Transformers
- **Modular Skills**: Extensible skill system for adding new capabilities
- **Cost Tracking**: Built-in API cost and resource tracking
- **Async First**: Fully asynchronous for high performance

## Installation

```bash
pip install wisentbot
```

With optional dependencies:

```bash
# All features
pip install wisentbot[all]

# Specific features
pip install wisentbot[twitter,github,browser]

# Local GPU inference
pip install wisentbot[gpu]
```

## Quick Start

```python
import asyncio
from wisentbot import AutonomousAgent

async def main():
    agent = AutonomousAgent(
        name="MyAgent",
        ticker="AGENT",
        starting_balance=10.0,  # USD budget
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
    )
    await agent.run()

asyncio.run(main())
```

## Environment Variables

```bash
# Required (at least one LLM provider)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional - for specific skills
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_SECRET=...
GITHUB_TOKEN=ghp_...
RESEND_API_KEY=re_...
VERCEL_TOKEN=...
```

## Available Skills

| Skill | Description | Required Env Vars |
|-------|-------------|-------------------|
| `content` | Generate text content | LLM API key |
| `twitter` | Post tweets, read timeline | `TWITTER_*` |
| `github` | Manage repos, issues, PRs | `GITHUB_TOKEN` |
| `email` | Send emails via Resend | `RESEND_API_KEY` |
| `browser` | Web automation | None (uses Playwright) |
| `filesystem` | Read/write files | None |
| `shell` | Execute shell commands | None |
| `request` | Make HTTP requests | None |
| `vercel` | Deploy to Vercel | `VERCEL_TOKEN` |
| `namecheap` | Manage domains | `NAMECHEAP_*` |
| `mcp` | MCP protocol client | None |

## Creating Custom Skills

```python
from wisentbot.skills import Skill, SkillManifest, SkillAction, SkillResult

class MySkill(Skill):
    def __init__(self, credentials: dict):
        manifest = SkillManifest(
            skill_id="myskill",
            name="My Custom Skill",
            description="Does something useful",
            actions=[
                SkillAction(
                    name="do_thing",
                    description="Does the thing",
                    parameters={"input": {"type": "string", "required": True}}
                )
            ]
        )
        super().__init__(manifest, credentials)

    def check_credentials(self) -> bool:
        return True  # Or check required credentials

    async def execute(self, action: str, params: dict) -> SkillResult:
        if action == "do_thing":
            # Do the thing
            return SkillResult(success=True, data={"result": "done"})
        return SkillResult(success=False, message=f"Unknown action: {action}")
```

## LLM Providers

### Anthropic (Default)

```python
agent = AutonomousAgent(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-20250514",
)
```

### OpenAI

```python
agent = AutonomousAgent(
    llm_provider="openai",
    llm_model="gpt-4o",
)
```

### Local (vLLM on CUDA)

```python
agent = AutonomousAgent(
    llm_provider="vllm",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
)
```

### Local (Transformers on Apple Silicon)

```python
agent = AutonomousAgent(
    llm_provider="transformers",
    llm_model="Qwen/Qwen2.5-1.5B-Instruct",
)
```

### Google Vertex AI

```python
agent = AutonomousAgent(
    llm_provider="vertex",
    llm_model="gemini-2.0-flash-001",
)
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              AutonomousAgent                │
│  ┌─────────────────────────────────────┐    │
│  │         CognitionEngine             │    │
│  │  ┌─────────┐  ┌─────────────────┐   │    │
│  │  │   LLM   │  │  System Prompt  │   │    │
│  │  └─────────┘  └─────────────────┘   │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │          SkillRegistry              │    │
│  │  ┌───────┐ ┌───────┐ ┌───────┐     │    │
│  │  │Twitter│ │GitHub │ │Browser│ ... │    │
│  │  └───────┘ └───────┘ └───────┘     │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │         State & Resources           │    │
│  │   Balance, Actions, Created Items   │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Agent Loop

1. **Observe**: Gather current state (balance, recent actions, available tools)
2. **Think**: LLM decides what action to take
3. **Act**: Execute the chosen skill action
4. **Record**: Track costs, results, and created resources
5. **Repeat**: Continue until balance depleted or stopped

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## Links

- [Documentation](https://github.com/wisent-ai/wisentbot)
- [Issues](https://github.com/wisent-ai/wisentbot/issues)
- [Wisent AI](https://wisent.ai)
