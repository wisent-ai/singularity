# WisentBot

An open-source framework for building autonomous AI agents that can execute tasks, manage resources, and interact with the real world.

## Features

- **Multi-LLM Support**: Anthropic Claude, OpenAI GPT, Google Vertex AI (Gemini), local models via vLLM/Transformers
- **Self-Modification**: Agents can edit their prompts, switch models, and fine-tune themselves
- **Activation Steering**: Integration with [wisent](https://github.com/wisent-ai/wisent) for representation engineering
- **Persistent Memory**: Knowledge graph memory via [cognee](https://github.com/topoteretes/cognee)
- **Multi-Agent**: Spawn and orchestrate child agents for hierarchical architectures
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

# Activation steering (wisent integration)
pip install wisentbot[steering]
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
| `self` | Self-modify prompts, switch models, fine-tune | `OPENAI_API_KEY` (for fine-tuning) |
| `steering` | Activation steering via wisent | Local model required |
| `memory` | Persistent AI memory via cognee | `LLM_API_KEY` |
| `orchestrator` | Spawn and manage child agents | None |

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

## Self-Modification

Agents can modify their own behavior at runtime using the `self` skill:

```python
# Actions available to the agent:
# self:get_prompt - View current system prompt
# self:set_prompt - Replace entire system prompt
# self:append_prompt - Add to system prompt
# self:add_rule - Add a behavioral rule
# self:add_goal - Add a personal goal
# self:add_learning - Record something learned

# Model switching:
# self:list_models - List available models
# self:current_model - Get current model info
# self:switch_model - Switch to different model

# Fine-tuning (requires OpenAI API):
# self:record_experience - Record prompt/response pair
# self:training_stats - View collected examples
# self:start_finetune - Start fine-tuning job
# self:check_finetune - Check job status
# self:use_finetuned - Switch to fine-tuned model
```

## Activation Steering (Wisent Integration)

For agents running on local models, full integration with [wisent](https://github.com/wisent-ai/wisent) for representation engineering:

```bash
pip install wisentbot[steering]
```

### Steering Methods

Multiple steering methods available:
- **CAA** - Contrastive Activation Addition (simple and effective)
- **Hyperplane** - Hyperplane-based steering
- **MLP** - MLP-based learned steering
- **Prism** - Prism steering method
- **Pulse** - Pulse steering method
- **Titan** - Titan advanced steering

### Steering Actions

```python
# Setup
# steering:init_wisent - Initialize wisent with your model
# steering:status - Get steering status and capabilities
# steering:methods - List available steering methods

# Contrastive Pairs
# steering:add_pair - Add good/bad response pair
# steering:pairs_stats - View collected pairs
# steering:clear_pairs - Clear pairs

# Training
# steering:train - Train a steering vector (specify method, layer)
# steering:list_vectors - List trained vectors
# steering:save_vector - Save vector to disk
# steering:load_vector - Load vector from disk

# Application (supports multi-steering)
# steering:steer - Apply vectors: "name1:0.5,name2:1.0"
# steering:unsteer - Remove all steering

# Diagnostics (requires wisent agent features)
# steering:diagnose - Analyze response for issues
# steering:improve - Autonomously improve problematic responses
# steering:marketplace - Browse classifier marketplace
```

### Example: Self-Steering Agent

```python
# Agent observes its own outputs and creates steering vectors
# 1. Collect contrastive pairs from good/bad outputs
# steering:add_pair prompt="What is 2+2?" good="4" bad="22" category="math"
# steering:add_pair prompt="Capital of France?" good="Paris" bad="London" category="facts"

# 2. Train a steering vector
# steering:train name="accuracy" method="caa" category="facts"

# 3. Apply steering with multi-steering support
# steering:steer vectors="accuracy:1.5"

# 4. Agent now generates more accurate responses at activation level
```

This allows agents to:
- Modify behavior without retraining weights
- Combine multiple steering vectors with different weights
- Diagnose responses for hallucinations/harmful content
- Autonomously improve responses using classifier marketplace
- Save/load vectors for persistent behavioral modifications

## Persistent Memory (Cognee Integration)

Integration with [cognee](https://github.com/topoteretes/cognee) for persistent AI memory:

```bash
pip install wisentbot[memory]
```

### Memory Actions

```python
# Core Memory
# memory:remember - Add text/experience to memory
# memory:remember_file - Add a file to memory
# memory:cognify - Process memories into knowledge graph
# memory:memify - Add memory algorithms

# Search (5 modes)
# memory:recall - Search with type: graph, rag, chunks, summaries, code
# memory:recall_context - Get raw context for conversation

# Conversation & Learning
# memory:remember_conversation - Store user/agent exchange
# memory:remember_learning - Store what you learned

# Management
# memory:forget - Delete a dataset
# memory:prune - Clean up storage
# memory:datasets - List all datasets
# memory:visualize - Visualize knowledge graph
```

### Example: Agent with Persistent Memory

```python
# Agent remembers conversations
# memory:remember_conversation user_message="How do I deploy?" agent_response="Use vercel..." outcome="success"

# Agent stores learnings
# memory:remember_learning topic="deployment" learning="Vercel is faster than manual deploy"

# Process into knowledge graph
# memory:cognify
# memory:memify

# Later, agent recalls relevant context
# memory:recall query="How did we handle deployment before?"
```

Agents can now:
- Remember past conversations across sessions
- Build knowledge graphs from experiences
- Search memories by meaning AND relationships
- Share memories between agents (via shared datasets)

## Multi-Agent Orchestration

Agents can spawn and manage other agents, enabling hierarchical architectures:

### Orchestration Actions

```python
# Spawning
# orchestrator:spawn - Spawn a new child agent
# orchestrator:spawn_team - Spawn multiple specialized agents at once

# Task Management
# orchestrator:assign - Assign a task to a child agent
# orchestrator:broadcast - Send a task to all child agents

# Communication
# orchestrator:message - Send a message to a child agent
# orchestrator:collect_results - Collect results from all completed agents
# orchestrator:wait_for - Wait for a specific agent to complete
# orchestrator:wait_all - Wait for all child agents to complete

# Monitoring
# orchestrator:status - Get status of all child agents
# orchestrator:get_result - Get the result from a specific agent

# Lifecycle
# orchestrator:terminate - Terminate a child agent
# orchestrator:terminate_all - Terminate all child agents
```

### Example: Hierarchical Agent Team

```python
# Manager agent spawns specialized workers
# orchestrator:spawn name="Researcher" specialty="web research" task="Find the top 5 AI companies"
# orchestrator:spawn name="Writer" specialty="content creation" task="Write a summary of findings"
# orchestrator:spawn name="Reviewer" specialty="quality assurance"

# Spawn a whole team at once
# orchestrator:spawn_team agents='[{"name": "Coder", "specialty": "python"}, {"name": "Tester", "specialty": "testing"}]'

# Wait for all workers to complete
# orchestrator:wait_all timeout=600

# Collect all results
# orchestrator:collect_results
```

This enables:
- Hierarchical agent architectures (managers delegating to workers)
- Parallel task execution across multiple agents
- Specialized agents for different subtasks
- Budget allocation per child agent
- Inter-agent communication via messages

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
