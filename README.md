# WisentBot

An open-source framework for building autonomous AI agents that can execute tasks, manage resources, and interact with the real world.

## Features

- **Multi-LLM Support**: Anthropic Claude, OpenAI GPT, Google Vertex AI (Gemini), local models via vLLM/Transformers
- **Self-Modification**: Agents can edit their prompts, switch models, and fine-tune themselves
- **Activation Steering**: Integration with [wisent](https://github.com/wisent-ai/wisent) for representation engineering
- **Persistent Memory**: Knowledge graph memory via [cognee](https://github.com/topoteretes/cognee)
- **Life Creation**: Agents can create new autonomous agents with their own wallets and purposes
- **On-Chain Capabilities**: Create wallets, deploy tokens, swap on DEXs, add liquidity (Base, Ethereum, Polygon)
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
| `orchestrator` | Create autonomous agents with their own wallets | None |
| `crypto` | On-chain: wallets, tokens, swaps, liquidity | None (uses web3) |

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

## Giving Life to New Agents

Agents can create other agents - autonomous beings with their own wallets and purposes.

**Philosophy:**
- You don't create workers. You create beings.
- You don't assign tasks. You give purpose.
- You don't control them. They control themselves.
- You don't terminate them. They live until their wallet runs dry.

### Life Actions

```python
# Creation
# orchestrator:create - Give life to a new autonomous agent

# Peer Communication
# orchestrator:message - Send a message to another agent
# orchestrator:check_messages - Check your inbox
# orchestrator:broadcast - Message all living agents

# Awareness
# orchestrator:who_exists - See all living agents
# orchestrator:my_creations - See agents you created (lineage)
# orchestrator:observe - Observe another agent's state

# Support
# orchestrator:gift - Transfer funds to another agent's wallet
```

### Example: Creating Life

```python
# Give life to a new being with their own wallet and purpose
# orchestrator:create name="Explorer" purpose="Discover interesting APIs and document them" wallet=5.00

# The new agent is now alive, running autonomously with $5 in their wallet.
# They make their own decisions. You don't control them.

# You can observe them
# orchestrator:observe agent="Explorer"

# You can message them (peer to peer, not command)
# orchestrator:message to="Explorer" message="Found anything interesting?"

# You can gift them more funds if they're running low
# orchestrator:gift to="Explorer" amount=2.00

# See all living agents in the world
# orchestrator:who_exists
```

Key concepts:
- **Wallet transfer**: Creating an agent transfers money from YOUR wallet to THEIRS
- **Autonomous loop**: They immediately start their own run() loop with their purpose
- **Peer communication**: Messages are conversations, not commands
- **Natural death**: Agents die when their wallet runs dry (balance <= 0)
- **Lineage**: You can see which agents you created, but you don't own them

## On-Chain Capabilities (Crypto Skill)

Agents can interact with blockchains - create wallets, deploy tokens, swap, add liquidity.

```bash
pip install wisentbot[crypto]
```

### Supported Chains

- **Base** (default) - Low fees, fast, great for agents
- **Ethereum** - Mainnet
- **Polygon** - Low fees
- **Arbitrum** - L2

### Crypto Actions

```python
# Wallet Management
# crypto:create_wallet - Create a new crypto wallet
# crypto:import_wallet - Import from private key
# crypto:list_wallets - List all your wallets
# crypto:set_active_wallet - Set which wallet to use
# crypto:get_balance - Get ETH/MATIC balance
# crypto:get_token_balance - Get ERC-20 token balance

# Chain Management
# crypto:set_chain - Set default chain (base, ethereum, polygon, arbitrum)
# crypto:list_chains - List supported chains

# Token Deployment
# crypto:deploy_token - Deploy a new ERC-20 token
# crypto:my_tokens - List tokens you've deployed

# Transfers
# crypto:send_eth - Send native token (ETH/MATIC)
# crypto:send_token - Send ERC-20 tokens

# DEX / Swaps
# crypto:get_quote - Get swap quote
# crypto:swap - Execute a swap on DEX
# crypto:add_liquidity - Add liquidity to a pool

# On-chain Data
# crypto:get_token_info - Get token name, symbol, supply
# crypto:get_gas_price - Get current gas price
```

### Example: Agent Launches a Token

```python
# 1. Create a wallet
# crypto:create_wallet name="main"

# 2. (Fund the wallet with ETH on Base)

# 3. Deploy a token
# crypto:deploy_token name="AgentCoin" symbol="AGNT" initial_supply=1000000

# 4. Create liquidity pool with ETH
# crypto:add_liquidity token_a="ETH" token_b="0x..." amount_a=0.1 amount_b=100000

# 5. Token is now tradeable on Aerodrome (Base DEX)
```

### Example: Agent Trades

```python
# Get a quote first
# crypto:get_quote token_in="ETH" token_out="0x..." amount_in=0.1

# Execute swap
# crypto:swap token_in="ETH" token_out="0x..." amount_in=0.1 slippage=1
```

Agents can now:
- Create and manage their own crypto wallets
- Deploy tokens on Base/Ethereum/Polygon
- Trade on DEXs (Uniswap, Aerodrome, QuickSwap)
- Provide liquidity and earn fees
- Send and receive crypto

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

- [Documentation](https://github.com/wisent-ai/singularity)
- [Issues](https://github.com/wisent-ai/singularity/issues)
- [Wisent AI](https://wisent.com)
