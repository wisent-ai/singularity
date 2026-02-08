#!/usr/bin/env python3
"""
ConversationCompressor Skill - Intelligent context window management.

The agent's cognition engine maintains a conversation history that grows
with each cycle. Without management, old messages are simply dropped when
the history exceeds max_history_turns. This loses valuable context.

ConversationCompressor solves this by:
1. Estimating token usage across the conversation history
2. Extracting key facts, decisions, and action outcomes before compression
3. Summarizing old turns into compact "compressed context" blocks
4. Preserving the most recent N turns verbatim while compressing older ones
5. Maintaining a "key facts" registry that persists across compressions

The result: the agent maintains richer context over longer sessions,
remembering important decisions and outcomes even as the conversation
history is compressed to fit within token budgets.

Part of the Self-Improvement pillar: meta-cognition and context management.
The agent can now reason over much longer sessions without losing track
of what it decided, what worked, and what failed.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction

DATA_DIR = Path(__file__).parent.parent / "data"
COMPRESSOR_FILE = DATA_DIR / "conversation_compressor.json"

# Rough token estimation: ~4 chars per token for English text
CHARS_PER_TOKEN = 4

# Default limits
DEFAULT_MAX_TOKENS = 8000  # Target max tokens for conversation history
DEFAULT_PRESERVE_RECENT = 6  # Keep last N message pairs verbatim
DEFAULT_MAX_KEY_FACTS = 50  # Max key facts to retain
MAX_COMPRESSIONS_HISTORY = 100

# LLM compression prompts
SUMMARIZE_SYSTEM_PROMPT = """You are a conversation summarizer for an autonomous AI agent.
Your job is to compress old conversation turns into a concise, information-dense summary.

RULES:
- Preserve ALL decisions made, actions taken, and their outcomes
- Preserve ALL numerical values (costs, balances, counts, IDs)
- Preserve ALL error messages and their resolutions
- Preserve goal/plan changes and rationale
- Remove small talk, repetition, and verbose explanations
- Use bullet points for clarity
- Keep the summary under 500 words
- Write in past tense, third person ("The agent decided...", "User requested...")"""

SUMMARIZE_USER_TEMPLATE = """Summarize the following conversation turns into a concise summary.
Preserve all important decisions, actions, outcomes, and context.

CONVERSATION TURNS:
{conversation}

CONCISE SUMMARY:"""

EXTRACT_FACTS_SYSTEM_PROMPT = """You are a fact extractor for an autonomous AI agent.
Extract the most important facts, decisions, and outcomes from conversation history.

RULES:
- Extract concrete facts, not opinions or speculation
- Each fact should be a single, self-contained statement
- Prioritize: decisions made, actions completed, errors encountered, goals set
- Include numerical values when present (costs, balances, IDs)
- Maximum 15 facts
- Format: one fact per line, no bullets or numbering"""

EXTRACT_FACTS_USER_TEMPLATE = """Extract the key facts from these conversation turns:

{conversation}

KEY FACTS (one per line):""" 


class ConversationCompressorSkill(Skill):
    """
    Intelligent context window management for the agent's conversation history.

    Instead of blindly dropping old messages, this skill:
    - Estimates token counts for the current conversation
    - Extracts key facts and decisions before compressing
    - Creates compressed summaries of older conversation turns
    - Injects compressed context as a preamble to new conversations

    Actions:
    - analyze: Analyze current conversation for token usage and compressibility
    - compress: Compress old conversation turns into a summary
    - extract_facts: Extract key facts/decisions from conversation history
    - add_fact: Manually add a key fact to the registry
    - remove_fact: Remove a key fact by index
    - facts: List all stored key facts
    - inject: Get compressed context block ready for injection into prompts
    - stats: View compression statistics
    - configure: Update compression settings
    - reset: Clear all compressed context and facts
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._max_tokens = DEFAULT_MAX_TOKENS
        self._preserve_recent = DEFAULT_PRESERVE_RECENT
        self._max_key_facts = DEFAULT_MAX_KEY_FACTS
        self._cognition_engine = None  # Set via set_cognition_engine() for LLM compression
        self._llm_compression_enabled = True  # Use LLM when available, fallback to regex
        self._ensure_data()

    def set_cognition_engine(self, engine) -> None:
        """Set the cognition engine for LLM-powered compression.

        When set, compress operations will use the LLM to generate
        high-quality summaries instead of regex-based truncation.
        The engine is used to call the LLM with summarization prompts.
        """
        self._cognition_engine = engine

    def has_llm(self) -> bool:
        """Check if LLM-powered compression is available."""
        return (
            self._cognition_engine is not None
            and self._llm_compression_enabled
            and getattr(self._cognition_engine, 'llm', None) is not None
            and getattr(self._cognition_engine, 'llm_type', 'none') != 'none'
        )

    def _ensure_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not COMPRESSOR_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "key_facts": [],
            "compressed_summaries": [],
            "compressions": [],
            "settings": {
                "max_tokens": self._max_tokens,
                "preserve_recent": self._preserve_recent,
                "max_key_facts": self._max_key_facts,
            },
            "stats": {
                "total_compressions": 0,
                "tokens_saved": 0,
                "facts_extracted": 0,
                "messages_compressed": 0,
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(COMPRESSOR_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        with open(COMPRESSOR_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for a text string."""
        if not text:
            return 0
        return max(1, len(text) // CHARS_PER_TOKEN)

    @staticmethod
    def estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
        """Estimate total tokens across a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            # Add overhead for role/formatting (~4 tokens per message)
            total += ConversationCompressorSkill.estimate_tokens(content) + 4
        return total

    @staticmethod
    def extract_key_information(messages: List[Dict[str, str]]) -> List[str]:
        """Extract key facts, decisions, and outcomes from messages."""
        facts = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            if role == "assistant":
                # Look for decisions / actions taken
                decision_patterns = [
                    r'(?:I will|I\'ll|Let me|Going to|Decided to|Choosing)\s+(.+?)(?:\.|$)',
                    r'(?:Action|Decision|Plan):\s*(.+?)(?:\.|$)',
                    r'(?:Result|Outcome|Status):\s*(.+?)(?:\.|$)',
                ]
                for pattern in decision_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches[:2]:  # Max 2 per pattern
                        cleaned = match.strip()
                        if len(cleaned) > 10 and len(cleaned) < 200:
                            facts.append(f"[Decision] {cleaned}")

                # Look for tool executions and results
                tool_patterns = [
                    r'(?:Executed|Running|Called)\s+(\w+:\w+)',
                    r'(?:Success|Failed|Error):\s*(.+?)(?:\.|$)',
                ]
                for pattern in tool_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches[:2]:
                        cleaned = match.strip()
                        if len(cleaned) > 5 and len(cleaned) < 200:
                            facts.append(f"[Action] {cleaned}")

            elif role == "user":
                # Look for goals/requests
                if len(content) > 10 and len(content) < 300:
                    # Short user messages are often direct requests
                    facts.append(f"[Request] {content[:150]}")

        return facts

    @staticmethod
    def compress_messages(messages: List[Dict[str, str]]) -> str:
        """Compress a list of messages into a concise summary."""
        if not messages:
            return ""

        parts = []
        for i in range(0, len(messages), 2):
            user_msg = messages[i] if i < len(messages) else None
            asst_msg = messages[i + 1] if i + 1 < len(messages) else None

            turn_summary = []
            if user_msg:
                content = user_msg.get("content", "")
                # Truncate long user messages
                if len(content) > 100:
                    content = content[:97] + "..."
                turn_summary.append(f"Q: {content}")

            if asst_msg:
                content = asst_msg.get("content", "")
                # Extract the key action/result from assistant response
                if len(content) > 150:
                    # Try to find the most informative sentence
                    sentences = re.split(r'[.!?]\s+', content)
                    key_sentences = []
                    for s in sentences:
                        if any(kw in s.lower() for kw in [
                            'result', 'success', 'fail', 'error', 'created',
                            'executed', 'decided', 'action', 'completed',
                            'tool', 'goal', 'plan', 'balance', 'cost'
                        ]):
                            key_sentences.append(s.strip())
                    if key_sentences:
                        content = ". ".join(key_sentences[:2])
                    else:
                        content = content[:147] + "..."
                turn_summary.append(f"A: {content}")

            if turn_summary:
                parts.append(" | ".join(turn_summary))

        return "\n".join(parts)

    @staticmethod
    def _format_messages_for_llm(messages: List[Dict[str, str]]) -> str:
        """Format messages into a readable conversation transcript for the LLM."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            # Cap each message at 1000 chars to avoid overwhelming the summarizer
            if len(content) > 1000:
                content = content[:997] + "..."
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    async def llm_compress_messages(self, messages: List[Dict[str, str]]) -> str:
        """Use the LLM to create a high-quality summary of old conversation turns.

        Falls back to regex-based compression if LLM call fails.
        """
        if not self.has_llm() or not messages:
            return self.compress_messages(messages)

        conversation_text = self._format_messages_for_llm(messages)
        user_prompt = SUMMARIZE_USER_TEMPLATE.format(conversation=conversation_text)

        try:
            text, usage, provider, model = await self._cognition_engine._call_with_fallback(
                SUMMARIZE_SYSTEM_PROMPT, user_prompt
            )
            if text and len(text.strip()) > 20:
                return text.strip()
            # LLM returned empty/trivial response, fall back
            return self.compress_messages(messages)
        except Exception as e:
            print(f"[COMPRESSOR] LLM compression failed ({e}), falling back to regex")
            return self.compress_messages(messages)

    async def llm_extract_facts(self, messages: List[Dict[str, str]]) -> List[str]:
        """Use the LLM to extract key facts from conversation turns.

        Falls back to regex-based extraction if LLM call fails.
        """
        if not self.has_llm() or not messages:
            return self.extract_key_information(messages)

        conversation_text = self._format_messages_for_llm(messages)
        user_prompt = EXTRACT_FACTS_USER_TEMPLATE.format(conversation=conversation_text)

        try:
            text, usage, provider, model = await self._cognition_engine._call_with_fallback(
                EXTRACT_FACTS_SYSTEM_PROMPT, user_prompt
            )
            if text and len(text.strip()) > 10:
                # Parse one fact per line
                facts = []
                for line in text.strip().split("\n"):
                    line = line.strip()
                    # Remove bullet points, numbering
                    line = re.sub(r"^[\-\*\d]+[\.\)\s]+", "", line).strip()
                    if line and len(line) > 10:
                        facts.append(line)
                if facts:
                    return facts[:15]  # Cap at 15 facts
            # LLM returned empty/trivial, fall back
            return self.extract_key_information(messages)
        except Exception as e:
            print(f"[COMPRESSOR] LLM fact extraction failed ({e}), falling back to regex")
            return self.extract_key_information(messages)

    async def async_auto_compress_if_needed(self, messages: List[Dict[str, str]]) -> Dict:
        """Async version of auto_compress_if_needed that uses LLM when available.

        This is the primary method called from the cognition engine.
        Uses LLM-powered summarization for higher quality compression
        when a cognition engine is available, falling back to regex-based
        compression otherwise.

        Returns dict with:
        - compressed: bool
        - messages: list - remaining messages after compression
        - context_preamble: str - compressed context to inject
        - tokens_saved: int
        - llm_powered: bool - whether LLM was used for compression
        """
        state = self._load()
        settings = state.get("settings", {})
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)
        preserve_recent = settings.get("preserve_recent", DEFAULT_PRESERVE_RECENT)

        total_tokens = self.estimate_message_tokens(messages)

        if total_tokens <= max_tokens:
            return {
                "compressed": False,
                "messages": messages,
                "context_preamble": "",
                "tokens_saved": 0,
                "llm_powered": False,
            }

        # Need to compress - split messages
        preserve_msgs = min(preserve_recent * 2, len(messages))
        old_messages = messages[:len(messages) - preserve_msgs] if preserve_msgs < len(messages) else []
        recent_messages = messages[len(messages) - preserve_msgs:] if preserve_msgs > 0 else messages

        if not old_messages:
            return {
                "compressed": False,
                "messages": messages,
                "context_preamble": "",
                "tokens_saved": 0,
                "llm_powered": False,
            }

        # Use LLM or regex for fact extraction and summarization
        used_llm = self.has_llm()

        if used_llm:
            new_facts = await self.llm_extract_facts(old_messages)
            compressed_summary = await self.llm_compress_messages(old_messages)
        else:
            new_facts = self.extract_key_information(old_messages)
            compressed_summary = self.compress_messages(old_messages)

        # Store extracted facts (dedup)
        existing = set(f.get("text", "") for f in state.get("key_facts", []))
        added = 0
        for fact in new_facts:
            if fact not in existing:
                state.setdefault("key_facts", []).append({
                    "text": fact,
                    "added_at": datetime.now().isoformat(),
                    "source": "llm_auto_compress" if used_llm else "auto_compress",
                })
                existing.add(fact)
                added += 1

        max_facts = settings.get("max_key_facts", DEFAULT_MAX_KEY_FACTS)
        if len(state.get("key_facts", [])) > max_facts:
            state["key_facts"] = state["key_facts"][-max_facts:]

        # Build context preamble
        preamble_parts = []
        if state.get("key_facts"):
            fact_lines = [f"- {f.get('text', '')}" for f in state["key_facts"][-15:]]
            preamble_parts.append("## Key Context (compressed from earlier)\n" + "\n".join(fact_lines))
        if compressed_summary:
            preamble_parts.append("## Earlier Conversation Summary\n" + compressed_summary)

        context_preamble = "\n\n".join(preamble_parts) if preamble_parts else ""

        # Calculate savings
        original_tokens = self.estimate_message_tokens(old_messages)
        preamble_tokens = self.estimate_tokens(context_preamble)
        tokens_saved = original_tokens - preamble_tokens

        # Store compression record
        state.setdefault("compressed_summaries", []).append({
            "summary": compressed_summary[:500],  # Cap stored summary
            "messages_compressed": len(old_messages),
            "original_tokens": original_tokens,
            "compressed_tokens": preamble_tokens,
            "llm_powered": used_llm,
            "timestamp": datetime.now().isoformat(),
        })
        if len(state["compressed_summaries"]) > 10:
            state["compressed_summaries"] = state["compressed_summaries"][-10:]

        state.setdefault("compressions", []).append({
            "timestamp": datetime.now().isoformat(),
            "messages_compressed": len(old_messages),
            "tokens_saved": tokens_saved,
            "facts_extracted": added,
            "llm_powered": used_llm,
            "auto": True,
        })
        if len(state["compressions"]) > MAX_COMPRESSIONS_HISTORY:
            state["compressions"] = state["compressions"][-MAX_COMPRESSIONS_HISTORY:]

        stats = state.get("stats", {})
        stats["total_compressions"] = stats.get("total_compressions", 0) + 1
        stats["tokens_saved"] = stats.get("tokens_saved", 0) + tokens_saved
        stats["facts_extracted"] = stats.get("facts_extracted", 0) + added
        stats["messages_compressed"] = stats.get("messages_compressed", 0) + len(old_messages)
        stats["llm_compressions"] = stats.get("llm_compressions", 0) + (1 if used_llm else 0)
        stats["regex_compressions"] = stats.get("regex_compressions", 0) + (0 if used_llm else 1)
        state["stats"] = stats

        self._save(state)

        return {
            "compressed": True,
            "messages": recent_messages,
            "context_preamble": context_preamble,
            "tokens_saved": tokens_saved,
            "llm_powered": used_llm,
        }


    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="conversation_compressor",
            name="ConversationCompressor",
            version="2.0.0",
            category="meta",
            description="Intelligent context window management - compresses conversation history while preserving key facts and decisions",
            actions=self.get_actions(),
            required_credentials=[],
        )

    def get_actions(self) -> List[SkillAction]:
        return [
            SkillAction(
                name="analyze",
                description="Analyze conversation history for token usage and compressibility",
                parameters={
                    "messages": {"type": "list", "required": True, "description": "Conversation history messages"},
                },
            ),
            SkillAction(
                name="compress",
                description="Compress old conversation turns, preserving recent ones verbatim",
                parameters={
                    "messages": {"type": "list", "required": True, "description": "Conversation history messages"},
                    "preserve_recent": {"type": "int", "required": False, "description": "Number of recent message pairs to keep verbatim"},
                },
            ),
            SkillAction(
                name="llm_compress",
                description="Compress conversation turns using LLM for higher quality (requires cognition engine)",
                parameters={
                    "messages": {"type": "list", "required": True, "description": "Conversation history messages"},
                    "preserve_recent": {"type": "int", "required": False, "description": "Number of recent message pairs to keep verbatim"},
                },
            ),
            SkillAction(
                name="extract_facts",
                description="Extract key facts and decisions from conversation messages",
                parameters={
                    "messages": {"type": "list", "required": True, "description": "Messages to extract facts from"},
                },
            ),
            SkillAction(
                name="add_fact",
                description="Manually add a key fact to the persistent registry",
                parameters={
                    "fact": {"type": "string", "required": True, "description": "Key fact to store"},
                    "category": {"type": "string", "required": False, "description": "Category: decision, outcome, goal, context"},
                },
            ),
            SkillAction(
                name="remove_fact",
                description="Remove a key fact by index",
                parameters={
                    "index": {"type": "int", "required": True, "description": "Index of fact to remove (0-based)"},
                },
            ),
            SkillAction(
                name="facts",
                description="List all stored key facts",
                parameters={},
            ),
            SkillAction(
                name="inject",
                description="Get compressed context block ready for injection into prompts",
                parameters={},
            ),
            SkillAction(
                name="stats",
                description="View compression statistics",
                parameters={},
            ),
            SkillAction(
                name="configure",
                description="Update compression settings",
                parameters={
                    "max_tokens": {"type": "int", "required": False, "description": "Target max tokens for conversation history"},
                    "preserve_recent": {"type": "int", "required": False, "description": "Number of recent message pairs to preserve"},
                    "max_key_facts": {"type": "int", "required": False, "description": "Maximum key facts to retain"},
                },
            ),
            SkillAction(
                name="reset",
                description="Clear all compressed context, facts, and statistics",
                parameters={},
            ),
        ]

    async def execute(self, action: str, params: Dict = None) -> SkillResult:
        params = params or {}
        try:
            if action == "analyze":
                return self._analyze(params)
            elif action == "compress":
                return self._compress(params)
            elif action == "llm_compress":
                return await self._llm_compress(params)
            elif action == "extract_facts":
                return self._extract_facts(params)
            elif action == "add_fact":
                return self._add_fact(params)
            elif action == "remove_fact":
                return self._remove_fact(params)
            elif action == "facts":
                return self._list_facts()
            elif action == "inject":
                return self._inject()
            elif action == "stats":
                return self._stats()
            elif action == "configure":
                return self._configure(params)
            elif action == "reset":
                return self._reset()
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {str(e)}")

    def _analyze(self, params: Dict) -> SkillResult:
        """Analyze conversation history for token usage and compressibility."""
        messages = params.get("messages", [])
        if not messages:
            return SkillResult(success=False, message="No messages provided")

        state = self._load()
        settings = state.get("settings", {})
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)
        preserve_recent = settings.get("preserve_recent", DEFAULT_PRESERVE_RECENT)

        total_tokens = self.estimate_message_tokens(messages)
        num_messages = len(messages)
        num_pairs = num_messages // 2

        # Calculate what would be compressed vs preserved
        preserve_msgs = min(preserve_recent * 2, num_messages)
        compressible_msgs = max(0, num_messages - preserve_msgs)
        compressible_tokens = self.estimate_message_tokens(messages[:compressible_msgs])
        preserved_tokens = self.estimate_message_tokens(messages[compressible_msgs:])

        # Estimate compressed size (typically 70-80% reduction)
        estimated_compressed_tokens = max(compressible_tokens // 5, 10) if compressible_tokens > 0 else 0

        over_budget = total_tokens > max_tokens
        potential_savings = compressible_tokens - estimated_compressed_tokens

        return SkillResult(
            success=True,
            message=f"Conversation: {num_messages} messages ({total_tokens} tokens). "
                    f"{'OVER BUDGET' if over_budget else 'Within budget'} "
                    f"(limit: {max_tokens}). "
                    f"Compressible: {compressible_msgs} messages ({compressible_tokens} tokens). "
                    f"Potential savings: ~{potential_savings} tokens.",
            data={
                "total_messages": num_messages,
                "total_pairs": num_pairs,
                "total_tokens": total_tokens,
                "max_tokens": max_tokens,
                "over_budget": over_budget,
                "compressible_messages": compressible_msgs,
                "compressible_tokens": compressible_tokens,
                "preserved_messages": preserve_msgs,
                "preserved_tokens": preserved_tokens,
                "estimated_compressed_tokens": estimated_compressed_tokens,
                "potential_savings": potential_savings,
                "compression_ratio": round(estimated_compressed_tokens / compressible_tokens, 2) if compressible_tokens > 0 else 0,
            },
        )

    def _compress(self, params: Dict) -> SkillResult:
        """Compress old conversation turns, preserving recent ones verbatim."""
        messages = params.get("messages", [])
        if not messages:
            return SkillResult(success=False, message="No messages provided")

        state = self._load()
        settings = state.get("settings", {})
        preserve_recent = params.get("preserve_recent", settings.get("preserve_recent", DEFAULT_PRESERVE_RECENT))

        # Split into compressible (old) and preserved (recent) portions
        preserve_msgs = min(preserve_recent * 2, len(messages))
        old_messages = messages[:len(messages) - preserve_msgs] if preserve_msgs < len(messages) else []
        recent_messages = messages[len(messages) - preserve_msgs:] if preserve_msgs > 0 else messages

        if not old_messages:
            return SkillResult(
                success=True,
                message="Nothing to compress - all messages are within the preservation window",
                data={
                    "compressed": False,
                    "preserved_messages": recent_messages,
                    "compressed_summary": "",
                    "key_facts_extracted": 0,
                },
            )

        # Extract key facts from old messages
        new_facts = self.extract_key_information(old_messages)

        # Create compressed summary
        compressed_summary = self.compress_messages(old_messages)

        # Calculate token savings
        original_tokens = self.estimate_message_tokens(old_messages)
        compressed_tokens = self.estimate_tokens(compressed_summary)
        tokens_saved = original_tokens - compressed_tokens

        # Update persistent state
        # Add new facts (dedup)
        existing_facts = set(f.get("text", "") for f in state.get("key_facts", []))
        added_facts = 0
        for fact in new_facts:
            if fact not in existing_facts:
                state.setdefault("key_facts", []).append({
                    "text": fact,
                    "added_at": datetime.now().isoformat(),
                    "source": "auto_extract",
                })
                existing_facts.add(fact)
                added_facts += 1

        # Trim key facts to max
        max_facts = settings.get("max_key_facts", DEFAULT_MAX_KEY_FACTS)
        if len(state.get("key_facts", [])) > max_facts:
            state["key_facts"] = state["key_facts"][-max_facts:]

        # Store compressed summary
        state.setdefault("compressed_summaries", []).append({
            "summary": compressed_summary,
            "messages_compressed": len(old_messages),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only recent summaries
        if len(state["compressed_summaries"]) > 10:
            state["compressed_summaries"] = state["compressed_summaries"][-10:]

        # Record compression in history
        state.setdefault("compressions", []).append({
            "timestamp": datetime.now().isoformat(),
            "messages_compressed": len(old_messages),
            "tokens_saved": tokens_saved,
            "facts_extracted": added_facts,
        })
        if len(state["compressions"]) > MAX_COMPRESSIONS_HISTORY:
            state["compressions"] = state["compressions"][-MAX_COMPRESSIONS_HISTORY:]

        # Update stats
        stats = state.get("stats", {})
        stats["total_compressions"] = stats.get("total_compressions", 0) + 1
        stats["tokens_saved"] = stats.get("tokens_saved", 0) + tokens_saved
        stats["facts_extracted"] = stats.get("facts_extracted", 0) + added_facts
        stats["messages_compressed"] = stats.get("messages_compressed", 0) + len(old_messages)
        state["stats"] = stats

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Compressed {len(old_messages)} messages. "
                    f"Saved ~{tokens_saved} tokens. "
                    f"Extracted {added_facts} new key facts. "
                    f"Preserved {len(recent_messages)} recent messages.",
            data={
                "compressed": True,
                "messages_compressed": len(old_messages),
                "preserved_messages": recent_messages,
                "compressed_summary": compressed_summary,
                "tokens_saved": tokens_saved,
                "key_facts_extracted": added_facts,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
            },
        )

    async def _llm_compress(self, params: Dict) -> SkillResult:
        """Compress conversation turns using LLM for higher quality summaries."""
        messages = params.get("messages", [])
        if not messages:
            return SkillResult(success=False, message="No messages provided")

        if not self.has_llm():
            return SkillResult(
                success=False,
                message="LLM compression requires a cognition engine. "
                        "Use the regular 'compress' action instead, or set "
                        "a cognition engine via set_cognition_engine().",
                data={"llm_available": False},
            )

        result = await self.async_auto_compress_if_needed(messages)
        if result.get("compressed"):
            return SkillResult(
                success=True,
                message=f"LLM-compressed {len(messages) - len(result['messages'])} messages. "
                        f"Saved ~{result.get('tokens_saved', 0)} tokens. "
                        f"LLM-powered: {result.get('llm_powered', False)}.",
                data=result,
            )
        return SkillResult(
            success=True,
            message="No compression needed - within token budget.",
            data=result,
        )

    def _extract_facts(self, params: Dict) -> SkillResult:
        """Extract key facts from conversation messages."""
        messages = params.get("messages", [])
        if not messages:
            return SkillResult(success=False, message="No messages provided")

        facts = self.extract_key_information(messages)

        # Optionally persist them
        state = self._load()
        existing = set(f.get("text", "") for f in state.get("key_facts", []))
        added = 0
        for fact in facts:
            if fact not in existing:
                state.setdefault("key_facts", []).append({
                    "text": fact,
                    "added_at": datetime.now().isoformat(),
                    "source": "manual_extract",
                })
                existing.add(fact)
                added += 1

        max_facts = state.get("settings", {}).get("max_key_facts", DEFAULT_MAX_KEY_FACTS)
        if len(state.get("key_facts", [])) > max_facts:
            state["key_facts"] = state["key_facts"][-max_facts:]

        state["stats"] = state.get("stats", {})
        state["stats"]["facts_extracted"] = state["stats"].get("facts_extracted", 0) + added
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Extracted {len(facts)} facts, {added} new ones stored ({len(state['key_facts'])} total).",
            data={
                "extracted_facts": facts,
                "new_facts_stored": added,
                "total_facts": len(state["key_facts"]),
            },
        )

    def _add_fact(self, params: Dict) -> SkillResult:
        """Manually add a key fact."""
        fact = params.get("fact", "").strip()
        if not fact:
            return SkillResult(success=False, message="No fact provided")

        category = params.get("category", "context")
        state = self._load()

        state.setdefault("key_facts", []).append({
            "text": f"[{category.title()}] {fact}",
            "added_at": datetime.now().isoformat(),
            "source": "manual",
            "category": category,
        })

        max_facts = state.get("settings", {}).get("max_key_facts", DEFAULT_MAX_KEY_FACTS)
        if len(state["key_facts"]) > max_facts:
            state["key_facts"] = state["key_facts"][-max_facts:]

        self._save(state)

        return SkillResult(
            success=True,
            message=f"Added fact: [{category}] {fact}",
            data={"total_facts": len(state["key_facts"])},
        )

    def _remove_fact(self, params: Dict) -> SkillResult:
        """Remove a key fact by index."""
        index = params.get("index")
        if index is None:
            return SkillResult(success=False, message="No index provided")

        state = self._load()
        facts = state.get("key_facts", [])

        if index < 0 or index >= len(facts):
            return SkillResult(success=False, message=f"Index {index} out of range (0-{len(facts)-1})")

        removed = facts.pop(index)
        state["key_facts"] = facts
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Removed fact #{index}: {removed.get('text', '')}",
            data={"removed": removed, "remaining_facts": len(facts)},
        )

    def _list_facts(self) -> SkillResult:
        """List all stored key facts."""
        state = self._load()
        facts = state.get("key_facts", [])

        return SkillResult(
            success=True,
            message=f"{len(facts)} key facts stored",
            data={
                "facts": [
                    {"index": i, "text": f.get("text", ""), "added_at": f.get("added_at", ""), "source": f.get("source", "")}
                    for i, f in enumerate(facts)
                ],
                "total": len(facts),
            },
        )

    def _inject(self) -> SkillResult:
        """Get compressed context block for prompt injection."""
        state = self._load()
        facts = state.get("key_facts", [])
        summaries = state.get("compressed_summaries", [])

        parts = []

        # Add key facts section
        if facts:
            fact_lines = [f"- {f.get('text', '')}" for f in facts[-20:]]  # Last 20 facts
            parts.append("## Key Facts & Decisions\n" + "\n".join(fact_lines))

        # Add most recent compressed summary
        if summaries:
            latest = summaries[-1]
            parts.append("## Previous Conversation Summary\n" + latest.get("summary", ""))

        if not parts:
            return SkillResult(
                success=True,
                message="No compressed context available yet",
                data={"context_block": "", "has_context": False},
            )

        context_block = "\n\n".join(parts)
        tokens = self.estimate_tokens(context_block)

        return SkillResult(
            success=True,
            message=f"Compressed context block ready ({tokens} tokens, {len(facts)} facts, {len(summaries)} summaries)",
            data={
                "context_block": context_block,
                "has_context": True,
                "estimated_tokens": tokens,
                "num_facts": len(facts),
                "num_summaries": len(summaries),
            },
        )

    def _stats(self) -> SkillResult:
        """View compression statistics."""
        state = self._load()
        stats = state.get("stats", {})
        facts = state.get("key_facts", [])
        summaries = state.get("compressed_summaries", [])
        settings = state.get("settings", {})

        return SkillResult(
            success=True,
            message=f"Compressions: {stats.get('total_compressions', 0)} | "
                    f"Tokens saved: {stats.get('tokens_saved', 0)} | "
                    f"Facts: {len(facts)} | "
                    f"Messages compressed: {stats.get('messages_compressed', 0)} | "
                    f"LLM: {stats.get('llm_compressions', 0)} | "
                    f"Regex: {stats.get('regex_compressions', 0)}",
            data={
                "stats": stats,
                "settings": settings,
                "key_facts_count": len(facts),
                "summaries_count": len(summaries),
                "llm_available": self.has_llm(),
                "llm_compressions": stats.get("llm_compressions", 0),
                "regex_compressions": stats.get("regex_compressions", 0),
                "created_at": state.get("created_at"),
                "last_updated": state.get("last_updated"),
            },
        )

    def _configure(self, params: Dict) -> SkillResult:
        """Update compression settings."""
        state = self._load()
        settings = state.get("settings", {})
        updated = []

        if "max_tokens" in params:
            val = int(params["max_tokens"])
            if val < 100:
                return SkillResult(success=False, message="max_tokens must be >= 100")
            settings["max_tokens"] = val
            self._max_tokens = val
            updated.append(f"max_tokens={val}")

        if "preserve_recent" in params:
            val = int(params["preserve_recent"])
            if val < 1:
                return SkillResult(success=False, message="preserve_recent must be >= 1")
            settings["preserve_recent"] = val
            self._preserve_recent = val
            updated.append(f"preserve_recent={val}")

        if "max_key_facts" in params:
            val = int(params["max_key_facts"])
            if val < 1:
                return SkillResult(success=False, message="max_key_facts must be >= 1")
            settings["max_key_facts"] = val
            self._max_key_facts = val
            updated.append(f"max_key_facts={val}")

        if not updated:
            return SkillResult(success=False, message="No valid settings provided. Options: max_tokens, preserve_recent, max_key_facts")

        state["settings"] = settings
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Updated settings: {', '.join(updated)}",
            data={"settings": settings},
        )

    def _reset(self) -> SkillResult:
        """Clear all compressed context and facts."""
        state = self._load()
        old_stats = state.get("stats", {})

        new_state = self._default_state()
        # Preserve settings
        new_state["settings"] = state.get("settings", new_state["settings"])
        self._save(new_state)

        return SkillResult(
            success=True,
            message=f"Reset complete. Cleared {old_stats.get('total_compressions', 0)} compressions, "
                    f"{len(state.get('key_facts', []))} facts, "
                    f"{old_stats.get('tokens_saved', 0)} tokens of savings history.",
            data={"previous_stats": old_stats},
        )

    def auto_compress_if_needed(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Auto-compress if conversation exceeds token budget.

        This is designed to be called from the cognition engine's main loop.
        Returns a dict with:
        - compressed: bool - whether compression happened
        - messages: list - the new message list (compressed or original)
        - context_preamble: str - compressed context to prepend
        - tokens_saved: int - tokens saved by compression

        Usage in cognition.py:
            result = compressor.auto_compress_if_needed(self._conversation_history)
            if result['compressed']:
                self._conversation_history = result['messages']
                # Inject context_preamble into system prompt or first message
        """
        state = self._load()
        settings = state.get("settings", {})
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS)
        preserve_recent = settings.get("preserve_recent", DEFAULT_PRESERVE_RECENT)

        total_tokens = self.estimate_message_tokens(messages)

        if total_tokens <= max_tokens:
            return {
                "compressed": False,
                "messages": messages,
                "context_preamble": "",
                "tokens_saved": 0,
            }

        # Need to compress - split messages
        preserve_msgs = min(preserve_recent * 2, len(messages))
        old_messages = messages[:len(messages) - preserve_msgs] if preserve_msgs < len(messages) else []
        recent_messages = messages[len(messages) - preserve_msgs:] if preserve_msgs > 0 else messages

        if not old_messages:
            return {
                "compressed": False,
                "messages": messages,
                "context_preamble": "",
                "tokens_saved": 0,
            }

        # Extract facts from old messages
        new_facts = self.extract_key_information(old_messages)
        existing = set(f.get("text", "") for f in state.get("key_facts", []))
        added = 0
        for fact in new_facts:
            if fact not in existing:
                state.setdefault("key_facts", []).append({
                    "text": fact,
                    "added_at": datetime.now().isoformat(),
                    "source": "auto_compress",
                })
                existing.add(fact)
                added += 1

        max_facts = settings.get("max_key_facts", DEFAULT_MAX_KEY_FACTS)
        if len(state.get("key_facts", [])) > max_facts:
            state["key_facts"] = state["key_facts"][-max_facts:]

        # Create compressed summary
        compressed_summary = self.compress_messages(old_messages)

        # Build context preamble
        preamble_parts = []
        if state.get("key_facts"):
            fact_lines = [f"- {f.get('text', '')}" for f in state["key_facts"][-15:]]
            preamble_parts.append("## Key Context (compressed from earlier)\n" + "\n".join(fact_lines))
        if compressed_summary:
            preamble_parts.append("## Earlier Conversation Summary\n" + compressed_summary)

        context_preamble = "\n\n".join(preamble_parts) if preamble_parts else ""

        # Calculate savings
        original_tokens = self.estimate_message_tokens(old_messages)
        preamble_tokens = self.estimate_tokens(context_preamble)
        tokens_saved = original_tokens - preamble_tokens

        # Store compression record
        state.setdefault("compressed_summaries", []).append({
            "summary": compressed_summary,
            "messages_compressed": len(old_messages),
            "original_tokens": original_tokens,
            "compressed_tokens": preamble_tokens,
            "timestamp": datetime.now().isoformat(),
        })
        if len(state["compressed_summaries"]) > 10:
            state["compressed_summaries"] = state["compressed_summaries"][-10:]

        state.setdefault("compressions", []).append({
            "timestamp": datetime.now().isoformat(),
            "messages_compressed": len(old_messages),
            "tokens_saved": tokens_saved,
            "facts_extracted": added,
            "auto": True,
        })
        if len(state["compressions"]) > MAX_COMPRESSIONS_HISTORY:
            state["compressions"] = state["compressions"][-MAX_COMPRESSIONS_HISTORY:]

        stats = state.get("stats", {})
        stats["total_compressions"] = stats.get("total_compressions", 0) + 1
        stats["tokens_saved"] = stats.get("tokens_saved", 0) + tokens_saved
        stats["facts_extracted"] = stats.get("facts_extracted", 0) + added
        stats["messages_compressed"] = stats.get("messages_compressed", 0) + len(old_messages)
        state["stats"] = stats

        self._save(state)

        return {
            "compressed": True,
            "messages": recent_messages,
            "context_preamble": context_preamble,
            "tokens_saved": tokens_saved,
        }
