"""
ContextManager - Merges multiple context sources into unified project_context.

The agent's LLM prompt includes a `project_context` field that provides the model
with situational awareness. Multiple systems (journal, working memory, action insights,
diagnostics, etc.) each want to inject context into this field. Without a manager,
they would overwrite each other.

ContextManager solves this by:
1. Allowing multiple ContextProviders to register
2. Each provider contributes a labeled section
3. Sections are merged by priority (lower number = appears first)
4. Total context can be capped to stay within token budgets
5. Providers can be enabled/disabled dynamically

Usage:
    ctx = ContextManager(max_chars=4000)
    ctx.register(StaticProvider("goals", "Build revenue system", priority=10))
    ctx.register(my_journal_provider)  # implements ContextProvider
    
    # In the agent loop:
    state = AgentState(project_context=ctx.get_merged_context())
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


class ContextProvider(ABC):
    """Base class for context providers.
    
    Each provider contributes a named section to the agent's context.
    Implement get_context() to return the current context string.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this provider."""
        ...

    @property
    def priority(self) -> int:
        """Sort priority. Lower = appears first in merged context. Default: 50."""
        return 50

    @property
    def enabled(self) -> bool:
        """Whether this provider is active. Default: True."""
        return True

    @abstractmethod
    def get_context(self) -> str:
        """Return current context string. Empty string = skip this section."""
        ...


class StaticProvider(ContextProvider):
    """Simple provider that returns a fixed string."""

    def __init__(self, name: str, content: str, priority: int = 50):
        self._name = name
        self._content = content
        self._priority = priority
        self._enabled = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_context(self) -> str:
        return self._content

    def set_content(self, content: str):
        self._content = content

    def set_enabled(self, enabled: bool):
        self._enabled = enabled


class CallableProvider(ContextProvider):
    """Provider that calls a function to get context."""

    def __init__(self, name: str, fn: Callable[[], str], priority: int = 50):
        self._name = name
        self._fn = fn
        self._priority = priority
        self._enabled = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_context(self) -> str:
        try:
            return self._fn()
        except Exception:
            return ""

    def set_enabled(self, enabled: bool):
        self._enabled = enabled


class ContextManager:
    """Merges multiple context sources into a unified project_context string.
    
    Providers are registered with a name and priority. When get_merged_context()
    is called, all enabled providers are queried, their outputs are labeled
    and concatenated in priority order, and the result is optionally truncated
    to max_chars.
    
    Args:
        max_chars: Maximum total characters for merged context. 0 = unlimited.
        separator: String between sections.
    """

    def __init__(self, max_chars: int = 0, separator: str = "\n\n"):
        self._providers: Dict[str, ContextProvider] = {}
        self._max_chars = max_chars
        self._separator = separator

    def register(self, provider: ContextProvider) -> None:
        """Register a context provider. Replaces any existing provider with same name."""
        self._providers[provider.name] = provider

    def unregister(self, name: str) -> bool:
        """Remove a provider by name. Returns True if it was found."""
        if name in self._providers:
            del self._providers[name]
            return True
        return False

    def get_provider(self, name: str) -> Optional[ContextProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> List[Dict]:
        """List all registered providers with their status."""
        result = []
        for p in self._providers.values():
            result.append({
                "name": p.name,
                "priority": p.priority,
                "enabled": p.enabled,
                "type": type(p).__name__,
            })
        return sorted(result, key=lambda x: x["priority"])

    def get_merged_context(self) -> str:
        """Merge all enabled providers into a single context string.
        
        Each provider's output is wrapped with a section header:
            === <Provider Name> ===
            <content>
        
        Providers are sorted by priority (lowest first).
        Empty outputs are skipped.
        Result is truncated to max_chars if set.
        """
        # Collect sections from enabled providers, sorted by priority
        sorted_providers = sorted(
            self._providers.values(),
            key=lambda p: p.priority
        )

        sections = []
        for provider in sorted_providers:
            if not provider.enabled:
                continue
            try:
                content = provider.get_context()
            except Exception:
                continue

            if not content or not content.strip():
                continue

            section = f"=== {provider.name} ===\n{content.strip()}"
            sections.append(section)

        merged = self._separator.join(sections)

        # Truncate if needed
        if self._max_chars > 0 and len(merged) > self._max_chars:
            merged = merged[:self._max_chars - 3] + "..."

        return merged

    @property
    def provider_count(self) -> int:
        """Number of registered providers."""
        return len(self._providers)

    @property
    def active_provider_count(self) -> int:
        """Number of enabled providers."""
        return sum(1 for p in self._providers.values() if p.enabled)

    def clear(self) -> None:
        """Remove all providers."""
        self._providers.clear()
