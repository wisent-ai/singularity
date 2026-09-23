"""
RuntimeMetrics - Lightweight performance metrics for the agent loop.

Tracks decision latency, execution latency, success/failure rates,
token usage trends, and throughput. Provides summaries that can be
injected into agent context for self-aware decision making.

Part of the Self-Improvement pillar: the agent can observe its own
performance characteristics and adapt accordingly.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricPoint:
    """A single metric observation."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class RuntimeMetrics:
    """
    Collects and summarizes agent runtime metrics.
    
    Keeps a sliding window of recent observations (default: last 100)
    to compute running statistics without unbounded memory growth.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._start_time = time.time()
        
        # Sliding windows for key metrics
        self._decision_latencies: deque = deque(maxlen=window_size)
        self._execution_latencies: deque = deque(maxlen=window_size)
        self._action_results: deque = deque(maxlen=window_size)
        self._token_counts: deque = deque(maxlen=window_size)
        self._api_costs: deque = deque(maxlen=window_size)
        self._cycle_times: deque = deque(maxlen=window_size)
        
        # Counters (cumulative)
        self.total_cycles = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_errors = 0
        self.total_tokens = 0
        self.total_api_cost = 0.0
        
        # Tool-specific tracking
        self._tool_counts: Dict[str, int] = {}
        self._tool_failures: Dict[str, int] = {}
        
        # Timing context managers
        self._pending_timers: Dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._pending_timers[name] = time.time()

    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a named timer and return elapsed seconds."""
        start = self._pending_timers.pop(name, None)
        if start is None:
            return None
        elapsed = time.time() - start
        
        if name == "decision":
            self._decision_latencies.append(elapsed)
        elif name == "execution":
            self._execution_latencies.append(elapsed)
        elif name == "cycle":
            self._cycle_times.append(elapsed)
        
        return elapsed

    def record_decision(self, latency: float, tokens: int, api_cost: float) -> None:
        """Record a decision (LLM call) metric."""
        self._decision_latencies.append(latency)
        self._token_counts.append(tokens)
        self._api_costs.append(api_cost)
        self.total_tokens += tokens
        self.total_api_cost += api_cost

    def record_execution(self, tool: str, latency: float, success: bool) -> None:
        """Record an action execution metric."""
        self._execution_latencies.append(latency)
        self.total_cycles += 1
        
        # Track tool usage
        self._tool_counts[tool] = self._tool_counts.get(tool, 0) + 1
        
        if success:
            self.total_successes += 1
            self._action_results.append(1)
        else:
            self.total_failures += 1
            self._action_results.append(0)
            self._tool_failures[tool] = self._tool_failures.get(tool, 0) + 1

    def record_error(self, tool: str) -> None:
        """Record an execution error."""
        self.total_errors += 1
        self._tool_failures[tool] = self._tool_failures.get(tool, 0) + 1

    @property
    def uptime_seconds(self) -> float:
        """Seconds since metrics started."""
        return time.time() - self._start_time

    @property
    def avg_decision_latency(self) -> float:
        """Average decision (LLM) latency in seconds."""
        if not self._decision_latencies:
            return 0.0
        return sum(self._decision_latencies) / len(self._decision_latencies)

    @property
    def avg_execution_latency(self) -> float:
        """Average action execution latency in seconds."""
        if not self._execution_latencies:
            return 0.0
        return sum(self._execution_latencies) / len(self._execution_latencies)

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 1.0
        return self.total_successes / total

    @property
    def recent_success_rate(self) -> float:
        """Success rate over the recent window."""
        if not self._action_results:
            return 1.0
        return sum(self._action_results) / len(self._action_results)

    @property
    def cycles_per_minute(self) -> float:
        """Throughput in cycles per minute."""
        uptime_minutes = self.uptime_seconds / 60
        if uptime_minutes == 0:
            return 0.0
        return self.total_cycles / uptime_minutes

    @property 
    def avg_tokens_per_cycle(self) -> float:
        """Average tokens consumed per decision."""
        if not self._token_counts:
            return 0.0
        return sum(self._token_counts) / len(self._token_counts)

    @property
    def avg_cost_per_cycle(self) -> float:
        """Average API cost per cycle."""
        if not self._api_costs:
            return 0.0
        return sum(self._api_costs) / len(self._api_costs)

    def top_tools(self, n: int = 5) -> List[Dict]:
        """Get the most frequently used tools."""
        sorted_tools = sorted(
            self._tool_counts.items(), key=lambda x: x[1], reverse=True
        )[:n]
        return [
            {
                "tool": tool,
                "count": count,
                "failures": self._tool_failures.get(tool, 0),
                "success_rate": 1.0 - (self._tool_failures.get(tool, 0) / count) if count > 0 else 1.0,
            }
            for tool, count in sorted_tools
        ]

    def failing_tools(self) -> List[Dict]:
        """Get tools with failure rates above 50%."""
        failing = []
        for tool, failures in self._tool_failures.items():
            total = self._tool_counts.get(tool, failures)
            rate = failures / total if total > 0 else 0
            if rate > 0.5:
                failing.append({
                    "tool": tool,
                    "failures": failures,
                    "total": total,
                    "failure_rate": rate,
                })
        return sorted(failing, key=lambda x: x["failure_rate"], reverse=True)

    def summary(self) -> Dict:
        """Get a full metrics summary."""
        uptime_min = self.uptime_seconds / 60
        return {
            "uptime_minutes": round(uptime_min, 1),
            "total_cycles": self.total_cycles,
            "cycles_per_minute": round(self.cycles_per_minute, 2),
            "success_rate": round(self.success_rate, 3),
            "recent_success_rate": round(self.recent_success_rate, 3),
            "avg_decision_latency_s": round(self.avg_decision_latency, 3),
            "avg_execution_latency_s": round(self.avg_execution_latency, 3),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_cycle": round(self.avg_tokens_per_cycle, 0),
            "total_api_cost_usd": round(self.total_api_cost, 6),
            "avg_cost_per_cycle_usd": round(self.avg_cost_per_cycle, 6),
            "total_errors": self.total_errors,
            "top_tools": self.top_tools(),
            "failing_tools": self.failing_tools(),
        }

    def context_summary(self) -> str:
        """
        Generate a concise text summary suitable for LLM context injection.
        
        This lets the agent be self-aware about its performance.
        """
        if self.total_cycles == 0:
            return ""
        
        lines = [f"Performance: {self.total_cycles} cycles in {self.uptime_seconds/60:.1f}min"]
        lines.append(f"  Success rate: {self.success_rate:.0%} (recent: {self.recent_success_rate:.0%})")
        lines.append(f"  Avg latency: decision={self.avg_decision_latency:.1f}s, execution={self.avg_execution_latency:.1f}s")
        lines.append(f"  Cost: ${self.total_api_cost:.4f} total, ${self.avg_cost_per_cycle:.4f}/cycle")
        
        failing = self.failing_tools()
        if failing:
            tools_str = ", ".join(f"{t['tool']}({t['failure_rate']:.0%} fail)" for t in failing[:3])
            lines.append(f"  ⚠ Unreliable tools: {tools_str}")
        
        return "\n".join(lines)
