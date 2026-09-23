#!/usr/bin/env python3
"""
Agent State Persistence - Save and restore agent state across restarts.

Allows an agent to resume from where it left off after a crash or restart.
State is saved to a JSON file after each cycle and loaded on startup.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class StatePersistence:
    """
    Persists agent state to disk for crash recovery and session continuity.
    
    Saves: recent_actions, cycle count, balance, cost tracking, created_resources,
    and any custom state the agent wants to persist.
    
    Uses atomic writes (write to temp, then rename) to prevent corruption
    from crashes during save.
    """

    def __init__(self, state_dir: str = "", agent_name: str = "agent"):
        """
        Initialize state persistence.
        
        Args:
            state_dir: Directory to store state files. Defaults to ./singularity/data/
            agent_name: Agent name used in the state filename
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            self.state_dir = Path(__file__).parent / "data"
        
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize agent name for filename
        safe_name = agent_name.lower().replace(" ", "_").replace("/", "_")
        self.state_file = self.state_dir / f"agent_state_{safe_name}.json"
        self.backup_file = self.state_dir / f"agent_state_{safe_name}.backup.json"
        
        self._last_save_time = 0.0
        self._min_save_interval = 1.0  # Don't save more often than every second

    def save(self, state: Dict[str, Any]) -> bool:
        """
        Save agent state to disk atomically.
        
        Args:
            state: Dictionary of state to persist
            
        Returns:
            True if save succeeded
        """
        try:
            # Add metadata
            state["_persisted_at"] = datetime.now().isoformat()
            state["_version"] = 1
            
            # Atomic write: write to temp file, then rename
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
            
            # Backup previous state before overwriting
            if self.state_file.exists():
                try:
                    self.state_file.rename(self.backup_file)
                except OSError:
                    # On Windows, rename fails if target exists
                    if self.backup_file.exists():
                        self.backup_file.unlink()
                    self.state_file.rename(self.backup_file)
            
            # Move temp to actual
            temp_file.rename(self.state_file)
            self._last_save_time = time.time()
            return True
            
        except Exception as e:
            print(f"[STATE] Failed to save state: {e}")
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load agent state from disk.
        
        Falls back to backup file if primary is corrupted.
        
        Returns:
            State dictionary, or None if no state exists
        """
        # Try primary file first
        state = self._load_file(self.state_file)
        if state is not None:
            return state
        
        # Try backup
        state = self._load_file(self.backup_file)
        if state is not None:
            print("[STATE] Loaded from backup file")
            return state
        
        return None

    def _load_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load and validate a state file."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                state = json.load(f)
            if isinstance(state, dict):
                return state
        except (json.JSONDecodeError, OSError) as e:
            print(f"[STATE] Failed to load {path}: {e}")
        return None

    def should_save(self) -> bool:
        """Check if enough time has passed since last save."""
        return (time.time() - self._last_save_time) >= self._min_save_interval

    def clear(self) -> bool:
        """Delete persisted state."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            if self.backup_file.exists():
                self.backup_file.unlink()
            return True
        except OSError:
            return False

    def exists(self) -> bool:
        """Check if persisted state exists."""
        return self.state_file.exists() or self.backup_file.exists()

    def get_persisted_at(self) -> Optional[str]:
        """Get timestamp of last persisted state."""
        state = self.load()
        if state:
            return state.get("_persisted_at")
        return None


def extract_agent_state(agent) -> Dict[str, Any]:
    """
    Extract persistable state from an AutonomousAgent instance.
    
    Args:
        agent: AutonomousAgent instance
        
    Returns:
        Dictionary of state that can be serialized to JSON
    """
    # Truncate recent_actions results to prevent huge state files
    recent_actions = []
    for action in agent.recent_actions:
        cleaned = {
            "cycle": action.get("cycle"),
            "tool": action.get("tool"),
            "params": action.get("params", {}),
            "result_status": action.get("result", {}).get("status", "unknown"),
            "result_message": str(action.get("result", {}).get("message", ""))[:500],
            "api_cost_usd": action.get("api_cost_usd", 0),
            "tokens": action.get("tokens", 0),
        }
        recent_actions.append(cleaned)

    return {
        "name": agent.name,
        "ticker": agent.ticker,
        "agent_type": agent.agent_type,
        "specialty": agent.specialty,
        "balance": agent.balance,
        "cycle": agent.cycle,
        "total_api_cost": agent.total_api_cost,
        "total_instance_cost": agent.total_instance_cost,
        "total_tokens_used": agent.total_tokens_used,
        "recent_actions": recent_actions[-50:],  # Keep last 50
        "created_resources": agent.created_resources,
    }


def restore_agent_state(agent, state: Dict[str, Any]) -> None:
    """
    Restore persisted state into an AutonomousAgent instance.
    
    Only restores fields that are safe to restore (won't override
    credentials, skills, or cognition config).
    
    Args:
        agent: AutonomousAgent instance  
        state: Previously saved state dictionary
    """
    # Only restore if this is the same agent
    if state.get("name") != agent.name or state.get("ticker") != agent.ticker:
        print(f"[STATE] State mismatch: saved={state.get('name')}/{state.get('ticker')}, "
              f"current={agent.name}/{agent.ticker}. Skipping restore.")
        return

    # Restore balance and costs
    if "balance" in state:
        agent.balance = state["balance"]
    if "cycle" in state:
        agent.cycle = state["cycle"]
    if "total_api_cost" in state:
        agent.total_api_cost = state["total_api_cost"]
    if "total_instance_cost" in state:
        agent.total_instance_cost = state["total_instance_cost"]
    if "total_tokens_used" in state:
        agent.total_tokens_used = state["total_tokens_used"]

    # Restore recent actions
    if "recent_actions" in state:
        agent.recent_actions = state["recent_actions"]

    # Restore created resources
    if "created_resources" in state:
        agent.created_resources = state["created_resources"]

    persisted_at = state.get("_persisted_at", "unknown")
    print(f"[STATE] Restored state from {persisted_at} (cycle {agent.cycle}, ${agent.balance:.4f})")
