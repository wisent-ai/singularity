""Basic tests for AutonomousAgent."""
import pytest
from singularity import AutonomousAgent


def test_agent_init():
    """Test agent can be initialized with defaults."""
    agent = AutonomousAgent(name="Test", ticker="TST", starting_balance=1.0)
    assert agent.name == "Test"
    assert agent.ticker == "TST"
    assert agent.balance == 1.0


def test_agent_zero_balance():
    """Test agent with zero balance doesn't run."""
    agent = AutonomousAgent(name="Test", ticker="TST", starting_balance=0)
    assert agent.balance == 0
