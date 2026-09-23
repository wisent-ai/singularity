""Allow running singularity as a module: python -m singularity"""
import asyncio
import sys

from .autonomous_agent import AutonomousAgent


def main():
    """CLI entry point for singularity."""
    agent = AutonomousAgent(
        name="Singularity",
        ticker="SING",
        starting_balance=10.0,
    )
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\
Agent stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
