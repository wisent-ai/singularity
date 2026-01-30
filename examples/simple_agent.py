#!/usr/bin/env python3
"""
Simple WisentBot Agent Example

This example shows how to create a basic autonomous agent
that can browse the web and generate content.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from wisentbot import AutonomousAgent


async def main():
    # Create an agent with a $5 budget
    agent = AutonomousAgent(
        name="ContentBot",
        ticker="CBOT",
        agent_type="content",
        specialty="writing blog posts and social media content",
        starting_balance=5.0,  # $5 USD budget
        cycle_interval_seconds=10.0,  # 10 seconds between actions
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
    )

    print(f"Starting {agent.name} with ${agent.balance} budget...")
    print("Press Ctrl+C to stop\n")

    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\nStopping agent...")
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
