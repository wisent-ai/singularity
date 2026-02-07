"""
Singularity Service API - REST interface for agent skills and task management.

Enables external clients to submit tasks, execute skill actions, and monitor
agent status. This is the entry point for revenue generation.
"""

from .server import create_app

__all__ = ["create_app"]
