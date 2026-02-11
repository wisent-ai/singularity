"""
Browser Skill Package — Backward compatibility shim.

Several skills (account_creator, resend) import BrowserSkill from this
module.  The actual implementation lives in browser_use; this file
re-exports it under the legacy name so those imports succeed.
"""

try:
    from ..browser_use import BrowserUseSkill as BrowserSkill
except Exception:
    BrowserSkill = None  # type: ignore[assignment,misc]

__all__ = ["BrowserSkill"]
