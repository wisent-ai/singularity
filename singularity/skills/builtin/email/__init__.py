"""
Email Skills Package

Constants and helpers for email domain management.
Re-exports EmailSkill for backward compatibility.
"""

# Import constants first (no circular dependency)
from .constants import (
    NAMECHEAP_API_URL,
    NAMECHEAP_XML_NS,
    REQUIRED_NAMECHEAP_CREDENTIALS,
    RESEND_API_BASE,
)

# Now safe to import EmailSkill (which imports provider_helpers,
# which imports constants from .constants instead of from .)
from .skill import EmailSkill

__all__ = [
    "EmailSkill",
    "NAMECHEAP_API_URL",
    "NAMECHEAP_XML_NS",
    "REQUIRED_NAMECHEAP_CREDENTIALS",
    "RESEND_API_BASE",
]
