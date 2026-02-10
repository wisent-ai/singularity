"""
Email Skills Package

Constants and helpers for email domain management.
Re-exports EmailSkill for backward compatibility.
"""

from .skill import EmailSkill

# Namecheap API constants
NAMECHEAP_API_URL = "https://api.namecheap.com/xml.response"
NAMECHEAP_XML_NS = {"ns": "http://api.namecheap.com/xml.response"}
REQUIRED_NAMECHEAP_CREDENTIALS = [
    "NAMECHEAP_API_KEY",
    "NAMECHEAP_API_USER",
    "NAMECHEAP_USERNAME",
    "NAMECHEAP_CLIENT_IP",
]

# Resend API constants
RESEND_API_BASE = "https://api.resend.com"

__all__ = ["EmailSkill"]
