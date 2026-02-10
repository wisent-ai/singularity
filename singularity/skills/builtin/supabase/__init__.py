"""
Supabase Skill Package

Re-exports SupabaseSkill for backward compatibility.
"""

from .skill import SupabaseSkill, setup_supabase_auth_project

__all__ = ["SupabaseSkill", "setup_supabase_auth_project"]
