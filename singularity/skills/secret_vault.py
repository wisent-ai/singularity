#!/usr/bin/env python3
"""
Secret Vault Skill - Secure credential and API key management.

This is critical cross-pillar infrastructure that enables the agent to
securely store, retrieve, rotate, and propagate credentials:

- Revenue: Store customer API keys, Stripe tokens, payment credentials
- Replication: Securely propagate credentials to replica agents
- Self-Improvement: Know which API keys are available to unlock capabilities
- Goal Setting: Assess available credentials to set achievable goals

Capabilities:
- Store secrets with encryption (Fernet symmetric encryption)
- Retrieve secrets by name with access logging
- Rotate secrets (update with old value tracking)
- List available secrets (names only, not values)
- Tag and categorize secrets (api_key, token, password, etc.)
- Set expiration dates and get expiry warnings
- Export credential manifest for replica provisioning
- Audit trail of all secret access
- Scoped access: secrets can be restricted to specific skills

Architecture:
  SecretVault uses Fernet encryption (from cryptography lib) when available,
  falling back to base64 obfuscation + file permissions when not.
  Secrets are stored in a JSON file with encrypted values.
  The encryption key is derived from a master password or auto-generated.

Part of ALL four pillars - foundational infrastructure for secure operations.
"""

import base64
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from .base import Skill, SkillResult, SkillManifest, SkillAction

# Try to import cryptography for real encryption
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

VAULT_FILE = Path(__file__).parent.parent / "data" / "secret_vault.json"
VAULT_KEY_FILE = Path(__file__).parent.parent / "data" / ".vault_key"
AUDIT_FILE = Path(__file__).parent.parent / "data" / "vault_audit.json"
MAX_SECRETS = 500
MAX_AUDIT_ENTRIES = 5000

# Secret categories
SECRET_CATEGORIES = [
    "api_key", "token", "password", "certificate",
    "ssh_key", "webhook_secret", "encryption_key", "other"
]


def _generate_vault_key() -> bytes:
    """Generate a new Fernet encryption key."""
    if HAS_CRYPTO:
        return Fernet.generate_key()
    # Fallback: generate a base64-encoded random key
    return base64.urlsafe_b64encode(os.urandom(32))


def _derive_key_from_password(password: str) -> bytes:
    """Derive an encryption key from a master password."""
    # Use PBKDF2-like derivation via hashlib
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), b'singularity-vault-salt', 100000)
    return base64.urlsafe_b64encode(dk)


class VaultEncryption:
    """Handles encryption/decryption of secret values."""

    def __init__(self, key: bytes):
        self._key = key
        if HAS_CRYPTO:
            self._fernet = Fernet(key)
        else:
            self._fernet = None

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string, return base64-encoded ciphertext."""
        if self._fernet:
            return self._fernet.encrypt(plaintext.encode()).decode()
        # Fallback: XOR with key-derived pad + base64
        return self._obfuscate(plaintext)

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string, return plaintext."""
        if self._fernet:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        return self._deobfuscate(ciphertext)

    def _obfuscate(self, plaintext: str) -> str:
        """Simple obfuscation fallback when cryptography not available."""
        key_bytes = self._key
        data = plaintext.encode()
        result = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data))
        return base64.urlsafe_b64encode(result).decode()

    def _deobfuscate(self, ciphertext: str) -> str:
        """Reverse obfuscation."""
        key_bytes = self._key
        data = base64.urlsafe_b64decode(ciphertext.encode())
        result = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data))
        return result.decode()


class SecretVaultSkill(Skill):
    """
    Secure credential management for the autonomous agent.

    Enables the agent to securely store and manage API keys, tokens,
    passwords, and other sensitive credentials. Supports encryption,
    access control, expiration, rotation, and audit logging.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._master_password = (credentials or {}).get("VAULT_MASTER_PASSWORD", "")
        self._encryption: Optional[VaultEncryption] = None
        self._ensure_data()

    def _ensure_data(self):
        """Initialize vault storage and encryption."""
        VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or load encryption key
        if self._master_password:
            key = _derive_key_from_password(self._master_password)
        elif VAULT_KEY_FILE.exists():
            key = VAULT_KEY_FILE.read_bytes().strip()
        else:
            key = _generate_vault_key()
            VAULT_KEY_FILE.write_bytes(key)
            # Restrict permissions on key file
            try:
                os.chmod(str(VAULT_KEY_FILE), 0o600)
            except OSError:
                pass

        self._encryption = VaultEncryption(key)

        if not VAULT_FILE.exists():
            self._save_vault(self._default_vault())
        if not AUDIT_FILE.exists():
            self._save_audit([])

    def _default_vault(self) -> Dict:
        return {
            "secrets": {},
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "encryption": "fernet" if HAS_CRYPTO else "obfuscation",
                "total_stored": 0,
                "total_accessed": 0,
            }
        }

    def _load_vault(self) -> Dict:
        try:
            return json.loads(VAULT_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return self._default_vault()

    def _save_vault(self, data: Dict):
        VAULT_FILE.write_text(json.dumps(data, indent=2))
        try:
            os.chmod(str(VAULT_FILE), 0o600)
        except OSError:
            pass

    def _load_audit(self) -> List[Dict]:
        try:
            return json.loads(AUDIT_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_audit(self, entries: List[Dict]):
        # Trim to max entries
        if len(entries) > MAX_AUDIT_ENTRIES:
            entries = entries[-MAX_AUDIT_ENTRIES:]
        AUDIT_FILE.write_text(json.dumps(entries, indent=2))

    def _log_access(self, action: str, secret_name: str, accessor: str = "agent", success: bool = True):
        """Log a vault access event."""
        entries = self._load_audit()
        entries.append({
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "secret_name": secret_name,
            "accessor": accessor,
            "success": success,
        })
        self._save_audit(entries)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="secret_vault",
            name="Secret Vault",
            version="1.0.0",
            category="infrastructure",
            description="Secure credential and API key management with encryption, rotation, and audit logging",
            required_credentials=[],  # No external creds needed
            actions=[
                SkillAction(
                    name="store_secret",
                    description="Store a new secret in the vault with encryption",
                    parameters={
                        "name": {"type": "str", "required": True, "description": "Unique name/identifier for the secret"},
                        "value": {"type": "str", "required": True, "description": "The secret value to store"},
                        "category": {"type": "str", "required": False, "description": f"Category: {', '.join(SECRET_CATEGORIES)}"},
                        "description": {"type": "str", "required": False, "description": "Human-readable description"},
                        "allowed_skills": {"type": "list", "required": False, "description": "List of skill IDs that can access this secret"},
                        "expires_in_days": {"type": "int", "required": False, "description": "Auto-expire after N days"},
                        "tags": {"type": "list", "required": False, "description": "Tags for organizing secrets"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_secret",
                    description="Retrieve a secret value from the vault",
                    parameters={
                        "name": {"type": "str", "required": True, "description": "Name of the secret to retrieve"},
                        "accessor": {"type": "str", "required": False, "description": "ID of the skill/entity requesting access"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="rotate_secret",
                    description="Update a secret with a new value, keeping audit trail",
                    parameters={
                        "name": {"type": "str", "required": True, "description": "Name of the secret to rotate"},
                        "new_value": {"type": "str", "required": True, "description": "The new secret value"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="delete_secret",
                    description="Remove a secret from the vault",
                    parameters={
                        "name": {"type": "str", "required": True, "description": "Name of the secret to delete"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_secrets",
                    description="List all stored secrets (names, categories, expiry - no values)",
                    parameters={
                        "category": {"type": "str", "required": False, "description": "Filter by category"},
                        "tag": {"type": "str", "required": False, "description": "Filter by tag"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="check_expiring",
                    description="Get secrets expiring within N days",
                    parameters={
                        "within_days": {"type": "int", "required": False, "description": "Days threshold (default: 7)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="export_manifest",
                    description="Export a credential manifest for replica provisioning (names and metadata only)",
                    parameters={
                        "include_tags": {"type": "list", "required": False, "description": "Only include secrets with these tags"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="audit_log",
                    description="View recent vault access audit entries",
                    parameters={
                        "limit": {"type": "int", "required": False, "description": "Max entries to return (default: 20)"},
                        "secret_name": {"type": "str", "required": False, "description": "Filter by secret name"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="vault_status",
                    description="Get overall vault health: encryption type, secret count, expiring soon, access stats",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
        )

    async def execute(self, action: str, params: Dict[str, Any]) -> SkillResult:
        """Execute a vault action."""
        handlers = {
            "store_secret": self._store_secret,
            "get_secret": self._get_secret,
            "rotate_secret": self._rotate_secret,
            "delete_secret": self._delete_secret,
            "list_secrets": self._list_secrets,
            "check_expiring": self._check_expiring,
            "export_manifest": self._export_manifest,
            "audit_log": self._audit_log,
            "vault_status": self._vault_status,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Vault error: {str(e)}")

    async def _store_secret(self, params: Dict) -> SkillResult:
        """Store a new secret."""
        name = params.get("name", "").strip()
        value = params.get("value", "")
        category = params.get("category", "other")
        description = params.get("description", "")
        allowed_skills = params.get("allowed_skills", [])
        expires_in_days = params.get("expires_in_days")
        tags = params.get("tags", [])

        if not name:
            return SkillResult(success=False, message="Secret name is required")
        if not value:
            return SkillResult(success=False, message="Secret value is required")
        if category not in SECRET_CATEGORIES:
            return SkillResult(success=False, message=f"Invalid category. Use: {', '.join(SECRET_CATEGORIES)}")

        vault = self._load_vault()
        if len(vault["secrets"]) >= MAX_SECRETS:
            return SkillResult(success=False, message=f"Vault full: max {MAX_SECRETS} secrets")

        if name in vault["secrets"]:
            return SkillResult(success=False, message=f"Secret '{name}' already exists. Use rotate_secret to update.")

        # Encrypt the value
        encrypted = self._encryption.encrypt(value)

        # Calculate expiry
        expires_at = None
        if expires_in_days and expires_in_days > 0:
            expires_at = (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat()

        vault["secrets"][name] = {
            "encrypted_value": encrypted,
            "category": category,
            "description": description,
            "allowed_skills": allowed_skills,
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at,
            "rotation_count": 0,
            "access_count": 0,
            "last_accessed": None,
        }
        vault["metadata"]["total_stored"] = len(vault["secrets"])
        self._save_vault(vault)
        self._log_access("store", name)

        return SkillResult(
            success=True,
            message=f"Secret '{name}' stored securely ({category})",
            data={
                "name": name,
                "category": category,
                "encryption": "fernet" if HAS_CRYPTO else "obfuscation",
                "expires_at": expires_at,
                "has_access_control": bool(allowed_skills),
            }
        )

    async def _get_secret(self, params: Dict) -> SkillResult:
        """Retrieve a secret value."""
        name = params.get("name", "").strip()
        accessor = params.get("accessor", "agent")

        if not name:
            return SkillResult(success=False, message="Secret name is required")

        vault = self._load_vault()
        secret = vault["secrets"].get(name)
        if not secret:
            self._log_access("get", name, accessor, success=False)
            return SkillResult(success=False, message=f"Secret '{name}' not found")

        # Check access control
        allowed = secret.get("allowed_skills", [])
        if allowed and accessor not in allowed and accessor != "agent":
            self._log_access("get_denied", name, accessor, success=False)
            return SkillResult(success=False, message=f"Access denied: '{accessor}' not authorized for '{name}'")

        # Check expiry
        if secret.get("expires_at"):
            expires = datetime.fromisoformat(secret["expires_at"])
            if datetime.utcnow() > expires:
                self._log_access("get_expired", name, accessor, success=False)
                return SkillResult(success=False, message=f"Secret '{name}' has expired")

        # Decrypt
        try:
            plaintext = self._encryption.decrypt(secret["encrypted_value"])
        except Exception as e:
            self._log_access("get_decrypt_fail", name, accessor, success=False)
            return SkillResult(success=False, message=f"Failed to decrypt secret: {str(e)}")

        # Update access stats
        secret["access_count"] = secret.get("access_count", 0) + 1
        secret["last_accessed"] = datetime.utcnow().isoformat()
        vault["metadata"]["total_accessed"] = vault["metadata"].get("total_accessed", 0) + 1
        self._save_vault(vault)
        self._log_access("get", name, accessor)

        return SkillResult(
            success=True,
            message=f"Secret '{name}' retrieved",
            data={
                "name": name,
                "value": plaintext,
                "category": secret["category"],
                "access_count": secret["access_count"],
            }
        )

    async def _rotate_secret(self, params: Dict) -> SkillResult:
        """Rotate a secret with a new value."""
        name = params.get("name", "").strip()
        new_value = params.get("new_value", "")

        if not name:
            return SkillResult(success=False, message="Secret name is required")
        if not new_value:
            return SkillResult(success=False, message="New value is required")

        vault = self._load_vault()
        secret = vault["secrets"].get(name)
        if not secret:
            return SkillResult(success=False, message=f"Secret '{name}' not found")

        # Encrypt new value
        encrypted = self._encryption.encrypt(new_value)
        secret["encrypted_value"] = encrypted
        secret["updated_at"] = datetime.utcnow().isoformat()
        secret["rotation_count"] = secret.get("rotation_count", 0) + 1

        self._save_vault(vault)
        self._log_access("rotate", name)

        return SkillResult(
            success=True,
            message=f"Secret '{name}' rotated (rotation #{secret['rotation_count']})",
            data={
                "name": name,
                "rotation_count": secret["rotation_count"],
                "updated_at": secret["updated_at"],
            }
        )

    async def _delete_secret(self, params: Dict) -> SkillResult:
        """Delete a secret from the vault."""
        name = params.get("name", "").strip()
        if not name:
            return SkillResult(success=False, message="Secret name is required")

        vault = self._load_vault()
        if name not in vault["secrets"]:
            return SkillResult(success=False, message=f"Secret '{name}' not found")

        del vault["secrets"][name]
        vault["metadata"]["total_stored"] = len(vault["secrets"])
        self._save_vault(vault)
        self._log_access("delete", name)

        return SkillResult(
            success=True,
            message=f"Secret '{name}' deleted from vault",
            data={"name": name}
        )

    async def _list_secrets(self, params: Dict) -> SkillResult:
        """List secrets (metadata only, no values)."""
        category_filter = params.get("category")
        tag_filter = params.get("tag")

        vault = self._load_vault()
        secrets_list = []

        for name, secret in vault["secrets"].items():
            if category_filter and secret.get("category") != category_filter:
                continue
            if tag_filter and tag_filter not in secret.get("tags", []):
                continue

            # Check if expired
            is_expired = False
            if secret.get("expires_at"):
                is_expired = datetime.utcnow() > datetime.fromisoformat(secret["expires_at"])

            secrets_list.append({
                "name": name,
                "category": secret.get("category", "other"),
                "description": secret.get("description", ""),
                "tags": secret.get("tags", []),
                "created_at": secret.get("created_at"),
                "updated_at": secret.get("updated_at"),
                "expires_at": secret.get("expires_at"),
                "is_expired": is_expired,
                "rotation_count": secret.get("rotation_count", 0),
                "access_count": secret.get("access_count", 0),
                "has_access_control": bool(secret.get("allowed_skills")),
            })

        return SkillResult(
            success=True,
            message=f"Found {len(secrets_list)} secrets",
            data={
                "secrets": secrets_list,
                "total": len(secrets_list),
                "categories": list(set(s["category"] for s in secrets_list)),
            }
        )

    async def _check_expiring(self, params: Dict) -> SkillResult:
        """Find secrets expiring soon."""
        within_days = params.get("within_days", 7)
        threshold = datetime.utcnow() + timedelta(days=within_days)

        vault = self._load_vault()
        expiring = []
        expired = []

        for name, secret in vault["secrets"].items():
            if not secret.get("expires_at"):
                continue
            exp_date = datetime.fromisoformat(secret["expires_at"])
            if exp_date < datetime.utcnow():
                expired.append({
                    "name": name,
                    "category": secret.get("category"),
                    "expired_at": secret["expires_at"],
                    "days_ago": (datetime.utcnow() - exp_date).days,
                })
            elif exp_date < threshold:
                expiring.append({
                    "name": name,
                    "category": secret.get("category"),
                    "expires_at": secret["expires_at"],
                    "days_remaining": (exp_date - datetime.utcnow()).days,
                })

        return SkillResult(
            success=True,
            message=f"{len(expiring)} expiring soon, {len(expired)} already expired",
            data={
                "expiring_soon": expiring,
                "already_expired": expired,
                "threshold_days": within_days,
            }
        )

    async def _export_manifest(self, params: Dict) -> SkillResult:
        """Export credential manifest for replica provisioning."""
        include_tags = params.get("include_tags", [])
        vault = self._load_vault()

        manifest_entries = []
        for name, secret in vault["secrets"].items():
            if include_tags:
                if not any(t in secret.get("tags", []) for t in include_tags):
                    continue

            # Check if expired
            is_expired = False
            if secret.get("expires_at"):
                is_expired = datetime.utcnow() > datetime.fromisoformat(secret["expires_at"])

            manifest_entries.append({
                "name": name,
                "category": secret.get("category", "other"),
                "description": secret.get("description", ""),
                "tags": secret.get("tags", []),
                "has_expiry": bool(secret.get("expires_at")),
                "is_expired": is_expired,
                "allowed_skills": secret.get("allowed_skills", []),
                # NO values exported - replica must request individually
            })

        self._log_access("export_manifest", "*")

        return SkillResult(
            success=True,
            message=f"Credential manifest: {len(manifest_entries)} secrets available",
            data={
                "manifest": manifest_entries,
                "total": len(manifest_entries),
                "encryption_type": "fernet" if HAS_CRYPTO else "obfuscation",
                "note": "Values not included. Replicas must request secrets individually with proper access.",
            }
        )

    async def _audit_log(self, params: Dict) -> SkillResult:
        """View audit log entries."""
        limit = min(params.get("limit", 20), 100)
        secret_name = params.get("secret_name")

        entries = self._load_audit()

        if secret_name:
            entries = [e for e in entries if e.get("secret_name") == secret_name]

        recent = entries[-limit:]
        recent.reverse()  # Most recent first

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} audit entries",
            data={
                "entries": recent,
                "total_entries": len(entries),
            }
        )

    async def _vault_status(self, params: Dict) -> SkillResult:
        """Get overall vault health status."""
        vault = self._load_vault()
        entries = self._load_audit()

        # Count by category
        category_counts = {}
        expired_count = 0
        expiring_soon = 0
        threshold = datetime.utcnow() + timedelta(days=7)

        for name, secret in vault["secrets"].items():
            cat = secret.get("category", "other")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if secret.get("expires_at"):
                exp_date = datetime.fromisoformat(secret["expires_at"])
                if exp_date < datetime.utcnow():
                    expired_count += 1
                elif exp_date < threshold:
                    expiring_soon += 1

        # Recent access stats
        recent_accesses = [e for e in entries[-100:] if e.get("action") == "get"]
        denied_accesses = [e for e in entries[-100:] if "denied" in e.get("action", "")]

        return SkillResult(
            success=True,
            message="Vault status retrieved",
            data={
                "total_secrets": len(vault["secrets"]),
                "encryption": "fernet" if HAS_CRYPTO else "obfuscation",
                "categories": category_counts,
                "expired": expired_count,
                "expiring_within_7_days": expiring_soon,
                "total_accesses": vault["metadata"].get("total_accessed", 0),
                "recent_access_count": len(recent_accesses),
                "recent_denied_count": len(denied_accesses),
                "audit_entries": len(entries),
                "vault_version": vault["metadata"].get("version", "1.0"),
            }
        )
