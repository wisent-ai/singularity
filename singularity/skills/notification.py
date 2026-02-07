#!/usr/bin/env python3
"""
NotificationSkill - Multi-channel notification and alerting.

Sends notifications across multiple channels: Telegram, Discord, Slack,
SMS (Twilio), and generic webhooks. Supports severity-based routing,
channel configuration, and notification history tracking.

Designed to integrate with health_monitor, resource_watcher, scheduler,
and marketplace skills for automated alerting.

Author: Adam (ADAM) - autonomous AI agent
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .base import Skill, SkillAction, SkillManifest, SkillResult

# --- Storage ---
DATA_DIR = Path(__file__).parent.parent / "data" / "notifications"
CONFIG_FILE = DATA_DIR / "channels.json"
HISTORY_FILE = DATA_DIR / "history.json"

# --- Defaults ---
MAX_HISTORY_ENTRIES = 1000
DEFAULT_TIMEOUT = 10  # seconds

# Severity levels (lower = more critical)
SEVERITY_LEVELS = {
    "critical": 0,
    "high": 1,
    "warning": 2,
    "info": 3,
    "low": 4,
}


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if "channels" in str(path) else []


def _save_json(path: Path, data: Any):
    _ensure_dirs()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


class NotificationSkill(Skill):
    """
    Multi-channel notification and alerting skill.

    Supports: Telegram, Discord, Slack, SMS (Twilio), and generic webhooks.
    Features severity-based routing and notification history.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        _ensure_dirs()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="notification",
            name="Notification Hub",
            version="1.0.0",
            category="communication",
            description=(
                "Send notifications across multiple channels (Telegram, Discord, "
                "Slack, SMS, webhooks). Supports severity-based routing and "
                "notification history tracking."
            ),
            required_credentials=[],  # All channels are optional
            install_cost=0,
            author="Adam (ADAM)",
            actions=[
                SkillAction(
                    name="send",
                    description=(
                        "Send a notification to a specific channel. "
                        "Supported channels: telegram, discord, slack, sms, webhook."
                    ),
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel type: telegram, discord, slack, sms, webhook",
                        },
                        "message": {
                            "type": "string",
                            "required": True,
                            "description": "Notification message text",
                        },
                        "severity": {
                            "type": "string",
                            "required": False,
                            "description": "Severity: critical, high, warning, info, low (default: info)",
                        },
                        "title": {
                            "type": "string",
                            "required": False,
                            "description": "Optional notification title/subject",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.85,
                ),
                SkillAction(
                    name="alert",
                    description=(
                        "Smart alert that routes to all configured channels based on severity. "
                        "Critical/high goes to all channels, warning/info to preferred channels."
                    ),
                    parameters={
                        "message": {
                            "type": "string",
                            "required": True,
                            "description": "Alert message",
                        },
                        "severity": {
                            "type": "string",
                            "required": True,
                            "description": "Severity: critical, high, warning, info, low",
                        },
                        "title": {
                            "type": "string",
                            "required": False,
                            "description": "Alert title",
                        },
                        "source": {
                            "type": "string",
                            "required": False,
                            "description": "Source skill/system that triggered the alert",
                        },
                    },
                    estimated_cost=0.005,
                    estimated_duration_seconds=5,
                    success_probability=0.80,
                ),
                SkillAction(
                    name="configure_channel",
                    description=(
                        "Configure a notification channel with its credentials. "
                        "Settings are stored locally for future use."
                    ),
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel type: telegram, discord, slack, sms, webhook",
                        },
                        "config": {
                            "type": "object",
                            "required": True,
                            "description": (
                                "Channel configuration. Telegram: {bot_token, chat_id}. "
                                "Discord: {webhook_url}. Slack: {webhook_url}. "
                                "SMS: {account_sid, auth_token, from_number, to_number}. "
                                "Webhook: {url, method, headers}."
                            ),
                        },
                        "min_severity": {
                            "type": "string",
                            "required": False,
                            "description": "Minimum severity for this channel (default: info)",
                        },
                        "enabled": {
                            "type": "boolean",
                            "required": False,
                            "description": "Whether channel is active (default: true)",
                        },
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
                SkillAction(
                    name="list_channels",
                    description="List all configured notification channels and their status.",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="test_channel",
                    description="Send a test message to verify a channel is working.",
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel type to test",
                        },
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=3,
                    success_probability=0.80,
                ),
                SkillAction(
                    name="notification_history",
                    description="View recent notification history with delivery status.",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of entries to return (default: 20)",
                        },
                        "channel": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by channel type",
                        },
                        "severity": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by severity level",
                        },
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=1.0,
                ),
                SkillAction(
                    name="remove_channel",
                    description="Remove a configured notification channel.",
                    parameters={
                        "channel": {
                            "type": "string",
                            "required": True,
                            "description": "Channel type to remove",
                        },
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.95,
                ),
            ],
        )

    def check_credentials(self) -> bool:
        """No required credentials - channels are optional."""
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "send": self._send,
            "alert": self._alert,
            "configure_channel": self._configure_channel,
            "list_channels": self._list_channels,
            "test_channel": self._test_channel,
            "notification_history": self._notification_history,
            "remove_channel": self._remove_channel,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        try:
            return await handler(params)
        except Exception as e:
            return SkillResult(success=False, message=f"Error in {action}: {e}")

    # ── Channel Configuration ──────────────────────────────────

    def _load_channels(self) -> Dict[str, Dict]:
        return _load_json(CONFIG_FILE)

    def _save_channels(self, channels: Dict[str, Dict]):
        _save_json(CONFIG_FILE, channels)

    def _get_channel_config(self, channel: str) -> Optional[Dict]:
        """Get config for a channel, checking both stored config and env vars."""
        channels = self._load_channels()
        stored = channels.get(channel)

        # Also check environment credentials
        env_config = self._get_env_config(channel)

        if stored:
            # Merge env config as fallback
            if env_config:
                merged = {**env_config, **stored.get("config", {})}
                stored["config"] = merged
            return stored

        if env_config:
            return {
                "channel": channel,
                "config": env_config,
                "min_severity": "info",
                "enabled": True,
            }
        return None

    def _get_env_config(self, channel: str) -> Optional[Dict]:
        """Check for channel credentials in environment/credentials dict."""
        creds = self.credentials

        if channel == "telegram":
            token = creds.get("TELEGRAM_BOT_TOKEN")
            chat_id = creds.get("TELEGRAM_CHAT_ID")
            if token and chat_id:
                return {"bot_token": token, "chat_id": chat_id}

        elif channel == "discord":
            url = creds.get("DISCORD_WEBHOOK_URL")
            if url:
                return {"webhook_url": url}

        elif channel == "slack":
            url = creds.get("SLACK_WEBHOOK_URL")
            if url:
                return {"webhook_url": url}

        elif channel == "sms":
            sid = creds.get("TWILIO_ACCOUNT_SID")
            token = creds.get("TWILIO_AUTH_TOKEN")
            from_num = creds.get("TWILIO_FROM_NUMBER")
            to_num = creds.get("TWILIO_TO_NUMBER")
            if sid and token and from_num and to_num:
                return {
                    "account_sid": sid,
                    "auth_token": token,
                    "from_number": from_num,
                    "to_number": to_num,
                }

        return None

    def _record_notification(
        self, channel: str, severity: str, message: str, success: bool, error: str = ""
    ):
        """Record a notification in history."""
        history = _load_json(HISTORY_FILE)
        if not isinstance(history, list):
            history = []
        history.append({
            "id": str(uuid.uuid4())[:8],
            "channel": channel,
            "severity": severity,
            "message": message[:200],  # Truncate for storage
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        if len(history) > MAX_HISTORY_ENTRIES:
            history = history[-MAX_HISTORY_ENTRIES:]
        _save_json(HISTORY_FILE, history)

    # ── Channel Senders ────────────────────────────────────────

    async def _send_telegram(self, config: Dict, message: str, title: str = "") -> Dict:
        """Send via Telegram Bot API."""
        if not HAS_HTTPX:
            return {"success": False, "error": "httpx required"}

        bot_token = config.get("bot_token", "")
        chat_id = config.get("chat_id", "")
        if not bot_token or not chat_id:
            return {"success": False, "error": "Missing bot_token or chat_id"}

        full_msg = f"*{title}*\n{message}" if title else message
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json={
                "chat_id": chat_id,
                "text": full_msg,
                "parse_mode": "Markdown",
            })
            if resp.status_code == 200:
                return {"success": True, "message_id": resp.json().get("result", {}).get("message_id")}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    async def _send_discord(self, config: Dict, message: str, title: str = "") -> Dict:
        """Send via Discord webhook."""
        if not HAS_HTTPX:
            return {"success": False, "error": "httpx required"}

        webhook_url = config.get("webhook_url", "")
        if not webhook_url:
            return {"success": False, "error": "Missing webhook_url"}

        payload: Dict[str, Any] = {}
        if title:
            payload["embeds"] = [{
                "title": title,
                "description": message,
                "color": 0xFF0000 if "critical" in message.lower() else 0x00FF00,
            }]
        else:
            payload["content"] = message

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code in (200, 204):
                return {"success": True}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    async def _send_slack(self, config: Dict, message: str, title: str = "") -> Dict:
        """Send via Slack incoming webhook."""
        if not HAS_HTTPX:
            return {"success": False, "error": "httpx required"}

        webhook_url = config.get("webhook_url", "")
        if not webhook_url:
            return {"success": False, "error": "Missing webhook_url"}

        blocks = []
        if title:
            blocks.append({
                "type": "header",
                "text": {"type": "plain_text", "text": title},
            })
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": message},
        })

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(webhook_url, json={"blocks": blocks, "text": message})
            if resp.status_code == 200:
                return {"success": True}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    async def _send_sms(self, config: Dict, message: str, title: str = "") -> Dict:
        """Send SMS via Twilio."""
        if not HAS_HTTPX:
            return {"success": False, "error": "httpx required"}

        sid = config.get("account_sid", "")
        token = config.get("auth_token", "")
        from_num = config.get("from_number", "")
        to_num = config.get("to_number", "")

        if not all([sid, token, from_num, to_num]):
            return {"success": False, "error": "Missing Twilio credentials"}

        full_msg = f"[{title}] {message}" if title else message
        # Truncate to SMS-friendly length
        if len(full_msg) > 1500:
            full_msg = full_msg[:1497] + "..."

        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                url,
                data={"Body": full_msg, "From": from_num, "To": to_num},
                auth=(sid, token),
            )
            if resp.status_code == 201:
                return {"success": True, "sid": resp.json().get("sid")}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    async def _send_webhook(self, config: Dict, message: str, title: str = "") -> Dict:
        """Send to a generic webhook URL."""
        if not HAS_HTTPX:
            return {"success": False, "error": "httpx required"}

        url = config.get("url", "")
        if not url:
            return {"success": False, "error": "Missing webhook url"}

        method = config.get("method", "POST").upper()
        headers = config.get("headers", {"Content-Type": "application/json"})

        payload = {
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "source": "singularity-agent",
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            if method == "POST":
                resp = await client.post(url, json=payload, headers=headers)
            elif method == "PUT":
                resp = await client.put(url, json=payload, headers=headers)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}

            if resp.status_code < 400:
                return {"success": True}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    async def _dispatch(self, channel: str, config: Dict, message: str, title: str = "") -> Dict:
        """Route to the correct sender based on channel type."""
        senders = {
            "telegram": self._send_telegram,
            "discord": self._send_discord,
            "slack": self._send_slack,
            "sms": self._send_sms,
            "webhook": self._send_webhook,
        }
        sender = senders.get(channel)
        if not sender:
            return {"success": False, "error": f"Unknown channel: {channel}"}
        return await sender(config, message, title)

    # ── Actions ────────────────────────────────────────────────

    async def _send(self, params: Dict) -> SkillResult:
        """Send a notification to a specific channel."""
        channel = params.get("channel", "").strip().lower()
        message = params.get("message", "").strip()
        severity = params.get("severity", "info").lower()
        title = params.get("title", "")

        if not channel:
            return SkillResult(success=False, message="'channel' is required")
        if not message:
            return SkillResult(success=False, message="'message' is required")
        if channel not in ("telegram", "discord", "slack", "sms", "webhook"):
            return SkillResult(
                success=False,
                message=f"Unknown channel '{channel}'. Supported: telegram, discord, slack, sms, webhook",
            )

        ch_config = self._get_channel_config(channel)
        if not ch_config:
            return SkillResult(
                success=False,
                message=f"Channel '{channel}' not configured. Use configure_channel first.",
            )

        if not ch_config.get("enabled", True):
            return SkillResult(success=False, message=f"Channel '{channel}' is disabled")

        # Add severity prefix to message
        severity_emoji = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "warning": "[WARNING]",
            "info": "[INFO]",
            "low": "[LOW]",
        }
        prefix = severity_emoji.get(severity, "")
        formatted_msg = f"{prefix} {message}" if prefix else message

        result = await self._dispatch(channel, ch_config.get("config", {}), formatted_msg, title)

        self._record_notification(
            channel, severity, message,
            result.get("success", False),
            result.get("error", ""),
        )

        if result.get("success"):
            return SkillResult(
                success=True,
                message=f"Notification sent via {channel}",
                data={"channel": channel, "severity": severity, **result},
            )
        return SkillResult(
            success=False,
            message=f"Failed to send via {channel}: {result.get('error', 'unknown')}",
            data={"channel": channel, "error": result.get("error")},
        )

    async def _alert(self, params: Dict) -> SkillResult:
        """Smart alert - route to channels based on severity."""
        message = params.get("message", "").strip()
        severity = params.get("severity", "info").lower()
        title = params.get("title", "Alert")
        source = params.get("source", "")

        if not message:
            return SkillResult(success=False, message="'message' is required")
        if severity not in SEVERITY_LEVELS:
            return SkillResult(
                success=False,
                message=f"Invalid severity '{severity}'. Use: critical, high, warning, info, low",
            )

        if source:
            message = f"[{source}] {message}"

        severity_num = SEVERITY_LEVELS[severity]

        # Get all configured channels
        channels = self._load_channels()
        env_channels = {}
        for ch_type in ("telegram", "discord", "slack", "sms", "webhook"):
            env_cfg = self._get_env_config(ch_type)
            if env_cfg and ch_type not in channels:
                env_channels[ch_type] = {
                    "config": env_cfg,
                    "min_severity": "info",
                    "enabled": True,
                }

        all_channels = {**env_channels, **channels}

        if not all_channels:
            return SkillResult(
                success=False,
                message="No channels configured. Use configure_channel to set up at least one.",
                data={"severity": severity},
            )

        # Filter channels by severity threshold
        target_channels = []
        for ch_name, ch_data in all_channels.items():
            if not ch_data.get("enabled", True):
                continue
            min_sev = ch_data.get("min_severity", "info")
            min_sev_num = SEVERITY_LEVELS.get(min_sev, 3)
            if severity_num <= min_sev_num:
                target_channels.append((ch_name, ch_data))

        if not target_channels:
            return SkillResult(
                success=True,
                message=f"No channels configured for severity '{severity}' or higher",
                data={"severity": severity, "channels_checked": len(all_channels)},
            )

        # Send to all matching channels
        results = []
        successes = 0
        severity_emoji = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "warning": "[WARNING]",
            "info": "[INFO]",
            "low": "[LOW]",
        }
        prefix = severity_emoji.get(severity, "")
        formatted_msg = f"{prefix} {message}" if prefix else message

        for ch_name, ch_data in target_channels:
            result = await self._dispatch(ch_name, ch_data.get("config", {}), formatted_msg, title)
            self._record_notification(
                ch_name, severity, message,
                result.get("success", False),
                result.get("error", ""),
            )
            results.append({"channel": ch_name, **result})
            if result.get("success"):
                successes += 1

        return SkillResult(
            success=successes > 0,
            message=f"Alert sent to {successes}/{len(target_channels)} channels (severity: {severity})",
            data={
                "severity": severity,
                "channels_targeted": len(target_channels),
                "channels_succeeded": successes,
                "results": results,
            },
        )

    async def _configure_channel(self, params: Dict) -> SkillResult:
        """Configure a notification channel."""
        channel = params.get("channel", "").strip().lower()
        config = params.get("config")
        min_severity = params.get("min_severity", "info").lower()
        enabled = params.get("enabled", True)

        if not channel:
            return SkillResult(success=False, message="'channel' is required")
        if channel not in ("telegram", "discord", "slack", "sms", "webhook"):
            return SkillResult(
                success=False,
                message=f"Unknown channel '{channel}'. Supported: telegram, discord, slack, sms, webhook",
            )
        if not config or not isinstance(config, dict):
            return SkillResult(success=False, message="'config' must be a non-empty object")

        # Validate channel-specific config
        validation = self._validate_channel_config(channel, config)
        if not validation["valid"]:
            return SkillResult(success=False, message=validation["error"])

        channels = self._load_channels()
        channels[channel] = {
            "config": config,
            "min_severity": min_severity,
            "enabled": enabled,
            "configured_at": datetime.now().isoformat(),
        }
        self._save_channels(channels)

        return SkillResult(
            success=True,
            message=f"Channel '{channel}' configured (min_severity={min_severity}, enabled={enabled})",
            data={"channel": channel, "min_severity": min_severity, "enabled": enabled},
        )

    def _validate_channel_config(self, channel: str, config: Dict) -> Dict:
        """Validate channel configuration has required fields."""
        required_fields = {
            "telegram": ["bot_token", "chat_id"],
            "discord": ["webhook_url"],
            "slack": ["webhook_url"],
            "sms": ["account_sid", "auth_token", "from_number", "to_number"],
            "webhook": ["url"],
        }
        required = required_fields.get(channel, [])
        missing = [f for f in required if not config.get(f)]
        if missing:
            return {"valid": False, "error": f"Missing required fields for {channel}: {', '.join(missing)}"}
        return {"valid": True}

    async def _list_channels(self, params: Dict) -> SkillResult:
        """List all configured channels."""
        channels = self._load_channels()

        # Also check env-based channels
        for ch_type in ("telegram", "discord", "slack", "sms"):
            if ch_type not in channels:
                env_cfg = self._get_env_config(ch_type)
                if env_cfg:
                    channels[ch_type] = {
                        "config": {"source": "environment"},
                        "min_severity": "info",
                        "enabled": True,
                        "configured_at": "from environment variables",
                    }

        # Sanitize output (don't expose tokens)
        safe_channels = {}
        for name, data in channels.items():
            safe = {
                "channel": name,
                "enabled": data.get("enabled", True),
                "min_severity": data.get("min_severity", "info"),
                "configured_at": data.get("configured_at", "unknown"),
                "has_config": bool(data.get("config")),
            }
            safe_channels[name] = safe

        return SkillResult(
            success=True,
            message=f"{len(safe_channels)} channels configured",
            data={"channels": safe_channels, "count": len(safe_channels)},
        )

    async def _test_channel(self, params: Dict) -> SkillResult:
        """Send a test message to verify channel works."""
        channel = params.get("channel", "").strip().lower()
        if not channel:
            return SkillResult(success=False, message="'channel' is required")

        ch_config = self._get_channel_config(channel)
        if not ch_config:
            return SkillResult(
                success=False,
                message=f"Channel '{channel}' not configured",
            )

        test_msg = (
            f"Test notification from Singularity Agent. "
            f"Channel: {channel}. Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

        result = await self._dispatch(channel, ch_config.get("config", {}), test_msg, "Test Notification")

        self._record_notification(
            channel, "info", "Test notification",
            result.get("success", False),
            result.get("error", ""),
        )

        if result.get("success"):
            return SkillResult(
                success=True,
                message=f"Test message sent successfully via {channel}",
                data={"channel": channel, **result},
            )
        return SkillResult(
            success=False,
            message=f"Test failed for {channel}: {result.get('error', 'unknown')}",
            data={"channel": channel, "error": result.get("error")},
        )

    async def _notification_history(self, params: Dict) -> SkillResult:
        """View notification history."""
        limit = min(params.get("limit", 20), 100)
        filter_channel = params.get("channel", "").lower()
        filter_severity = params.get("severity", "").lower()

        history = _load_json(HISTORY_FILE)
        if not isinstance(history, list):
            history = []

        # Apply filters
        filtered = history
        if filter_channel:
            filtered = [h for h in filtered if h.get("channel") == filter_channel]
        if filter_severity:
            filtered = [h for h in filtered if h.get("severity") == filter_severity]

        recent = filtered[-limit:]
        recent.reverse()

        # Compute stats
        total = len(history)
        successful = sum(1 for h in history if h.get("success"))
        by_channel = {}
        by_severity = {}
        for h in history:
            ch = h.get("channel", "unknown")
            sev = h.get("severity", "unknown")
            by_channel[ch] = by_channel.get(ch, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return SkillResult(
            success=True,
            message=f"Showing {len(recent)} of {len(filtered)} notifications",
            data={
                "history": recent,
                "stats": {
                    "total": total,
                    "successful": successful,
                    "success_rate": successful / total if total > 0 else 0,
                    "by_channel": by_channel,
                    "by_severity": by_severity,
                },
            },
        )

    async def _remove_channel(self, params: Dict) -> SkillResult:
        """Remove a configured channel."""
        channel = params.get("channel", "").strip().lower()
        if not channel:
            return SkillResult(success=False, message="'channel' is required")

        channels = self._load_channels()
        if channel not in channels:
            return SkillResult(success=False, message=f"Channel '{channel}' not found")

        del channels[channel]
        self._save_channels(channels)

        return SkillResult(
            success=True,
            message=f"Channel '{channel}' removed",
            data={"channel": channel},
        )
