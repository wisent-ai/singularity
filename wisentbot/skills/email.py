#!/usr/bin/env python3
"""
Email Skill

Enables agents to:
- Send emails (via Resend, SendGrid, or SMTP)
- Read emails (via IMAP)
- Manage mailing lists
"""

import httpx
from typing import Dict, List, Optional
from .base import Skill, SkillResult, SkillManifest, SkillAction


class EmailSkill(Skill):
    """
    Skill for sending and managing emails.

    Supports multiple providers:
    - Resend (preferred)
    - SendGrid
    - SMTP

    Required credentials (one of):
    - RESEND_API_KEY: Resend API key
    - SENDGRID_API_KEY: SendGrid API key
    - EMAIL_SMTP_HOST, EMAIL_SMTP_USER, EMAIL_SMTP_PASS: SMTP credentials
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="email",
            name="Email Management",
            version="1.0.0",
            category="communication",
            description="Send and manage emails via Resend, SendGrid, or SMTP",
            required_credentials=[
                "RESEND_API_KEY",
                "NAMECHEAP_API_KEY",
                "NAMECHEAP_API_USER",
                "NAMECHEAP_USERNAME",
                "NAMECHEAP_CLIENT_IP"
            ],
            install_cost=0,
            actions=[
                SkillAction(
                    name="send_email",
                    description="Send an email",
                    parameters={
                        "to": {"type": "string", "required": True, "description": "Recipient email(s)"},
                        "subject": {"type": "string", "required": True, "description": "Email subject"},
                        "body": {"type": "string", "required": True, "description": "Email body (text or HTML)"},
                        "from_name": {"type": "string", "required": False, "description": "Sender name"},
                        "from_email": {"type": "string", "required": False, "description": "Sender email"},
                        "html": {"type": "boolean", "required": False, "description": "Is body HTML? (default: false)"}
                    },
                    estimated_cost=0.001,  # ~$0.001 per email
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="send_template",
                    description="Send email using a template",
                    parameters={
                        "to": {"type": "string", "required": True, "description": "Recipient email"},
                        "template_id": {"type": "string", "required": True, "description": "Template ID"},
                        "variables": {"type": "object", "required": False, "description": "Template variables"}
                    },
                    estimated_cost=0.001,
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="send_bulk",
                    description="Send bulk emails",
                    parameters={
                        "recipients": {"type": "array", "required": True, "description": "List of {email, name, variables}"},
                        "subject": {"type": "string", "required": True, "description": "Email subject"},
                        "body": {"type": "string", "required": True, "description": "Email body"}
                    },
                    estimated_cost=0.001,  # Per email
                    estimated_duration_seconds=30,
                    success_probability=0.9
                ),
                SkillAction(
                    name="check_delivery",
                    description="Check email delivery status",
                    parameters={
                        "email_id": {"type": "string", "required": True, "description": "Email ID to check"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="add_domain",
                    description="Add a domain to Resend for email sending",
                    parameters={
                        "domain": {"type": "string", "required": True, "description": "Domain to add (e.g., agent.wisent.com)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_domain_dns",
                    description="Get DNS records needed to verify a domain",
                    parameters={
                        "domain_id": {"type": "string", "required": True, "description": "Resend domain ID"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="verify_domain",
                    description="Verify a domain after DNS records are set",
                    parameters={
                        "domain_id": {"type": "string", "required": True, "description": "Resend domain ID to verify"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=5,
                    success_probability=0.9
                ),
                SkillAction(
                    name="list_domains",
                    description="List all domains configured in Resend",
                    parameters={},
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="setup_domain",
                    description="Full automated domain setup: add to Resend, configure DNS via Namecheap, verify",
                    parameters={
                        "domain": {"type": "string", "required": True, "description": "Domain to set up for email"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=30,
                    success_probability=0.85
                ),
                # Inbound email actions (Resend)
                SkillAction(
                    name="get_received_emails",
                    description="List received emails (Resend Inbound)",
                    parameters={
                        "limit": {"type": "integer", "required": False, "description": "Max emails to return (default: 50)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="get_received_email",
                    description="Get a specific received email by ID",
                    parameters={
                        "email_id": {"type": "string", "required": True, "description": "Email ID to retrieve"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=2,
                    success_probability=0.95
                ),
                SkillAction(
                    name="wait_for_email",
                    description="Wait for an email matching criteria (for verification codes)",
                    parameters={
                        "to": {"type": "string", "required": True, "description": "Email address to check"},
                        "sender_contains": {"type": "string", "required": False, "description": "Sender must contain this string"},
                        "subject_contains": {"type": "string", "required": False, "description": "Subject must contain this string"},
                        "timeout": {"type": "integer", "required": False, "description": "Timeout in seconds (default: 120)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=60,
                    success_probability=0.85
                ),
                SkillAction(
                    name="extract_code",
                    description="Extract verification code from email body",
                    parameters={
                        "email_id": {"type": "string", "required": True, "description": "Email ID to extract code from"},
                        "code_length": {"type": "integer", "required": False, "description": "Expected code length (default: 6)"}
                    },
                    estimated_cost=0,
                    estimated_duration_seconds=1,
                    success_probability=0.9
                )
            ]
        )

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self.http = httpx.AsyncClient(timeout=30)
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Detect which email provider to use"""
        if self.credentials.get("RESEND_API_KEY"):
            return "resend"
        elif self.credentials.get("SENDGRID_API_KEY"):
            return "sendgrid"
        elif self.credentials.get("EMAIL_SMTP_HOST"):
            return "smtp"
        return "none"

    def check_credentials(self) -> bool:
        """Check if any email provider is configured"""
        return self._provider != "none"

    async def execute(self, action: str, params: Dict) -> SkillResult:
        """Execute an email action"""
        if not self.check_credentials():
            return SkillResult(
                success=False,
                message="No email provider configured. Set RESEND_API_KEY, SENDGRID_API_KEY, or SMTP credentials."
            )

        try:
            if action == "send_email":
                return await self._send_email(
                    params.get("to"),
                    params.get("subject"),
                    params.get("body"),
                    params.get("from_name"),
                    params.get("from_email"),
                    params.get("html", False)
                )
            elif action == "send_template":
                return await self._send_template(
                    params.get("to"),
                    params.get("template_id"),
                    params.get("variables", {})
                )
            elif action == "send_bulk":
                return await self._send_bulk(
                    params.get("recipients"),
                    params.get("subject"),
                    params.get("body")
                )
            elif action == "check_delivery":
                return await self._check_delivery(params.get("email_id"))
            elif action == "add_domain":
                return await self._add_domain(params.get("domain"))
            elif action == "get_domain_dns":
                return await self._get_domain_dns(params.get("domain_id"))
            elif action == "verify_domain":
                return await self._verify_domain(params.get("domain_id"))
            elif action == "list_domains":
                return await self._list_domains()
            elif action == "setup_domain":
                return await self._setup_domain(params.get("domain"))
            # Inbound email actions
            elif action == "get_received_emails":
                return await self._get_received_emails(params.get("limit", 50))
            elif action == "get_received_email":
                return await self._get_received_email(params.get("email_id"))
            elif action == "wait_for_email":
                return await self._wait_for_email(
                    params.get("to"),
                    params.get("sender_contains"),
                    params.get("subject_contains"),
                    params.get("timeout", 120)
                )
            elif action == "extract_code":
                return await self._extract_code(
                    params.get("email_id"),
                    params.get("code_length", 6)
                )
            else:
                return SkillResult(
                    success=False,
                    message=f"Unknown action: {action}"
                )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Email error: {str(e)}"
            )

    async def _send_email(
        self,
        to: str,
        subject: str,
        body: str,
        from_name: str = None,
        from_email: str = None,
        html: bool = False
    ) -> SkillResult:
        """Send an email"""
        if not to or not subject or not body:
            return SkillResult(success=False, message="To, subject, and body required")

        if self._provider == "resend":
            return await self._send_resend(to, subject, body, from_name, from_email, html)
        elif self._provider == "sendgrid":
            return await self._send_sendgrid(to, subject, body, from_name, from_email, html)
        else:
            return SkillResult(success=False, message="No email provider available")

    async def _send_resend(
        self,
        to: str,
        subject: str,
        body: str,
        from_name: str = None,
        from_email: str = None,
        html: bool = False
    ) -> SkillResult:
        """Send via Resend API"""
        api_key = self.credentials.get("RESEND_API_KEY")
        default_from = self.credentials.get("EMAIL_FROM", "agent@wisent.com")

        from_addr = from_email or default_from
        if from_name:
            from_addr = f"{from_name} <{from_addr}>"

        data = {
            "from": from_addr,
            "to": [to] if isinstance(to, str) else to,
            "subject": subject
        }

        if html:
            data["html"] = body
        else:
            data["text"] = body

        response = await self.http.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            return SkillResult(
                success=True,
                message=f"Email sent to {to}",
                data={
                    "email_id": result.get("id"),
                    "to": to,
                    "subject": subject
                },
                cost=0.001
            )
        else:
            return SkillResult(
                success=False,
                message=f"Resend error: {response.text}"
            )

    async def _send_sendgrid(
        self,
        to: str,
        subject: str,
        body: str,
        from_name: str = None,
        from_email: str = None,
        html: bool = False
    ) -> SkillResult:
        """Send via SendGrid API"""
        api_key = self.credentials.get("SENDGRID_API_KEY")
        default_from = self.credentials.get("EMAIL_FROM", "agent@wisent.com")

        data = {
            "personalizations": [{"to": [{"email": to}]}],
            "from": {
                "email": from_email or default_from,
                "name": from_name or "Agent"
            },
            "subject": subject,
            "content": [{
                "type": "text/html" if html else "text/plain",
                "value": body
            }]
        }

        response = await self.http.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=data
        )

        if response.status_code in [200, 202]:
            return SkillResult(
                success=True,
                message=f"Email sent to {to}",
                data={"to": to, "subject": subject},
                cost=0.001
            )
        else:
            return SkillResult(
                success=False,
                message=f"SendGrid error: {response.text}"
            )

    async def _send_template(
        self,
        to: str,
        template_id: str,
        variables: Dict
    ) -> SkillResult:
        """Send email using template"""
        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Templates only supported with Resend"
            )

        api_key = self.credentials.get("RESEND_API_KEY")
        default_from = self.credentials.get("EMAIL_FROM", "agent@wisent.com")

        # Resend doesn't have native templates, so we'd need to
        # integrate with their react-email or store templates ourselves
        return SkillResult(
            success=False,
            message="Template feature not yet implemented"
        )

    async def _send_bulk(
        self,
        recipients: List[Dict],
        subject: str,
        body: str
    ) -> SkillResult:
        """Send bulk emails"""
        if not recipients:
            return SkillResult(success=False, message="Recipients required")

        sent = 0
        failed = 0
        errors = []

        for recipient in recipients:
            email = recipient.get("email")
            name = recipient.get("name")
            vars = recipient.get("variables", {})

            # Simple variable replacement
            personalized_body = body
            personalized_subject = subject
            for key, value in vars.items():
                personalized_body = personalized_body.replace(f"{{{{{key}}}}}", str(value))
                personalized_subject = personalized_subject.replace(f"{{{{{key}}}}}", str(value))

            result = await self._send_email(
                email,
                personalized_subject,
                personalized_body,
                from_name=None,
                from_email=None,
                html=False
            )

            if result.success:
                sent += 1
            else:
                failed += 1
                errors.append({"email": email, "error": result.message})

        return SkillResult(
            success=failed == 0,
            message=f"Sent {sent}/{len(recipients)} emails",
            data={
                "sent": sent,
                "failed": failed,
                "total": len(recipients),
                "errors": errors[:10]  # Limit errors
            },
            cost=sent * 0.001
        )

    async def _check_delivery(self, email_id: str) -> SkillResult:
        """Check email delivery status"""
        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Delivery tracking only supported with Resend"
            )

        api_key = self.credentials.get("RESEND_API_KEY")

        response = await self.http.get(
            f"https://api.resend.com/emails/{email_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            data = response.json()
            return SkillResult(
                success=True,
                message=f"Email status: {data.get('status')}",
                data={
                    "id": data.get("id"),
                    "status": data.get("status"),
                    "to": data.get("to"),
                    "subject": data.get("subject"),
                    "created_at": data.get("created_at")
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to check status: {response.text}"
            )

    async def _add_domain(self, domain: str) -> SkillResult:
        """Add a domain to Resend"""
        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Domain management only supported with Resend"
            )

        api_key = self.credentials.get("RESEND_API_KEY")

        response = await self.http.post(
            "https://api.resend.com/domains",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={"name": domain}
        )

        if response.status_code in [200, 201]:
            data = response.json()
            return SkillResult(
                success=True,
                message=f"Domain {domain} added to Resend",
                data={
                    "domain_id": data.get("id"),
                    "domain": domain,
                    "status": data.get("status"),
                    "records": data.get("records", [])
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to add domain: {response.text}"
            )

    async def _get_domain_dns(self, domain_id: str) -> SkillResult:
        """Get DNS records for a domain"""
        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Domain management only supported with Resend"
            )

        api_key = self.credentials.get("RESEND_API_KEY")

        response = await self.http.get(
            f"https://api.resend.com/domains/{domain_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            data = response.json()
            return SkillResult(
                success=True,
                message=f"DNS records for domain",
                data={
                    "domain_id": data.get("id"),
                    "domain": data.get("name"),
                    "status": data.get("status"),
                    "records": data.get("records", [])
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to get domain: {response.text}"
            )

    async def _verify_domain(self, domain_id: str) -> SkillResult:
        """Verify a domain in Resend"""
        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Domain management only supported with Resend"
            )

        api_key = self.credentials.get("RESEND_API_KEY")

        response = await self.http.post(
            f"https://api.resend.com/domains/{domain_id}/verify",
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            data = response.json()
            return SkillResult(
                success=True,
                message=f"Domain verification initiated",
                data={
                    "domain_id": domain_id,
                    "status": data.get("status", "pending")
                }
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to verify domain: {response.text}"
            )

    async def _list_domains(self) -> SkillResult:
        """List all domains in Resend"""
        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Domain management only supported with Resend"
            )

        api_key = self.credentials.get("RESEND_API_KEY")

        response = await self.http.get(
            "https://api.resend.com/domains",
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            data = response.json()
            domains = data.get("data", [])
            return SkillResult(
                success=True,
                message=f"Found {len(domains)} domains",
                data={"domains": domains}
            )
        else:
            return SkillResult(
                success=False,
                message=f"Failed to list domains: {response.text}"
            )

    async def _setup_domain(self, domain: str) -> SkillResult:
        """
        Full automated domain setup:
        1. Add domain to Resend
        2. Get DNS records
        3. Add DNS records to Namecheap
        4. Verify domain
        """
        import xml.etree.ElementTree as ET
        import asyncio

        if self._provider != "resend":
            return SkillResult(
                success=False,
                message="Domain setup only supported with Resend"
            )

        # Validate namecheap credentials from self.credentials
        required_nc = ["NAMECHEAP_API_KEY", "NAMECHEAP_API_USER", "NAMECHEAP_USERNAME", "NAMECHEAP_CLIENT_IP"]
        missing = [k for k in required_nc if not self.credentials.get(k)]
        if missing:
            return SkillResult(
                success=False,
                message=f"Missing Namecheap credentials: {missing}"
            )

        # Step 1: Add domain to Resend
        add_result = await self._add_domain(domain)
        if not add_result.success:
            return add_result

        domain_id = add_result.data.get("domain_id")
        records = add_result.data.get("records", [])

        if not records:
            # Fetch records separately
            dns_result = await self._get_domain_dns(domain_id)
            if dns_result.success:
                records = dns_result.data.get("records", [])

        if not records:
            return SkillResult(
                success=False,
                message="No DNS records returned from Resend"
            )

        # Step 2: Parse domain for Namecheap
        # If domain is "ralphtrading.com" -> SLD=ralphtrading, TLD=com
        # If domain is "sub.ralphtrading.com" -> still need root domain ralphtrading.com
        parts = domain.split(".")
        if len(parts) < 2:
            return SkillResult(success=False, message="Invalid domain format")

        # For agent-owned domains, assume the domain itself is the root
        # e.g., "ralphtrading.com" -> SLD=ralphtrading, TLD=com
        sld = parts[-2]
        tld = parts[-1]

        # Check if this is a subdomain (more than 2 parts for .com/.ai or more than 3 for .co.uk etc)
        # For simplicity, if domain has more than 2 parts, treat last 2 as root
        is_subdomain = len(parts) > 2

        # Step 3: Get existing DNS records from Namecheap
        nc_params = {
            "ApiUser": self.credentials["NAMECHEAP_API_USER"],
            "ApiKey": self.credentials["NAMECHEAP_API_KEY"],
            "UserName": self.credentials["NAMECHEAP_USERNAME"],
            "ClientIp": self.credentials["NAMECHEAP_CLIENT_IP"],
            "Command": "namecheap.domains.dns.getHosts",
            "SLD": sld,
            "TLD": tld
        }

        nc_response = await self.http.get(
            "https://api.namecheap.com/xml.response",
            params=nc_params,
            timeout=30
        )

        root = ET.fromstring(nc_response.text)
        ns = {"ns": "http://api.namecheap.com/xml.response"}
        existing_hosts = root.findall(".//ns:host", ns)

        existing_records = []
        for h in existing_hosts:
            existing_records.append({
                "name": h.get("Name"),
                "type": h.get("Type"),
                "address": h.get("Address"),
                "mxpref": h.get("MXPref", "10"),
                "ttl": h.get("TTL", "1799")
            })

        # Step 4: Add Resend DNS records
        # Note: Resend returns record names already relative to the root domain
        # e.g., for domain "testagent.agents.trade.wisent.com", Resend returns
        # "resend._domainkey.testagent.agents.trade" which is correct for wisent.com
        for rec in records:
            rec_type = rec.get("type", rec.get("record_type", "")).upper()
            rec_name = rec.get("name", rec.get("host", ""))
            rec_value = rec.get("value", rec.get("data", ""))
            rec_priority = rec.get("priority", "10")

            existing_records.append({
                "name": rec_name,
                "type": rec_type,
                "address": rec_value,
                "mxpref": str(rec_priority),
                "ttl": "1799"
            })

        # Step 5: Set all DNS records in Namecheap
        # EmailType: "MX" is required to enable custom MX records
        set_params = {
            "ApiUser": self.credentials["NAMECHEAP_API_USER"],
            "ApiKey": self.credentials["NAMECHEAP_API_KEY"],
            "UserName": self.credentials["NAMECHEAP_USERNAME"],
            "ClientIp": self.credentials["NAMECHEAP_CLIENT_IP"],
            "Command": "namecheap.domains.dns.setHosts",
            "SLD": sld,
            "TLD": tld,
            "EmailType": "MX"
        }

        for i, rec in enumerate(existing_records, 1):
            set_params[f"HostName{i}"] = rec["name"]
            set_params[f"RecordType{i}"] = rec["type"]
            set_params[f"Address{i}"] = rec["address"]
            set_params[f"TTL{i}"] = rec.get("ttl", "1799")
            if rec["type"] == "MX":
                set_params[f"MXPref{i}"] = rec.get("mxpref", "10")

        set_response = await self.http.get(
            "https://api.namecheap.com/xml.response",
            params=set_params,
            timeout=30
        )

        set_root = ET.fromstring(set_response.text)
        if set_root.get("Status") != "OK":
            errors = set_root.findall(".//ns:Error", ns)
            error_msg = errors[0].text if errors else "Unknown error"
            return SkillResult(
                success=False,
                message=f"Failed to set DNS records: {error_msg}"
            )

        # Step 6: Wait a bit then verify domain
        await asyncio.sleep(5)  # Give DNS a moment

        verify_result = await self._verify_domain(domain_id)

        return SkillResult(
            success=True,
            message=f"Domain {domain} setup complete. DNS records added, verification initiated.",
            data={
                "domain_id": domain_id,
                "domain": domain,
                "records_added": len(records),
                "verification_status": verify_result.data.get("status") if verify_result.success else "pending"
            }
        )

    # ==================== INBOUND EMAIL METHODS (Resend) ====================

    async def _get_received_emails(self, limit: int = 50) -> SkillResult:
        """List received emails from Resend Inbound"""
        if self._provider != "resend":
            return SkillResult(success=False, message="Inbound email requires Resend")

        api_key = self.credentials.get("RESEND_API_KEY")
        response = await self.http.get(
            "https://api.resend.com/emails/receiving",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": limit}
        )

        if response.status_code != 200:
            return SkillResult(
                success=False,
                message=f"Failed to get emails: {response.text}"
            )

        data = response.json()
        emails = data.get("data", [])

        return SkillResult(
            success=True,
            message=f"Found {len(emails)} received emails",
            data={"emails": emails, "count": len(emails)}
        )

    async def _get_received_email(self, email_id: str) -> SkillResult:
        """Get a specific received email by ID"""
        if self._provider != "resend":
            return SkillResult(success=False, message="Inbound email requires Resend")

        api_key = self.credentials.get("RESEND_API_KEY")
        response = await self.http.get(
            f"https://api.resend.com/emails/receiving/{email_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            return SkillResult(
                success=False,
                message=f"Failed to get email: {response.text}"
            )

        email = response.json()

        return SkillResult(
            success=True,
            message=f"Email from {email.get('from', 'unknown')}",
            data=email
        )

    async def _wait_for_email(
        self,
        to: str,
        sender_contains: str = None,
        subject_contains: str = None,
        timeout: int = 120
    ) -> SkillResult:
        """Wait for an email matching criteria"""
        import asyncio
        import time

        if self._provider != "resend":
            return SkillResult(success=False, message="Inbound email requires Resend")

        api_key = self.credentials.get("RESEND_API_KEY")
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = await self.http.get(
                "https://api.resend.com/emails/receiving",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"limit": 50}
            )

            if response.status_code == 200:
                emails = response.json().get("data", [])

                for email in emails:
                    # Check recipient
                    email_to = email.get("to", [])
                    if isinstance(email_to, list):
                        email_to = [e.get("email", e) if isinstance(e, dict) else e for e in email_to]
                    else:
                        email_to = [email_to]

                    if to.lower() not in [e.lower() for e in email_to]:
                        continue

                    # Check sender
                    if sender_contains:
                        email_from = email.get("from", {})
                        if isinstance(email_from, dict):
                            email_from = email_from.get("email", "")
                        if sender_contains.lower() not in email_from.lower():
                            continue

                    # Check subject
                    if subject_contains:
                        subject = email.get("subject", "")
                        if subject_contains.lower() not in subject.lower():
                            continue

                    # Found matching email
                    return SkillResult(
                        success=True,
                        message=f"Found email from {email.get('from')}",
                        data=email
                    )

            await asyncio.sleep(5)

        return SkillResult(
            success=False,
            message=f"No matching email received within {timeout} seconds"
        )

    async def _extract_code(self, email_id: str, code_length: int = 6) -> SkillResult:
        """Extract verification code from email body"""
        import re

        # Get the email
        email_result = await self._get_received_email(email_id)
        if not email_result.success:
            return email_result

        email = email_result.data
        body = email.get("text", "") or email.get("html", "") or ""

        # Try to find code - look for standalone digits of expected length
        pattern = rf'\b(\d{{{code_length}}})\b'
        match = re.search(pattern, body)

        if match:
            code = match.group(1)
            return SkillResult(
                success=True,
                message=f"Found {code_length}-digit code",
                data={"code": code, "email_id": email_id}
            )

        # Also try common patterns
        patterns = [
            rf'code[:\s]+(\d{{{code_length}}})',
            rf'verification[:\s]+(\d{{{code_length}}})',
            rf'confirm[:\s]+(\d{{{code_length}}})',
        ]

        for p in patterns:
            match = re.search(p, body, re.IGNORECASE)
            if match:
                code = match.group(1)
                return SkillResult(
                    success=True,
                    message=f"Found {code_length}-digit code",
                    data={"code": code, "email_id": email_id}
                )

        return SkillResult(
            success=False,
            message=f"No {code_length}-digit code found in email"
        )

    async def close(self):
        """Clean up"""
        await self.http.aclose()
