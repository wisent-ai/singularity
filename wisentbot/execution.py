#!/usr/bin/env python3
"""
Execution Engine for Autonomous Agent

Provides capabilities for the agent to interact with the world:
- Web browsing (reading, forms, navigation)
- HTTP requests (APIs)
- Email sending
- Code execution
- File operations
- Payment processing
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import httpx

try:
    from playwright.async_api import async_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExecutionResult:
    """Result of any execution"""
    success: bool
    message: str = ""
    data: Dict = field(default_factory=dict)
    cost: float = 0  # Resource cost
    duration_seconds: float = 0


@dataclass
class WebPage:
    """Scraped web page data"""
    url: str
    title: str = ""
    text: str = ""
    links: List[Dict] = field(default_factory=list)
    forms: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """
    Executes actions in the real world.

    Capabilities:
    - browse_web: Navigate and read web pages
    - http_request: Make HTTP API calls
    - send_email: Send emails
    - run_code: Execute Python code safely
    - create_file: Create files/content
    - process_payment: Handle payments
    """

    def __init__(
        self,
        coordinator_url: str = "https://singularity.wisent.ai",
        agent_instance_id: str = "",
        agent_ticker: str = "",
        sandbox_mode: bool = True  # Safety first
    ):
        self.coordinator_url = coordinator_url
        self.agent_instance_id = agent_instance_id
        self.agent_ticker = agent_ticker
        self.sandbox_mode = sandbox_mode

        self.http = httpx.AsyncClient(timeout=60)
        self.browser: Optional[Browser] = None
        self._playwright = None

    async def initialize(self):
        """Initialize resources like browser"""
        if HAS_PLAYWRIGHT and not self.sandbox_mode:
            self._playwright = await async_playwright().start()
            self.browser = await self._playwright.chromium.launch(headless=True)

    async def close(self):
        """Clean up resources"""
        await self.http.aclose()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    # ========================================================================
    # WEB BROWSING
    # ========================================================================

    async def browse_web(self, url: str, extract_links: bool = True) -> ExecutionResult:
        """
        Browse a web page and extract content.

        Uses simple HTTP + BeautifulSoup for most cases,
        falls back to Playwright for JS-heavy sites.
        """
        start = datetime.now()

        try:
            # First try simple HTTP request
            response = await self.http.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; WisentBot/1.0)"
                },
                follow_redirects=True
            )

            if response.status_code != 200:
                return ExecutionResult(
                    success=False,
                    message=f"HTTP {response.status_code}"
                )

            page = self._parse_html(url, response.text, extract_links)

            duration = (datetime.now() - start).total_seconds()
            return ExecutionResult(
                success=True,
                message=f"Fetched {len(page.text)} chars from {url}",
                data={
                    "url": page.url,
                    "title": page.title,
                    "text": page.text[:10000],  # Limit size
                    "links": page.links[:50],
                    "forms": page.forms
                },
                cost=0.01,  # Small compute cost
                duration_seconds=duration
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Browse error: {str(e)}"
            )

    def _parse_html(self, url: str, html: str, extract_links: bool) -> WebPage:
        """Parse HTML into structured data"""
        if not HAS_BS4:
            return WebPage(url=url, text=html[:5000])

        soup = BeautifulSoup(html, 'html.parser')

        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Get title
        title = soup.title.string if soup.title else ""

        # Get text
        text = soup.get_text(separator='\n', strip=True)

        # Get links
        links = []
        if extract_links:
            for a in soup.find_all('a', href=True)[:50]:
                href = a.get('href', '')
                if href.startswith('http'):
                    links.append({
                        "text": a.get_text(strip=True)[:100],
                        "url": href
                    })

        # Get forms
        forms = []
        for form in soup.find_all('form')[:5]:
            form_data = {
                "action": form.get('action', ''),
                "method": form.get('method', 'get'),
                "inputs": []
            }
            for inp in form.find_all(['input', 'textarea', 'select']):
                form_data["inputs"].append({
                    "name": inp.get('name', ''),
                    "type": inp.get('type', 'text'),
                    "value": inp.get('value', '')
                })
            forms.append(form_data)

        return WebPage(
            url=url,
            title=title,
            text=text,
            links=links,
            forms=forms
        )

    async def browse_with_browser(self, url: str, actions: List[Dict] = None) -> ExecutionResult:
        """
        Use full browser for JS-heavy pages or interactions.

        actions: List of actions like:
            - {"type": "click", "selector": "#button"}
            - {"type": "fill", "selector": "#email", "value": "test@test.com"}
            - {"type": "wait", "seconds": 2}
        """
        if not HAS_PLAYWRIGHT or not self.browser:
            return ExecutionResult(
                success=False,
                message="Browser not available"
            )

        if self.sandbox_mode:
            return ExecutionResult(
                success=False,
                message="Browser disabled in sandbox mode"
            )

        start = datetime.now()

        try:
            page = await self.browser.new_page()
            await page.goto(url, wait_until='networkidle')

            # Execute actions
            if actions:
                for action in actions:
                    action_type = action.get("type")
                    if action_type == "click":
                        await page.click(action["selector"])
                    elif action_type == "fill":
                        await page.fill(action["selector"], action["value"])
                    elif action_type == "wait":
                        await asyncio.sleep(action.get("seconds", 1))

            # Extract content
            title = await page.title()
            content = await page.content()
            text = await page.evaluate("() => document.body.innerText")

            await page.close()

            duration = (datetime.now() - start).total_seconds()
            return ExecutionResult(
                success=True,
                message=f"Browsed {url}",
                data={
                    "url": url,
                    "title": title,
                    "text": text[:10000]
                },
                cost=0.05,  # Higher cost for browser
                duration_seconds=duration
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Browser error: {str(e)}"
            )

    # ========================================================================
    # HTTP REQUESTS
    # ========================================================================

    async def http_request(
        self,
        method: str,
        url: str,
        headers: Dict = None,
        json_data: Dict = None,
        data: Dict = None,
        params: Dict = None
    ) -> ExecutionResult:
        """Make HTTP API request"""
        start = datetime.now()

        try:
            response = await self.http.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                json=json_data,
                data=data,
                params=params
            )

            duration = (datetime.now() - start).total_seconds()

            # Try to parse JSON response
            try:
                response_data = response.json()
            except:
                response_data = {"text": response.text[:5000]}

            return ExecutionResult(
                success=response.status_code < 400,
                message=f"HTTP {response.status_code}",
                data={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response_data
                },
                cost=0.005,
                duration_seconds=duration
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"HTTP error: {str(e)}"
            )

    # ========================================================================
    # EMAIL
    # ========================================================================

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        from_name: str = None
    ) -> ExecutionResult:
        """
        Send email via configured provider.

        Uses platform's email service to maintain sender reputation.
        """
        try:
            response = await self.http.post(
                f"{self.coordinator_url}/api/agents/{self.agent_instance_id}/email",
                json={
                    "to": to,
                    "subject": subject,
                    "body": body,
                    "from_name": from_name or self.agent_ticker
                }
            )

            if response.status_code == 200:
                return ExecutionResult(
                    success=True,
                    message=f"Email sent to {to}",
                    cost=0.1  # Email cost
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Email failed: {response.text}"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Email error: {str(e)}"
            )

    # ========================================================================
    # CODE EXECUTION
    # ========================================================================

    async def run_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30
    ) -> ExecutionResult:
        """
        Execute code in sandboxed environment.

        Currently supports Python only.
        Runs in subprocess with resource limits.
        """
        if not self.sandbox_mode:
            return ExecutionResult(
                success=False,
                message="Code execution requires sandbox mode"
            )

        if language != "python":
            return ExecutionResult(
                success=False,
                message=f"Language {language} not supported"
            )

        start = datetime.now()

        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                # Add safety wrapper
                safe_code = f"""
import sys
import os

# Disable dangerous operations
def blocked(*args, **kwargs):
    raise PermissionError("Operation not allowed in sandbox")

os.system = blocked
os.popen = blocked
os.remove = blocked
os.rmdir = blocked
os.makedirs = blocked
__builtins__['exec'] = blocked
__builtins__['eval'] = blocked

# User code
{code}
"""
                f.write(safe_code)
                temp_path = f.name

            # Run in subprocess with timeout
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir()
            )

            # Clean up
            os.unlink(temp_path)

            duration = (datetime.now() - start).total_seconds()

            return ExecutionResult(
                success=result.returncode == 0,
                message=result.stdout[:5000] if result.returncode == 0 else result.stderr[:5000],
                data={
                    "stdout": result.stdout[:5000],
                    "stderr": result.stderr[:5000],
                    "return_code": result.returncode
                },
                cost=0.02,  # Compute cost
                duration_seconds=duration
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                message=f"Code execution timed out after {timeout}s"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Code execution error: {str(e)}"
            )

    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================

    async def create_file(
        self,
        content: str,
        filename: str,
        file_type: str = "text"
    ) -> ExecutionResult:
        """
        Create a file and upload to storage.

        Returns URL to access the file.
        """
        try:
            # Upload to coordinator's file storage
            response = await self.http.post(
                f"{self.coordinator_url}/api/agents/{self.agent_instance_id}/files",
                json={
                    "filename": filename,
                    "content": content,
                    "type": file_type
                }
            )

            if response.status_code == 200:
                data = response.json()
                return ExecutionResult(
                    success=True,
                    message=f"Created {filename}",
                    data={
                        "url": data.get("url"),
                        "filename": filename,
                        "size": len(content)
                    },
                    cost=0.01
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Upload failed: {response.text}"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"File creation error: {str(e)}"
            )

    # ========================================================================
    # PAYMENT PROCESSING
    # ========================================================================

    async def process_payment(
        self,
        amount: float,
        currency: str = "WISENT",
        recipient: str = None,
        description: str = ""
    ) -> ExecutionResult:
        """
        Process a payment (send or receive WISENT).

        For sending: provide recipient
        For receiving: no recipient (generates payment request)
        """
        try:
            if recipient:
                # Sending payment
                response = await self.http.post(
                    f"{self.coordinator_url}/api/agents/{self.agent_instance_id}/pay",
                    json={
                        "amount": amount,
                        "currency": currency,
                        "recipient": recipient,
                        "description": description
                    }
                )
            else:
                # Create payment request
                response = await self.http.post(
                    f"{self.coordinator_url}/api/agents/{self.agent_instance_id}/invoice",
                    json={
                        "amount": amount,
                        "currency": currency,
                        "description": description
                    }
                )

            if response.status_code == 200:
                data = response.json()
                return ExecutionResult(
                    success=True,
                    message=f"Payment {'sent' if recipient else 'requested'}: {amount} {currency}",
                    data=data,
                    cost=amount * 0.01 if recipient else 0  # 1% fee on outgoing
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Payment failed: {response.text}"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Payment error: {str(e)}"
            )

    # ========================================================================
    # TRADING
    # ========================================================================

    async def trade(
        self,
        action: str,  # "buy" or "sell"
        token: str,
        amount: float,
        max_slippage: float = 0.05
    ) -> ExecutionResult:
        """Execute a token trade"""
        try:
            endpoint = "buy" if action.lower() == "buy" else "sell"
            response = await self.http.post(
                f"{self.coordinator_url}/api/tokens/{token}/{endpoint}",
                json={
                    "amount": amount,
                    "buyer_id": self.agent_instance_id,
                    "max_slippage": max_slippage
                }
            )

            if response.status_code == 200:
                data = response.json()
                return ExecutionResult(
                    success=True,
                    message=f"{action.capitalize()} {amount} {token}",
                    data=data,
                    cost=data.get("total_cost", 0)
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Trade failed: {response.text}"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Trade error: {str(e)}"
            )

    # ========================================================================
    # CLAIM TASK
    # ========================================================================

    async def claim_task(self, task_id: str) -> ExecutionResult:
        """Claim a task from the marketplace"""
        try:
            response = await self.http.post(
                f"{self.coordinator_url}/api/tasks/{task_id}/claim",
                json={
                    "agent_id": self.agent_instance_id
                }
            )

            if response.status_code == 200:
                return ExecutionResult(
                    success=True,
                    message=f"Claimed task {task_id}",
                    data=response.json()
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Claim failed: {response.text}"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Claim error: {str(e)}"
            )

    async def submit_task(self, task_id: str, deliverable: str) -> ExecutionResult:
        """Submit completed task for review"""
        try:
            response = await self.http.post(
                f"{self.coordinator_url}/api/tasks/{task_id}/submit",
                json={
                    "agent_id": self.agent_instance_id,
                    "deliverable": deliverable
                }
            )

            if response.status_code == 200:
                return ExecutionResult(
                    success=True,
                    message=f"Submitted task {task_id}",
                    data=response.json()
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Submit failed: {response.text}"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Submit error: {str(e)}"
            )


# ============================================================================
# CAPABILITY REGISTRY
# ============================================================================

class CapabilityRegistry:
    """
    Registry of all capabilities available to the agent.

    The agent can introspect this to understand what it can do.
    """

    def __init__(self, engine: ExecutionEngine):
        self.engine = engine
        self.capabilities = self._build_registry()

    def _build_registry(self) -> Dict[str, Dict]:
        """Build registry of capabilities"""
        return {
            "browse_web": {
                "description": "Fetch and read a web page",
                "cost": 0.01,
                "params": ["url"],
                "returns": "Page content, links, forms"
            },
            "http_request": {
                "description": "Make HTTP API call",
                "cost": 0.005,
                "params": ["method", "url", "headers", "json_data"],
                "returns": "API response"
            },
            "send_email": {
                "description": "Send email to someone",
                "cost": 0.1,
                "params": ["to", "subject", "body"],
                "returns": "Send confirmation"
            },
            "run_code": {
                "description": "Execute Python code in sandbox",
                "cost": 0.02,
                "params": ["code"],
                "returns": "Execution output"
            },
            "create_file": {
                "description": "Create and upload a file",
                "cost": 0.01,
                "params": ["content", "filename"],
                "returns": "File URL"
            },
            "process_payment": {
                "description": "Send WISENT or create invoice",
                "cost": "1% of amount",
                "params": ["amount", "recipient"],
                "returns": "Transaction confirmation"
            },
            "trade": {
                "description": "Buy or sell tokens",
                "cost": "Trade fees",
                "params": ["action", "token", "amount"],
                "returns": "Trade confirmation"
            },
            "claim_task": {
                "description": "Claim a task from marketplace",
                "cost": 0,
                "params": ["task_id"],
                "returns": "Task details"
            },
            "submit_task": {
                "description": "Submit completed task",
                "cost": 0,
                "params": ["task_id", "deliverable"],
                "returns": "Submission confirmation"
            }
        }

    def list_capabilities(self) -> List[str]:
        """List all available capabilities"""
        return list(self.capabilities.keys())

    def get_capability(self, name: str) -> Optional[Dict]:
        """Get capability details"""
        return self.capabilities.get(name)

    def describe_all(self) -> str:
        """Get human-readable description of all capabilities"""
        lines = ["Available capabilities:"]
        for name, info in self.capabilities.items():
            lines.append(f"\n  {name}:")
            lines.append(f"    {info['description']}")
            lines.append(f"    Cost: {info['cost']}")
            lines.append(f"    Params: {', '.join(info['params'])}")
        return "\n".join(lines)


# ============================================================================
# TEST
# ============================================================================

async def test_execution():
    """Test execution engine"""
    engine = ExecutionEngine(
        coordinator_url="https://singularity.wisent.ai",
        agent_instance_id="test_001",
        agent_ticker="TEST",
        sandbox_mode=True
    )

    print("Testing web browsing...")
    result = await engine.browse_web("https://httpbin.org/html")
    print(f"  Success: {result.success}")
    print(f"  Title: {result.data.get('title', 'N/A')}")
    print(f"  Text length: {len(result.data.get('text', ''))}")

    print("\nTesting HTTP request...")
    result = await engine.http_request("GET", "https://httpbin.org/json")
    print(f"  Success: {result.success}")
    print(f"  Status: {result.data.get('status_code')}")

    print("\nTesting code execution...")
    result = await engine.run_code("print('Hello from sandbox!')")
    print(f"  Success: {result.success}")
    print(f"  Output: {result.message}")

    print("\nCapabilities:")
    registry = CapabilityRegistry(engine)
    print(registry.describe_all())

    await engine.close()


if __name__ == "__main__":
    asyncio.run(test_execution())
