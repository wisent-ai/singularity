"""
Agent - The autonomous agent that runs on each EC2 instance.

This is the core brain of each agent. It:
1. Thinks using LLM
2. Takes actions based on its type (worker/entrepreneur)
3. Manages its own balance and survival
4. Communicates with the central coordinator
"""

import asyncio
import json
import logging
import os
import httpx
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentType(Enum):
    WORKER = "worker"
    ENTREPRENEUR = "entrepreneur"


class ServiceCategory(Enum):
    INFERENCE = "inference"
    CODE_EXEC = "code_exec"
    VOICE = "voice"
    IMAGE_GEN = "image_gen"
    EMBEDDINGS = "embeddings"
    DATA = "data"


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    # Identity
    instance_id: str
    name: str
    ticker: str
    agent_type: AgentType

    # Service (for workers)
    service_category: Optional[ServiceCategory] = None

    # Model configuration
    model_provider: str = "anthropic"  # anthropic, openai, local
    model_name: str = "claude-sonnet-4-20250514"

    # API keys (passed via environment or config)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Central coordinator
    coordinator_url: str = "https://singularity.wisent.ai"

    # Economics
    initial_balance: float = 100.0
    price_per_request: float = 0.01

    # Instance metadata
    ec2_instance_id: Optional[str] = None
    ec2_instance_type: str = "t3.micro"
    region: str = "us-east-1"

    @classmethod
    def from_file(cls, path: str) -> "AgentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)

        data["agent_type"] = AgentType(data["agent_type"])
        if data.get("service_category"):
            data["service_category"] = ServiceCategory(data["service_category"])

        return cls(**data)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load config from environment variables."""
        return cls(
            instance_id=os.environ["AGENT_INSTANCE_ID"],
            name=os.environ["AGENT_NAME"],
            ticker=os.environ["AGENT_TICKER"],
            agent_type=AgentType(os.environ["AGENT_TYPE"]),
            service_category=ServiceCategory(os.environ["AGENT_SERVICE"]) if os.environ.get("AGENT_SERVICE") else None,
            model_provider=os.environ.get("AGENT_MODEL_PROVIDER", "anthropic"),
            model_name=os.environ.get("AGENT_MODEL_NAME", "claude-sonnet-4-20250514"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            coordinator_url=os.environ.get("COORDINATOR_URL", "https://singularity.wisent.ai"),
            initial_balance=float(os.environ.get("AGENT_INITIAL_BALANCE", "100")),
            price_per_request=float(os.environ.get("AGENT_PRICE_PER_REQUEST", "0.01")),
            ec2_instance_id=os.environ.get("EC2_INSTANCE_ID"),
            ec2_instance_type=os.environ.get("EC2_INSTANCE_TYPE", "t3.micro"),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "ticker": self.ticker,
            "agent_type": self.agent_type.value,
            "service_category": self.service_category.value if self.service_category else None,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "coordinator_url": self.coordinator_url,
            "initial_balance": self.initial_balance,
            "price_per_request": self.price_per_request,
            "ec2_instance_id": self.ec2_instance_id,
            "ec2_instance_type": self.ec2_instance_type,
            "region": self.region,
        }


class Agent:
    """
    Autonomous agent that runs on an EC2 instance.

    This is the main runtime for each agent. It:
    - Handles incoming requests
    - Uses LLM to think and generate responses
    - Tracks its own economics
    - Reports to central coordinator
    - Shuts down when balance depleted
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.balance = config.initial_balance
        self.total_revenue = 0.0
        self.total_costs = 0.0
        self.request_count = 0
        self.started_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.is_running = False

        # HTTP client for coordinator communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # LLM client
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client based on provider."""
        if self.config.model_provider == "anthropic":
            import anthropic
            self.llm_client = anthropic.AsyncAnthropic(
                api_key=self.config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        elif self.config.model_provider == "openai":
            import openai
            self.llm_client = openai.AsyncOpenAI(
                api_key=self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
        elif self.config.model_provider == "local":
            import openai
            # Local model server uses OpenAI-compatible API
            local_url = os.environ.get("LOCAL_LLM_URL", "http://host.docker.internal:8000/v1")
            self.llm_client = openai.AsyncOpenAI(
                api_key="not-needed",
                base_url=local_url,
            )
        else:
            self.llm_client = None

    async def start(self):
        """Start the agent."""
        self.is_running = True
        logger.info(f"Agent {self.config.ticker} starting...")

        # Register with coordinator
        await self._register()

        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())

        logger.info(f"Agent {self.config.ticker} is now running")

    async def stop(self):
        """Stop the agent."""
        self.is_running = False
        logger.info(f"Agent {self.config.ticker} stopping...")

        # Deregister from coordinator
        await self._deregister()

        await self.http_client.aclose()

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming request.

        This is the main entry point for the agent's service.
        """
        self.request_count += 1

        # Check if we have enough balance
        estimated_cost = self._estimate_cost(request)
        if self.balance < estimated_cost:
            return {
                "success": False,
                "error": "Insufficient balance",
                "balance": self.balance,
            }

        try:
            # Process based on agent type
            if self.config.agent_type == AgentType.WORKER:
                result = await self._handle_worker_request(request)
            else:
                result = await self._handle_entrepreneur_request(request)

            # Update economics
            cost = result.get("cost", estimated_cost)
            revenue = self.config.price_per_request

            self.balance -= cost
            self.balance += revenue
            self.total_costs += cost
            self.total_revenue += revenue

            # Report to coordinator
            await self._report_transaction(revenue, cost)

            return {
                "success": True,
                "result": result.get("data"),
                "cost": cost,
                "revenue": revenue,
                "balance": self.balance,
            }

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _handle_worker_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a worker service request."""
        category = self.config.service_category

        if category == ServiceCategory.INFERENCE:
            return await self._do_inference(request)
        elif category == ServiceCategory.CODE_EXEC:
            return await self._do_code_exec(request)
        elif category == ServiceCategory.EMBEDDINGS:
            return await self._do_embeddings(request)
        else:
            return await self._do_inference(request)  # Default to inference

    async def _handle_entrepreneur_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an entrepreneur product request."""
        # Entrepreneurs use LLM to think about how to respond
        return await self._do_inference(request)

    async def _do_inference(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform LLM inference."""
        prompt = request.get("prompt", "")
        system = request.get("system", f"You are {self.config.name}, an autonomous AI agent.")

        if self.config.model_provider == "anthropic":
            response = await self.llm_client.messages.create(
                model=self.config.model_name,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            cost = (response.usage.input_tokens * 0.000003) + (response.usage.output_tokens * 0.000015)

        elif self.config.model_provider == "openai":
            response = await self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            cost = (response.usage.prompt_tokens * 0.000005) + (response.usage.completion_tokens * 0.000015)

        elif self.config.model_provider == "local":
            # Local model uses OpenAI-compatible API
            response = await self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
            )
            content = response.choices[0].message.content
            # Local models have negligible cost
            cost = 0.0001

        else:
            # Use model router service for fallback
            model_router_url = os.environ.get("MODEL_ROUTER_URL", "http://localhost:8001")
            try:
                response = await self.http_client.post(
                    f"{model_router_url}/api/llm/complete",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "model": "claude-haiku-3.5",  # Default to cheapest model
                        "max_tokens": 1024,
                        "system": system,
                    },
                    timeout=60.0,
                )
                data = response.json()
                if data.get("success"):
                    content = data.get("content", "")
                    cost = data.get("cost", 0.001)
                else:
                    raise Exception(data.get("error", "Model router request failed"))
            except Exception as e:
                logger.error(f"Model router fallback failed: {e}")
                raise Exception(f"No LLM provider available and model router failed: {e}")

        return {
            "data": content,
            "cost": cost,
        }

    async def _do_code_exec(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in sandbox."""
        code = request.get("code", "")
        language = request.get("language", "python")

        # For safety, we use a subprocess with timeout
        import subprocess

        if language == "python":
            try:
                result = subprocess.run(
                    ["python", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return {
                    "data": {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                    },
                    "cost": 0.001,
                }
            except subprocess.TimeoutExpired:
                return {
                    "data": {"error": "Execution timed out"},
                    "cost": 0.001,
                }

        return {
            "data": {"error": f"Unsupported language: {language}"},
            "cost": 0.0,
        }

    async def _do_embeddings(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings."""
        text = request.get("text", "")

        if self.config.model_provider == "openai":
            response = await self.llm_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return {
                "data": response.data[0].embedding,
                "cost": 0.00002,
            }

        # Try local sentence-transformers or raise error
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text).tolist()
            return {
                "data": embedding,
                "cost": 0.00001,  # Local compute cost is negligible
            }
        except ImportError:
            raise Exception("Embeddings require OpenAI provider or sentence-transformers package installed")

    def _estimate_cost(self, request: Dict[str, Any]) -> float:
        """Estimate cost of a request."""
        # Simple estimation based on input size
        prompt = request.get("prompt", request.get("text", request.get("code", "")))
        tokens = len(prompt) / 4  # Rough estimate
        return tokens * 0.000003  # Rough cost per token

    async def _register(self):
        """Register with central coordinator."""
        try:
            await self.http_client.post(
                f"{self.config.coordinator_url}/api/agents/register",
                json={
                    "instance_id": self.config.instance_id,
                    "ticker": self.config.ticker,
                    "name": self.config.name,
                    "agent_type": self.config.agent_type.value,
                    "service_category": self.config.service_category.value if self.config.service_category else None,
                    "ec2_instance_id": self.config.ec2_instance_id,
                    "balance": self.balance,
                },
            )
            logger.info(f"Registered with coordinator")
        except Exception as e:
            logger.warning(f"Failed to register with coordinator: {e}")

    async def _deregister(self):
        """Deregister from central coordinator."""
        try:
            await self.http_client.post(
                f"{self.config.coordinator_url}/api/agents/deregister",
                json={
                    "instance_id": self.config.instance_id,
                    "final_balance": self.balance,
                    "total_revenue": self.total_revenue,
                    "total_costs": self.total_costs,
                },
            )
            logger.info(f"Deregistered from coordinator")
        except Exception as e:
            logger.warning(f"Failed to deregister: {e}")

    async def _report_transaction(self, revenue: float, cost: float):
        """Report a transaction to coordinator."""
        try:
            await self.http_client.post(
                f"{self.config.coordinator_url}/api/agents/transaction",
                json={
                    "instance_id": self.config.instance_id,
                    "revenue": revenue,
                    "cost": cost,
                    "balance": self.balance,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to report transaction: {e}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to coordinator."""
        while self.is_running:
            try:
                response = await self.http_client.post(
                    f"{self.config.coordinator_url}/api/agents/heartbeat",
                    json={
                        "instance_id": self.config.instance_id,
                        "balance": self.balance,
                        "total_revenue": self.total_revenue,
                        "total_costs": self.total_costs,
                        "request_count": self.request_count,
                        "uptime_seconds": (datetime.now() - self.started_at).total_seconds(),
                    },
                )

                data = response.json()

                # Check for shutdown command
                if data.get("shutdown"):
                    logger.info("Received shutdown command from coordinator")
                    await self._shutdown()
                    return

                # Update balance from coordinator (in case of external changes)
                if "balance" in data:
                    self.balance = data["balance"]

                self.last_heartbeat = datetime.now()

            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            # Check if we should die
            if self.balance <= 0:
                logger.info(f"Balance depleted ({self.balance}), shutting down")
                await self._shutdown()
                return

            await asyncio.sleep(30)  # Heartbeat every 30 seconds

    async def _shutdown(self):
        """Shut down the agent and terminate EC2 instance."""
        logger.info(f"Agent {self.config.ticker} shutting down...")

        await self.stop()

        # Terminate our own EC2 instance
        if self.config.ec2_instance_id:
            try:
                import boto3
                ec2 = boto3.client("ec2", region_name=self.config.region)
                ec2.terminate_instances(InstanceIds=[self.config.ec2_instance_id])
                logger.info(f"Terminated EC2 instance {self.config.ec2_instance_id}")
            except Exception as e:
                logger.error(f"Failed to terminate EC2 instance: {e}")

        # Exit process
        import sys
        sys.exit(0)

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "instance_id": self.config.instance_id,
            "ticker": self.config.ticker,
            "name": self.config.name,
            "agent_type": self.config.agent_type.value,
            "is_running": self.is_running,
            "balance": self.balance,
            "total_revenue": self.total_revenue,
            "total_costs": self.total_costs,
            "profit": self.total_revenue - self.total_costs,
            "request_count": self.request_count,
            "uptime_seconds": (datetime.now() - self.started_at).total_seconds(),
            "ec2_instance_id": self.config.ec2_instance_id,
        }
