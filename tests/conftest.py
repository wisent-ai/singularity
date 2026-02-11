"""
Test configuration — patch missing dependencies before any imports
from the singularity package.

This runs before every test module to ensure optional dependencies
that may not be installed in the test environment are mocked.
"""

import sys
import types as builtins_types


def _create_mock_module(name):
    """Create a mock module and register it in sys.modules."""
    mod = builtins_types.ModuleType(name)
    mod.__dict__.setdefault("__all__", [])
    sys.modules[name] = mod
    return mod


# List of all external dependencies that may not be installed
MOCK_MODULES = [
    "dotenv", "python_dotenv",
    "anthropic",
    "openai",
    "httpx",
    "aiohttp",
    "tweepy",
    "playwright", "playwright.async_api",
    "PIL", "PIL.Image",
    "yaml", "pyyaml",
    "stripe",
    "web3",
    "resend",
    "bs4", "beautifulsoup4",
    "google", "google.cloud", "google.cloud.aiplatform",
    "torch", "transformers", "vllm",
    "cognee",
]


class _MockClass:
    """Generic mock class that absorbs all attribute access and calls."""
    _mock_false = False
    def __init__(self, *a, **kw):
        pass
    def __call__(self, *a, **kw):
        return _MockClass()
    def __getattr__(self, name):
        if name == "is_available":
            return lambda: False
        return _MockClass()
    def __bool__(self):
        return False
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        pass


for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        mock = _create_mock_module(mod_name)
        # Add common attributes that modules expect
        if mod_name == "dotenv":
            mock.load_dotenv = lambda *a, **kw: None
        elif mod_name == "torch":
            # torch needs proper cuda/backends/mps stubs
            class _MockCuda:
                @staticmethod
                def is_available():
                    return False
            class _MockMps:
                @staticmethod
                def is_available():
                    return False
            class _MockBackends:
                mps = _MockMps()
            mock.cuda = _MockCuda()
            mock.backends = _MockBackends()
            mock.device = _MockClass
            mock.Tensor = _MockClass
            mock.float16 = "float16"
            mock.float32 = "float32"
        elif mod_name == "httpx":
            class _MockAsyncClient:
                def __init__(self, **kw):
                    self.headers = {}
                async def get(self, *a, **kw):
                    raise NotImplementedError("httpx mocked in tests")
                async def post(self, *a, **kw):
                    raise NotImplementedError("httpx mocked in tests")
                async def put(self, *a, **kw):
                    raise NotImplementedError("httpx mocked in tests")
                async def delete(self, *a, **kw):
                    raise NotImplementedError("httpx mocked in tests")
                async def aclose(self):
                    pass
            mock.AsyncClient = _MockAsyncClient
        elif mod_name == "aiohttp":
            mock.ClientSession = _MockClass
        elif mod_name == "anthropic":
            mock.Anthropic = _MockClass
            mock.AsyncAnthropic = _MockClass
        elif mod_name == "openai":
            mock.OpenAI = _MockClass
            mock.AsyncOpenAI = _MockClass
        elif mod_name == "stripe":
            mock.PaymentLink = _MockClass
            mock.Product = _MockClass
            mock.Price = _MockClass
        elif mod_name == "playwright.async_api":
            mock.async_playwright = _MockClass
            mock.Browser = _MockClass
            mock.Page = _MockClass
