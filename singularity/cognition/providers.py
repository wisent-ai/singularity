"""
Cognition providers - LLM backend detection and generation.

Supports: Anthropic, OpenAI, Vertex AI (Claude + Gemini), vLLM, Transformers.
"""

# CRITICAL: Set multiprocessing start method before importing torch/vllm
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import asyncio
from .types import TokenUsage

# Optional torch import - only needed for local LLM inference
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass


def get_device():
    """Detect available compute device."""
    if not HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from anthropic import AnthropicVertex
    HAS_VERTEX_CLAUDE = True
except ImportError:
    HAS_VERTEX_CLAUDE = False

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    HAS_VERTEX_GEMINI = True
except ImportError:
    HAS_VERTEX_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

HAS_VLLM = False
if HAS_TORCH and DEVICE == "cuda":
    try:
        from vllm import LLM, SamplingParams
        HAS_VLLM = True
    except ImportError:
        pass

HAS_TRANSFORMERS = False
if HAS_TORCH:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        HAS_TRANSFORMERS = True
    except ImportError:
        pass


async def generate_with_backend(
    engine,  # CognitionEngine instance
    prompt: str,
) -> tuple:
    """Generate text using the configured backend (single-turn). Returns (text, token_usage)."""
    messages = [{"role": "user", "content": prompt}]
    return await generate_with_messages(engine, messages)


async def generate_with_messages(
    engine,  # CognitionEngine instance
    messages: list,
    system: str = "",
    max_tokens: int = 16384,
) -> tuple:
    """Generate text using the configured backend with multi-turn messages.

    Args:
        engine: CognitionEngine instance
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts
        system: System prompt (used as separate param for Anthropic/Vertex)
        max_tokens: Max output tokens
    Returns:
        (text, token_usage)
    """

    if engine.llm_type == "vllm":
        prompt = "\n".join(f"{'Human' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages)
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: engine.llm.generate([prompt], engine.sampling_params)
        )
        if not outputs or not outputs[0].outputs:
            return "", TokenUsage()
        output = outputs[0].outputs[0]
        usage = TokenUsage(
            input_tokens=len(outputs[0].prompt_token_ids),
            output_tokens=len(output.token_ids)
        )
        return output.text, usage

    elif engine.llm_type == "transformers":
        prompt = "\n".join(m['content'] for m in messages if m['role'] == 'user')
        loop = asyncio.get_event_loop()
        text, usage = await loop.run_in_executor(
            None, _generate_transformers, engine, prompt
        )
        return text, usage

    elif engine.llm_type == "anthropic":
        kwargs = {"model": engine.llm_model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        response = await engine.llm.messages.create(**kwargs)
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        if not response.content:
            return "", usage
        return response.content[0].text, usage

    elif engine.llm_type == "vertex":
        kwargs = {"model": engine.llm_model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: engine.llm.messages.create(**kwargs)
        )
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        if not response.content:
            return "", usage
        return response.content[0].text, usage

    elif engine.llm_type == "vertex_gemini":
        prompt = "\n".join(m['content'] for m in messages)
        loop = asyncio.get_event_loop()
        model = GenerativeModel(engine.llm_model)
        config = GenerationConfig(max_output_tokens=max_tokens, temperature=0.2)
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt, generation_config=config)
        )
        usage = TokenUsage(
            input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        )
        return response.text, usage

    elif engine.llm_type == "openai":
        oai_messages = messages.copy()
        if system:
            oai_messages = [{"role": "system", "content": system}] + oai_messages
        response = await engine.llm.chat.completions.create(
            model=engine.llm_model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=0.2
        )
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0
        )
        if not response.choices:
            return "", usage
        return response.choices[0].message.content or "", usage

    return "", TokenUsage()


def _generate_transformers(engine, prompt: str) -> tuple:
    """Generate with HuggingFace transformers (sync). Returns (text, token_usage)."""
    messages = [{"role": "user", "content": prompt}]

    if hasattr(engine.tokenizer, 'apply_chat_template'):
        text = engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = engine.tokenizer(text, return_tensors="pt").to(DEVICE)
    input_token_count = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = engine.llm.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=engine.tokenizer.eos_token_id
        )

    output_token_count = outputs.shape[1] - input_token_count

    response = engine.tokenizer.decode(
        outputs[0][input_token_count:],
        skip_special_tokens=True
    )

    usage = TokenUsage(
        input_tokens=input_token_count,
        output_tokens=output_token_count
    )
    return response, usage


# === Available models for runtime switching ===

AVAILABLE_MODELS = {
    "vertex": {
        "gemini-2.0-flash-001": {"cost": "low", "speed": "fast", "capability": "good"},
        "gemini-1.5-flash-002": {"cost": "very_low", "speed": "very_fast", "capability": "basic"},
        "gemini-1.5-pro-002": {"cost": "medium", "speed": "medium", "capability": "excellent"},
        "claude-3-5-sonnet-v2@20241022": {"cost": "medium", "speed": "medium", "capability": "excellent"},
        "claude-3-5-haiku@20241022": {"cost": "low", "speed": "fast", "capability": "good"},
    },
    "anthropic": {
        "claude-sonnet-4-20250514": {"cost": "medium", "speed": "medium", "capability": "excellent"},
        "claude-3-5-sonnet-20241022": {"cost": "medium", "speed": "medium", "capability": "excellent"},
        "claude-3-5-haiku-20241022": {"cost": "low", "speed": "fast", "capability": "good"},
    },
    "openai": {
        "gpt-4o": {"cost": "medium", "speed": "medium", "capability": "excellent"},
        "gpt-4o-mini": {"cost": "low", "speed": "fast", "capability": "good"},
        "gpt-4-turbo": {"cost": "high", "speed": "slow", "capability": "excellent"},
    },
}


def _lazy_import(name):
    """Lazy-import provider modules to avoid loading unused dependencies."""
    if name == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic
    elif name == "vertex_claude":
        from anthropic import AnthropicVertex
        return AnthropicVertex
    elif name == "vertexai":
        import vertexai
        return vertexai
    elif name == "openai":
        import openai
        return openai
    elif name == "vllm":
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    elif name == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
