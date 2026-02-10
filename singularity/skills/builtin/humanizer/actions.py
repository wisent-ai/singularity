"""
Humanizer actions: humanize text via AuthorMist (local or endpoint).
"""

import asyncio
from typing import Dict

from singularity.skills.base import SkillResult


PROMPT_TEMPLATE = """Please paraphrase the following text to make it more human-like while preserving the original meaning:

{text}

Paraphrased text:"""


async def humanize(skill, params: Dict) -> SkillResult:
    """Paraphrase text through AuthorMist to make it undetectable."""
    text = params.get("text")
    if not text:
        return SkillResult(success=False, message="Missing required parameter: text")

    temperature = float(params.get("temperature", 0.7))
    max_tokens = int(params.get("max_tokens", 1024))
    temperature = max(0.1, min(1.0, temperature))

    # Use inference endpoint if configured
    if skill._endpoint:
        return await _humanize_endpoint(skill, text, temperature, max_tokens)

    return await _humanize_local(skill, text, temperature, max_tokens)


async def humanize_chunks(skill, params: Dict) -> SkillResult:
    """Humanize long text by splitting into chunks."""
    text = params.get("text")
    if not text:
        return SkillResult(success=False, message="Missing required parameter: text")

    chunk_size = int(params.get("chunk_size", 400))
    temperature = float(params.get("temperature", 0.7))
    temperature = max(0.1, min(1.0, temperature))

    words = text.split()
    if len(words) <= chunk_size:
        return await humanize(skill, {"text": text, "temperature": str(temperature)})

    # Split into chunks at sentence boundaries where possible
    chunks = _split_into_chunks(text, chunk_size)
    humanized_parts = []

    for i, chunk in enumerate(chunks):
        if skill._endpoint:
            result = await _humanize_endpoint(skill, chunk, temperature, 1024)
        else:
            result = await _humanize_local(skill, chunk, temperature, 1024)

        if not result.success:
            return SkillResult(
                success=False,
                message=f"Failed on chunk {i + 1}/{len(chunks)}: {result.message}",
                data={"completed_chunks": i, "total_chunks": len(chunks)}
            )
        humanized_parts.append(result.data.get("humanized_text", ""))

    full_text = " ".join(humanized_parts)
    return SkillResult(
        success=True,
        message=f"Humanized {len(chunks)} chunks ({len(words)} words)",
        data={
            "humanized_text": full_text,
            "original_length": len(words),
            "output_length": len(full_text.split()),
            "chunks_processed": len(chunks)
        }
    )


async def check_model(skill, params: Dict) -> SkillResult:
    """Check if AuthorMist model is loaded and ready."""
    if skill._endpoint:
        try:
            resp = await skill.http.get(skill._endpoint, headers=_endpoint_headers(skill))
            return SkillResult(
                success=True,
                message=f"Using HF Inference Endpoint (status: {resp.status_code})",
                data={"mode": "endpoint", "url": skill._endpoint, "status": resp.status_code}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Endpoint unreachable: {e}")

    loaded = skill._model is not None
    if loaded:
        return SkillResult(
            success=True,
            message=f"AuthorMist loaded on {skill._device}",
            data={"mode": "local", "device": skill._device, "loaded": True}
        )

    # Check if transformers + torch are available
    try:
        import torch
        import transformers  # noqa: F401
        gpu = torch.cuda.is_available()
        return SkillResult(
            success=True,
            message=f"Model not loaded yet (GPU: {gpu}). Will load on first humanize call.",
            data={"mode": "local", "loaded": False, "gpu_available": gpu}
        )
    except ImportError as e:
        return SkillResult(
            success=False,
            message=f"Missing dependency: {e}. Install torch + transformers.",
            data={"mode": "unavailable", "error": str(e)}
        )


# ── Internal helpers ──────────────────────────────────────────────────

async def _humanize_local(skill, text: str, temperature: float, max_tokens: int) -> SkillResult:
    """Run AuthorMist locally via transformers."""
    loop = asyncio.get_event_loop()

    def _generate():
        import torch
        skill._load_model()

        prompt = PROMPT_TEMPLATE.format(text=text)
        inputs = skill._tokenizer(prompt, return_tensors="pt").to(skill._device)

        with torch.no_grad():
            outputs = skill._model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=skill._tokenizer.eos_token_id,
            )

        full_output = skill._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract text after the "Paraphrased text:" marker
        marker = "Paraphrased text:"
        if marker in full_output:
            return full_output.split(marker, 1)[1].strip()
        # Fallback: return everything after the input prompt
        return full_output[len(prompt):].strip()

    humanized = await loop.run_in_executor(None, _generate)

    if not humanized:
        return SkillResult(success=False, message="Model produced empty output")

    return SkillResult(
        success=True,
        message=f"Humanized {len(text.split())} words (local, {skill._device})",
        data={
            "humanized_text": humanized,
            "original_length": len(text.split()),
            "output_length": len(humanized.split()),
            "mode": "local",
            "device": skill._device
        }
    )


async def _humanize_endpoint(skill, text: str, temperature: float, max_tokens: int) -> SkillResult:
    """Run AuthorMist via HuggingFace Inference Endpoint."""
    prompt = PROMPT_TEMPLATE.format(text=text)

    resp = await skill.http.post(
        skill._endpoint,
        headers=_endpoint_headers(skill),
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": True,
            }
        }
    )

    if resp.status_code != 200:
        return SkillResult(success=False, message=f"Endpoint error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    if isinstance(data, list) and data:
        generated = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        generated = data.get("generated_text", "")
    else:
        return SkillResult(success=False, message=f"Unexpected response: {str(data)[:200]}")

    # Extract after marker
    marker = "Paraphrased text:"
    if marker in generated:
        humanized = generated.split(marker, 1)[1].strip()
    else:
        humanized = generated[len(prompt):].strip() if generated.startswith(prompt[:50]) else generated.strip()

    if not humanized:
        return SkillResult(success=False, message="Endpoint returned empty output")

    return SkillResult(
        success=True,
        message=f"Humanized {len(text.split())} words (endpoint)",
        data={
            "humanized_text": humanized,
            "original_length": len(text.split()),
            "output_length": len(humanized.split()),
            "mode": "endpoint"
        }
    )


def _endpoint_headers(skill) -> Dict:
    headers = {"Content-Type": "application/json"}
    if skill._hf_token:
        headers["Authorization"] = f"Bearer {skill._hf_token}"
    return headers


def _split_into_chunks(text: str, chunk_size: int) -> list:
    """Split text into chunks, preferring sentence boundaries."""
    sentences = text.replace(". ", ".\n").split("\n")
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        words = len(sentence.split())
        if current_len + words > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += words

    if current:
        chunks.append(" ".join(current))
    return chunks
