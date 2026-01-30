#!/usr/bin/env python3
"""
Simple OpenAI-compatible API server for Qwen3-8B using transformers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
import uuid

app = FastAPI(title="Qwen3 8B Server")

# Global model and tokenizer
model = None
tokenizer = None

import os
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3-8b"
    messages: List[Message]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"Loading {MODEL_NAME}...")

    # Determine device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        dtype = torch.float32
        print("WARNING: Running on CPU - will be slow")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    print(f"Model loaded on {device}!")


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    # Format messages using tokenizer's chat template
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs = inputs.to(model.device)

    prompt_tokens = inputs.input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode
    generated = outputs[0][prompt_tokens:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Clean up end tokens
    if "<|im_end|>" in response_text:
        response_text = response_text.split("<|im_end|>")[0]

    completion_tokens = len(generated)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=response_text.strip()),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-8b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen"
            }
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
