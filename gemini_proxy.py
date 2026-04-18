"""
OpenAI-compatible proxy that routes requests to the Gemini CLI.
Runs on http://localhost:8080/v1 — RouteLLM uses this as its strong model.
"""

import asyncio
import json
import shutil
import subprocess
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Gemini CLI Proxy")

# Resolve the full path to gemini (handles Windows .cmd wrappers)
GEMINI_CMD = shutil.which("gemini") or "gemini"
REQUEST_TIMEOUT = 120


def messages_to_prompt(messages: list[dict]) -> str:
    """Flatten OpenAI message list into a single prompt string for Gemini CLI."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if role == "system":
            parts.append(f"[System instructions: {content}]")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = messages_to_prompt(messages)

    def run_gemini() -> str:
        result = subprocess.run(
            [GEMINI_CMD, "-p", prompt],
            capture_output=True,
            timeout=REQUEST_TIMEOUT,
            shell=False,
        )
        return result.stdout.decode("utf-8", errors="replace").strip()

    try:
        response_text = await asyncio.wait_for(
            asyncio.to_thread(run_gemini),
            timeout=REQUEST_TIMEOUT + 5,
        )
    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
        return JSONResponse({"error": {"message": "Gemini CLI timed out", "type": "timeout"}}, status_code=504)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gemini-cli",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split()),
        },
    })


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "gemini-cli",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
        }],
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
