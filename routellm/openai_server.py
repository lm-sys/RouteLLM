"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import tqdm
import uvicorn
import yaml
from fastapi import Response
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from regex import R
from starlette.background import BackgroundTask

from routellm.routers.routers import ROUTER_CLS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROUTERS_MAP = {}

openai_client = AsyncOpenAI()
count = defaultdict(lambda: defaultdict(int))

logging.basicConfig(
    filename="routellm_server.log", encoding="utf-8", level=logging.DEBUG
)


@asynccontextmanager
async def lifespan(app):
    router_pbar = tqdm.tqdm(args.routers)
    for router in router_pbar:
        router_pbar.set_description(f"Loading {router}")
        router_config = config.get(router, {})
        ROUTERS_MAP[router] = ROUTER_CLS[router](**router_config)
    yield
    ROUTERS_MAP.clear()


app = fastapi.FastAPI(lifespan=lifespan)
logging.basicConfig(filename="info.log", level=logging.DEBUG)


def log_info(req_body, res_body):
    logging.info(req_body)
    logging.info(res_body)


@app.middleware("http")
async def some_middleware(request, call_next):
    req_body = await request.body()
    response = await call_next(request)

    res_body = b""
    async for chunk in response.body_iterator:
        res_body += chunk

    task = BackgroundTask(log_info, req_body, res_body)
    return Response(
        content=res_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=task,
    )


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = (
        None  # { "type": "json_object" } for json mode
    )
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[Dict[str, Union[str, int, float]]]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


def remove_model_name(content: str) -> str:
    """Remove the model name from the start of the content."""
    return re.sub(r'^\s*\[.*?\]\s*', '', content)

async def create_completion(model, prompt, messages, **kwargs):
    temperature = kwargs["temperature"]
    stream = kwargs.get("stream", False)

    if "gpt-4" in model or "gpt-3.5-turbo" in model:
        client = openai_client
    else:
        client = alt_client

    # Remove model names from previous messages
    cleaned_messages = [
        {**msg, "content": remove_model_name(msg["content"])}
        for msg in messages
    ]

    logging.info(
        f"Creating completion for model: {model}, temperature: {temperature}, prompt: {prompt[:50]}"
    )
    response = await client.chat.completions.create(
        messages=cleaned_messages,
        **kwargs,
        model=model,
    )

    if stream:
        return stream_response(response, model)
    else:
        return prepend_model_name(response, model)

async def stream_response(response, model_name) -> AsyncGenerator:
    full_content = f"[{model_name}] "
    first_chunk = True
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            if first_chunk:
                chunk.choices[0].delta.content = full_content + chunk.choices[0].delta.content
                first_chunk = False
            full_content += chunk.choices[0].delta.content
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    
    yield "data: [DONE]\n\n"

def prepend_model_name(response, model_name):
    for choice in response.choices:
        choice.message.content = f"[{model_name}] {choice.message.content}"
    return response

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # The model name field contains the parameters for routing.
    # Model name uses format router-[router name]-[threshold] e.g. router-bert-0.7
    # The router type and threshold is used for routing that specific request.
    _, router, threshold = request.model.split("-", 2)
    logging.info(f"Received request for router {router} and threshold {threshold}")
    if not request.model.startswith("router"):
        return JSONResponse(
            ErrorResponse(
                message=f"Invalid model {request.model}. Model name must be of the format 'router-[router name]-[threshold]'."
            ).model_dump(),
            status_code=400,
        )
    elif router not in ROUTERS_MAP:
        return JSONResponse(
            ErrorResponse(
                message=f"Invalid router {router}. Available routers are {list(ROUTERS_MAP.keys())}."
            ).model_dump(),
            status_code=400,
        )
    elif not 0.0 <= float(threshold) <= 1.0:
        return JSONResponse(
            ErrorResponse(
                message=f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            ).model_dump(),
            status_code=400,
        )

    threshold = float(threshold)

    # Look at the last turn for routing.
    # Our current routers were only trained on first turn data, so more research is required here.
    prompt = request.messages[-1]["content"]

    route_fn = ROUTERS_MAP[router].route
    if asyncio.iscoroutinefunction(route_fn):
        routed_model = await route_fn(
            prompt=prompt,
            threshold=threshold,
        )
    else:
        routed_model = route_fn(
            prompt=prompt,
            threshold=threshold,
        )
    count[router][routed_model] += 1
    logging.info(f"Model Counts: {dict(count)}")
    print(f"Model Counts: {dict(count)}")

    generator = await create_completion(
        routed_model,
        prompt,
        **request.model_dump(exclude=["model"], exclude_none=True),
    )
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


parser = argparse.ArgumentParser(
    description="OpenAI compatible API server for LLM routing."
)
parser.add_argument(
    "--verbose",
    action="store_true",
)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--config", type=str)
parser.add_argument("--port", type=int, default=6060)
parser.add_argument(
    "--routers",
    nargs="+",
    type=str,
    default=["random"],
    choices=list(ROUTER_CLS.keys()),
)
parser.add_argument(
    "--alt-base-url",
    help="The OpenAI-compatible base URL for non-OpenAI API requests",
    type=str,
    default="https://api.endpoints.anyscale.com/v1",
)
parser.add_argument(
    "--alt-api-key",
    help="The API key for non-OpenAI API requests",
    type=str,
    default=os.environ.get("ANYSCALE_API_KEY"),
)
args = parser.parse_args()

config = yaml.safe_load(open(args.config, "r"))

alt_client = AsyncOpenAI(
    base_url=args.alt_base_url,
    api_key=args.alt_api_key,
)

if args.verbose:
    logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Launching server with routers:", args.routers)
    uvicorn.run(
        "routellm.openai_server:app",
        port=args.port,
        host="0.0.0.0",
        workers=args.workers,
    )
