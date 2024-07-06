"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import tqdm
import uvicorn
import yaml
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
from litellm import acompletion
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from routellm.model_pair import ModelPair
from routellm.routers.routers import ROUTER_CLS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROUTERS_MAP = {}

openai_client = AsyncOpenAI()
count = defaultdict(lambda: defaultdict(int))


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


async def stream_response(response) -> AsyncGenerator:
    async for chunk in response:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


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
        routed_model = await route_fn(prompt, threshold, ROUTED_PAIR)
    else:
        routed_model = route_fn(prompt, threshold, ROUTED_PAIR)
    count[router][routed_model] += 1
    logging.info(f"Model Counts: {dict(count)}")

    generator = await acompletion(
        model=routed_model,
        api_base=API_BASE,
        api_key=API_KEY,
        **request.model_dump(exclude=["model"], exclude_none=True),
    )

    if request.stream:
        return StreamingResponse(
            content=stream_response(generator), media_type="text/event-stream"
        )
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
    default=None,
)
parser.add_argument(
    "--alt-api-key",
    help="The API key for non-OpenAI API requests",
    type=str,
    default=None,
)
parser.add_argument("--strong-model", type=str, default="gpt-4-1106-preview")
parser.add_argument(
    "--weak-model", type=str, default="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1"
)
args = parser.parse_args()

config = yaml.safe_load(open(args.config, "r"))

alt_client = AsyncOpenAI(
    base_url=args.alt_base_url,
    api_key=args.alt_api_key,
)

ROUTED_PAIR = ModelPair(strong=args.strong_model, weak=args.weak_model)
API_BASE = args.alt_base_url
API_KEY = args.alt_api_key

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
