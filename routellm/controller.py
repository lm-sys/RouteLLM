from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import tqdm
from litellm import acompletion, completion

from routellm.model_pair import ModelPair
from routellm.routers.routers import ROUTER_CLS


class RoutingError(Exception):
    pass


class Controller:
    def __init__(
        self,
        routers: list[str],
        config: dict[str, dict[str, Any]],
        routed_pair: ModelPair,
        api_base: str = None,
        api_key: str = None,
        progress_bar: bool = False,
    ):
        self.routed_pair = routed_pair
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key

        self.model_counts = defaultdict(lambda: defaultdict(int))

        router_pbar = tqdm.tqdm(routers) if progress_bar else None
        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Some Python magic to match the OpenAI Python SDK
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate(self, router: str, threshold: float):
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        elif not isinstance(threshold, float) or not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _get_routed_model_for_completion(self, kwargs: dict[str, Any]):
        if "messages" not in kwargs or not kwargs["messages"]:
            raise ValueError("messages must be a non-empty list")

        model = kwargs["model"]
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        self._validate(router, float(threshold))

        # Look at the last turn for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        prompt = kwargs["messages"][-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.routed_pair)

        self.model_counts[router][routed_model] += 1

        return routed_model

    def route(self, prompt: str, router: str, threshold: float):
        self._validate(router, threshold)

        return self.routers[router].route(prompt, threshold, self.routed_pair)

    # Matches OpenAI's Chat Completions interface
    def completion(self, **kwargs):
        routed_model = self._get_routed_model_for_completion(kwargs)
        kwargs["model"] = routed_model
        return completion(api_base=self.api_base, api_key=self.api_key, **kwargs)

    # Matches OpenAI's Async Chat Completions interface
    async def acompletion(self, **kwargs):
        routed_model = self._get_routed_model_for_completion(kwargs)
        kwargs["model"] = routed_model
        return await acompletion(api_base=self.api_base, api_key=self.api_key, **kwargs)
