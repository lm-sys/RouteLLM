import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock, patch

routers_module = ModuleType("routellm.routers.routers")
routers_module.ROUTER_CLS = {}
sys.modules["routellm.routers.routers"] = routers_module

from routellm.controller import BAI_API_BASE, Controller

MESSAGES = [{"role": "user", "content": "hello"}]


class StubRouter:
    def __init__(self, model):
        self.model = model

    def route(self, prompt, threshold, model_pair):
        return self.model


def controller_for_model(model, **kwargs):
    controller = Controller(
        routers=[],
        strong_model="strong",
        weak_model="weak",
        **kwargs,
    )
    controller.routers["random"] = StubRouter(model)
    return controller


def test_completion_uses_bai_defaults(monkeypatch):
    monkeypatch.setenv("BAI_API_KEY", "bai-key")
    monkeypatch.delenv("BAI_API_BASE", raising=False)
    controller = controller_for_model("bai/gpt-5.2")

    with patch("routellm.controller.completion", Mock()) as completion:
        controller.completion(router="random", threshold=0.5, messages=MESSAGES)

    completion.assert_called_once_with(
        api_base=BAI_API_BASE,
        api_key="bai-key",
        model="openai/gpt-5.2",
        messages=MESSAGES,
    )


def test_completion_uses_explicit_bai_api_overrides(monkeypatch):
    monkeypatch.setenv("BAI_API_KEY", "env-key")
    monkeypatch.setenv("BAI_API_BASE", "https://env.example/v1")
    controller = controller_for_model(
        "bai/gpt-5.2",
        api_base="https://custom.example/v1",
        api_key="custom-key",
    )

    with patch("routellm.controller.completion", Mock()) as completion:
        controller.completion(router="random", threshold=0.5, messages=MESSAGES)

    completion.assert_called_once_with(
        api_base="https://custom.example/v1",
        api_key="custom-key",
        model="openai/gpt-5.2",
        messages=MESSAGES,
    )


def test_acompletion_uses_bai_defaults(monkeypatch):
    monkeypatch.setenv("BAI_API_KEY", "bai-key")
    monkeypatch.delenv("BAI_API_BASE", raising=False)
    controller = controller_for_model("bai/gpt-5.2")

    with patch("routellm.controller.acompletion", AsyncMock()) as acompletion:
        asyncio.run(
            controller.acompletion(router="random", threshold=0.5, messages=MESSAGES)
        )

    acompletion.assert_awaited_once_with(
        api_base=BAI_API_BASE,
        api_key="bai-key",
        model="openai/gpt-5.2",
        messages=MESSAGES,
    )


def test_completion_leaves_non_bai_model_unchanged(monkeypatch):
    monkeypatch.setenv("BAI_API_KEY", "bai-key")
    controller = controller_for_model("anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1")

    with patch("routellm.controller.completion", Mock()) as completion:
        controller.completion(router="random", threshold=0.5, messages=MESSAGES)

    completion.assert_called_once_with(
        api_base=None,
        api_key=None,
        model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=MESSAGES,
    )
