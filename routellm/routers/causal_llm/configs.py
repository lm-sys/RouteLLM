from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict

PROMPT_FORMAT_CONFIGS = {
    "meta-llama/Llama-2-7b-hf": {
        "system": "<<SYS>>\n{instruction}\n<</SYS>>\n\n",
        "assistant": " {instruction} </s><s>",
        "trailing_assistant": "",
        "user": "[INST] {system}{instruction} [/INST]",
        "system_in_user": True,
        "default_system_message": "",
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "system": "<<SYS>>\n{instruction}\n<</SYS>>\n\n",
        "assistant": " {instruction} </s><s>",
        "trailing_assistant": "",
        "user": "[INST] {system}{instruction} [/INST]",
        "system_in_user": True,
        "default_system_message": "",
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "system": "{instruction} + ",
        "assistant": "{instruction}</s> ",
        "trailing_assistant": "",
        "user": "[INST] {system}{instruction} [/INST]",
        "system_in_user": True,
        "default_system_message": "",
    },
}


class ModelTypeEnum(str, Enum):
    CAUSAL = "causal"
    NEW_HEAD = "new_head"


class RouterModelConfig(BaseModel):
    model_id: str
    model_type: ModelTypeEnum
    num_outputs: int

    # output special tokens (e.g. [[1]], [[2]], etc.) for CAUSAL models
    special_tokens: List[str] = []
    flash_attention_2: bool = False
    attention_dropout: float = 0.0

    model_config = ConfigDict(protected_namespaces=())
