from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict

PROMPT_FORMAT_CONFIGS = {
    "meta-llama/Meta-Llama-3-8B": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "trailing_assistant": "",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "system_in_user": False,
        "bos": "<|begin_of_text|>",
        "default_system_message": "",
    },
}


class ModelTypeEnum(str, Enum):
    CAUSAL = "causal"


class RouterModelConfig(BaseModel):
    model_id: str
    model_type: ModelTypeEnum
    num_outputs: int

    # output special tokens (e.g. [[1]], [[2]], etc.) for CAUSAL models
    special_tokens: List[str] = []
    flash_attention_2: bool = False
    attention_dropout: float = 0.0

    model_config = ConfigDict(protected_namespaces=())
