import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from routellm.routers.causal_llm.configs import (
    PROMPT_FORMAT_CONFIGS,
    ModelTypeEnum,
    RouterModelConfig,
)
from routellm.routers.causal_llm.prompt_format import PromptFormat


def load_model_config(yaml_path: str):
    with open(yaml_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    return RouterModelConfig(**yaml_data)


def load_prompt_format(model_id):
    prompt_format_dict = PROMPT_FORMAT_CONFIGS[model_id]
    return PromptFormat(**prompt_format_dict, is_generation=True)


def get_model(config: RouterModelConfig, model_ckpt: str, pad_token_id: int = 2):
    if config.model_type == ModelTypeEnum.CAUSAL:
        return AutoModelForCausalLM.from_pretrained(
            model_ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            use_flash_attention_2=config.flash_attention_2,
            attention_dropout=config.attention_dropout,
        )
    elif config.model_type == ModelTypeEnum.NEW_HEAD:
        return AutoModelForSequenceClassification.from_pretrained(
            model_ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            use_flash_attention_2=config.flash_attention_2,
            num_labels=config.num_outputs,
            pad_token_id=pad_token_id,
        )
    else:
        raise NotImplementedError(
            f"ModelType {config.model_type} is not implemented yet!"
        )


def get_tokenizer(model_id, special_tokens=None, truncation_side="left"):
    # Context for legacy=True: https://github.com/huggingface/transformers/issues/25176
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    tokenizer.truncation_side = truncation_side
    return tokenizer


def to_openai_api_messages(system_message, classifier_message, messages):
    """Convert the conversation to OpenAI chat completion format."""

    ret = [{"role": "system", "content": system_message}]
    for i, turn in enumerate(messages):
        if i % 2 == 0:
            ret.append(
                {"role": "user", "content": classifier_message.format(question=turn)}
            )
        else:
            ret.append({"role": "assistant", "content": turn})
    return ret
