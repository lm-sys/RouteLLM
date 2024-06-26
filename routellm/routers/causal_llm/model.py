import re
import time
from typing import List

import numpy as np
import torch

from routellm.routers.causal_llm.configs import ModelTypeEnum, RouterModelConfig
from routellm.routers.causal_llm.llm_utils import get_model, get_tokenizer
from routellm.routers.causal_llm.prompt_format import PromptFormat


class CausalLLMClassifier:
    def __init__(
        self,
        config: RouterModelConfig,
        ckpt_local_path: str,
        prompt_format: PromptFormat,
        score_threshold: int,
        prompt_field: str = "messages",
        use_last_turn: bool = False,
        additional_fields: List[str] = list(["label", "pidx"]),
        max_new_tokens: int = 6,
    ):
        """
        This model is trained to predict a score [1, 5] for a given user query.
        Higher score means higher chance the simple model will perform well (think score of it's response)

        score_threshold: defines probability of routing as follows
            P(routing=1) := sum(prob(score)) for all score (in [1,5]) s.t. score >= score_threshold
        """
        # Initialize the batch generator
        print(f"Loading model checkpoint from {ckpt_local_path} ...")
        s = time.time()

        assert config.model_type == ModelTypeEnum.CAUSAL

        # assert that config has 5 special tokens with the format [[rating]]
        assert len(config.special_tokens) == config.num_outputs
        for i in range(1, config.num_outputs + 1):
            assert f"[[{i}]]" in config.special_tokens

        model = get_model(config=config, model_ckpt=ckpt_local_path)
        self.model = model.to("cuda").eval()

        self.prompt_format = prompt_format
        self.use_last_turn = use_last_turn
        self.prompt_field = prompt_field
        self.additinal_fields = additional_fields

        self.tokenizer = get_tokenizer(
            config.model_id,
            special_tokens=config.special_tokens,
            truncation_side="left",
            padding_side="left",
        )
        self.orig_vocab_size = len(self.tokenizer) - config.num_outputs
        self.max_new_tokens = max_new_tokens
        self.score_threshold = score_threshold
        assert (
            self.score_threshold == config.num_outputs - 1
        ), "this is the default value for now."
        print(f"Done loading model in {time.time() - s} seconds.")

    def preprocess(self, row):
        """prepare each prompt before feeding it to the model"""
        # add additional fields to the final output (e.g. for later evaluation)
        data_row = {}
        for field in self.additinal_fields:
            data_row[field] = row[field]
        # select turns from the prompt field
        openai_messages = (
            row[self.prompt_field]
            if self.use_last_turn
            else row[self.prompt_field][:-1]
        )
        # convert openai messages formot to model's prompt format
        text = self.prompt_format.generate_prompt(openai_messages)
        # tokenize and encode
        data_row["input_ids"] = np.array(self.tokenizer.encode(text))

        return data_row

    def __call__(self, row):
        row = self.preprocess(row)
        input_ids = torch.as_tensor(row["input_ids"]).to("cuda").reshape(1, -1)
        with torch.no_grad():
            output_new = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # see https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L101
        row["output_ids"] = output_new.sequences.squeeze()[input_ids.shape[1] :].cpu()
        assert len(row["output_ids"]) == len(row["output_ids"])

        # find the first token within the special tokens range. This is our score prediction.
        label_token_idx = next(
            (i for i, x in enumerate(row["output_ids"]) if x >= self.orig_vocab_size),
            None,
        )
        if label_token_idx is None:
            return None

        # extract logits of predicted labels from scores of each output token
        # (check hf github modeling llama)
        score_logits = np.array(
            output_new.scores[label_token_idx][0].to("cpu")[self.orig_vocab_size :]
        )
        row["score_logits"] = score_logits
        binary_prob, softmax_scores = self.compute_routing_prob(score_logits)
        row["softmax_scores"] = softmax_scores
        row["binary_prob"] = binary_prob

        row = self.postprocess(row)
        return row

    def compute_routing_prob(self, score_logits):
        """convert score_logits to binary probability of routing the query"""
        exp_scores = np.exp(score_logits - np.max(score_logits))
        softmax_scores = exp_scores / np.sum(exp_scores)
        binary_prob = np.sum(softmax_scores[self.score_threshold - 1 :])
        return binary_prob, softmax_scores

    def postprocess(self, row):
        """process model's predictions"""

        output_str = self.tokenizer.decode(row["output_ids"])
        row["output_tokens"] = self.tokenizer.convert_ids_to_tokens(row["output_ids"])
        row["output_str"] = output_str
        row["score_pred"] = self.parse_score(output_str)
        # to debug, check both logits and generation prediction match
        logits_pred = np.argmax(row["score_logits"]) + 1
        assert logits_pred == row["score_pred"]

        # clean up input data
        del row["input_ids"]

        return row

    def parse_score(self, text):
        """Extract int score from the predicted string with the format [[5]]"""
        match = re.search(r"\[\[([\d\.]+)\]\]", text)
        if match:
            return int(float(match.group(1)))
        else:
            raise Exception(f"Bad score format {text}.")
