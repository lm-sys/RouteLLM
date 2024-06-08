import abc
import functools
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import routellm.routers.similarity_weighted.utils as sw_utils
from routellm.constants import MODEL_LIST
from routellm.routers.causal_llm.llm_utils import (
    load_model_config,
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier


def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models.
    # This is normally the winrate of the GPT-4 model, but it may have a different meaning depending on the router.
    # If this value is >= the user defined cutoff, the router will choose GPT-4, otherwise, it will choose Mixtral.
    @abc.abstractmethod
    def calculate_threshold(self, prompt):
        pass

    def route(self, prompt, threshold):
        if self.calculate_threshold(prompt) >= threshold:
            return MODEL_LIST[1]
        else:
            return MODEL_LIST[0]

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


@no_parallel
class CausalLLMRouter(Router):
    def __init__(
        self,
        config,
        model_list,
        score_threshold=2,
    ):
        self.model_list = model_list
        model_config = load_model_config(config["model_config_path"])
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=config["model_checkpoint_path"],
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        with open(config["system_message"], "r") as pr:
            system_message = pr.read()
        with open(config["classifier_message"], "r") as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_threshold(self, prompt):
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        return 1 - output["binary_prob"]


@no_parallel
class BERTRouter(Router):
    def __init__(
        self,
        config,
        model_list,
        num_labels=3,
    ):
        self.model_list = model_list
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config["model_path"], num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    def calculate_threshold(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Compute prob of label 1 and 2 (tie, tier 2 wins)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob


class SWRankingRouter(Router):
    def __init__(self, config, model_list, num_tiers=10):
        self.model_list = model_list
        try:
            if self.ARENA_DF_PATH.endswith(".json"):
                self.arena_df = pd.read_json(config["arena_df_path"])
            else:
                self.arena_df = pd.read_json(config["arena_df_path"], lines=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Expected data file not found at path: {config['arena_df_path']}"
            )
        try:
            self.arena_conv_embedding = np.load(config["arena_embedding_path"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Expected data file not found at path: {config['arena_embedding_path']}"
            )
        self.embedding_model = "text-embedding-3-small"

        model_ratings = sw_utils.compute_elo_mle_with_tie(self.arena_df)
        self.model2tier = sw_utils.compute_tiers(model_ratings, num_tiers=num_tiers)

        self.arena_df["model_a"] = self.arena_df["model_a"].apply(
            lambda x: self.model2tier[x]
        )
        self.arena_df["model_b"] = self.arena_df["model_b"].apply(
            lambda x: self.model2tier[x]
        )

        res = sw_utils.compute_elo_mle_with_tie(self.arena_df)
        self.gap = (
            res[self.model2tier["gpt-4-1106-preview"]]
            - res[self.model2tier["mixtral-8x7b-instruct-v0.1"]]
        )

    def get_weightings(self, similarities):
        max_sim = np.max(similarities)
        return 10 * 10 ** (similarities / max_sim)

    def calculate_threshold(
        self,
        prompt,
    ):
        prompt_emb = (
            (
                sw_utils.OPENAI_CLIENT.embeddings.create(
                    input=[prompt], model=self.embedding_model
                )
            )
            .data[0]
            .embedding
        )
        similarities = np.dot(self.arena_conv_embedding, prompt_emb) / (
            np.linalg.norm(self.arena_conv_embedding, axis=1)
            * np.linalg.norm(prompt_emb)
        )

        weightings = self.get_weightings(similarities)
        res = sw_utils.compute_elo_mle_with_tie(self.arena_df, sample_weight=weightings)

        mixtral_score, gpt4_score = (
            res[self.model2tier["mixtral-8x7b-instruct-v0.1"]],
            res[self.model2tier["gpt-4-1106-preview"]],
        )
        mixtral_winrate = 1 / (1 + 10 ** ((gpt4_score - mixtral_score) / 400))
        gpt4_winrate = 1 - mixtral_winrate

        # If the expected gpt4 winrate is greater than the threshold, use gpt4
        return gpt4_winrate


@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(self, config, model_list):
        self.model_list = model_list
        self.model = sw_utils.MFModel(config["hidden_size"])
        self.model.load(config["checkpoint_path"])
        self.model = self.model.eval().to("cuda")
        assert model_list == ["mixtral-8x7b-instruct-v0.1", "gpt-4-1106-preview"]
        self.gpt4_id = 24
        self.mixtral_id = 36

    def calculate_threshold(self, prompt):
        winrate = self.model.pred_win_rate(self.gpt4_id, self.mixtral_id, prompt)
        return winrate


# Parallelism makes the randomness non deterministic
@no_parallel
class RandomRouter(Router):
    def __init__(self, model_list):
        self.model_list = model_list

    def calculate_threshold(
        self,
        prompt,
    ):
        del prompt
        return random.uniform(0, 1)


ROUTER_CLS = {
    "random": RandomRouter,
    "matrix_factorization": MatrixFactorizationRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
    "sw_ranking": SWRankingRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
