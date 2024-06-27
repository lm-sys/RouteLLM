import abc
import functools
import random

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from routellm.model_pair import ROUTED_PAIR
from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier
from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel
from routellm.routers.similarity_weighted.utils import (
    OPENAI_CLIENT,
    compute_elo_mle_with_tie,
    compute_tiers,
    preprocess_battles,
)


def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return ROUTED_PAIR.strong
        else:
            return ROUTED_PAIR.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


@no_parallel
class CausalLLMRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        system_message="routellm/routers/causal_llm/system_ft_v5.txt",
        classifier_message="routellm/routers/causal_llm/classifier_ft_v5.txt",
        score_threshold=4,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
        model_type="causal",
        model_id="meta-llama/Meta-Llama-3-8B",
        flash_attention_2=False,
    ):
        model_config = RouterModelConfig(
            model_id=model_id,
            model_type=model_type,
            flash_attention_2=flash_attention_2,
            special_tokens=special_tokens,
            num_outputs=num_outputs,
        )
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=checkpoint_path,
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        with open(system_message, "r") as pr:
            system_message = pr.read()
        with open(classifier_message, "r") as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_strong_win_rate(self, prompt):
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        if output is None:
            # Route to strong model if output is invalid
            return 1
        else:
            return 1 - output["binary_prob"]


@no_parallel
class BERTRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        num_labels=3,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def calculate_strong_win_rate(self, prompt):
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
    def __init__(
        self,
        arena_battle_datasets,
        arena_embedding_datasets,
        strong_model,
        weak_model,
        num_tiers=10,
    ):
        self.strong_model = strong_model
        self.weak_model = weak_model

        self.arena_df = concatenate_datasets(
            [load_dataset(dataset, split="train") for dataset in arena_battle_datasets]
        ).to_pandas()
        self.arena_df = preprocess_battles(self.arena_df)

        embeddings = [
            np.array(load_dataset(dataset, split="train").to_dict()["embeddings"])
            for dataset in arena_embedding_datasets
        ]
        self.arena_conv_embedding = np.concatenate(embeddings)
        self.embedding_model = "text-embedding-3-small"

        assert len(self.arena_df) == len(
            self.arena_conv_embedding
        ), "Number of battle embeddings is mismatched to data"

        model_ratings = compute_elo_mle_with_tie(self.arena_df)
        self.model2tier = compute_tiers(model_ratings, num_tiers=num_tiers)

        self.arena_df["model_a"] = self.arena_df["model_a"].apply(
            lambda x: self.model2tier[x]
        )
        self.arena_df["model_b"] = self.arena_df["model_b"].apply(
            lambda x: self.model2tier[x]
        )

    def get_weightings(self, similarities):
        max_sim = np.max(similarities)
        return 10 * 10 ** (similarities / max_sim)

    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        prompt_emb = (
            (
                OPENAI_CLIENT.embeddings.create(
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
        res = compute_elo_mle_with_tie(self.arena_df, sample_weight=weightings)

        weak_score, strong_score = (
            res[self.model2tier[self.weak_model]],
            res[self.model2tier[self.strong_model]],
        )
        weak_winrate = 1 / (1 + 10 ** ((strong_score - weak_score) / 400))
        strong_winrate = 1 - weak_winrate

        # If the expected strong winrate is greater than the threshold, use strong
        return strong_winrate


@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        strong_model,
        weak_model,
        hidden_size=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MFModel.from_pretrained(
            checkpoint_path,
            dim=hidden_size,
            num_models=num_models,
            text_dim=text_dim,
            num_classes=num_classes,
            use_proj=use_proj,
        )
        self.model = self.model.eval().to(device)
        self.strong_model_id = MODEL_IDS[strong_model]
        self.weak_model_id = MODEL_IDS[weak_model]

    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate


@no_parallel
class NDRouter(Router):
    def __init__(
        self,
        model_path="notdiamond/notdiamond-0001",
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def calculate_strong_win_rate(self, prompt):
        inputs = self.tokenizer(
            prompt, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        binary_prob = np.sum(softmax_scores[0])
        return 1 - binary_prob


# Parallelism makes the randomness non deterministic
@no_parallel
class RandomRouter(Router):
    def calculate_strong_win_rate(
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
    "nd": NDRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
