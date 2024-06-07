import abc
import os
from collections import Counter

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from routellm.constants import MODEL_LIST

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

tqdm.pandas()
pandarallel.initialize(progress_bar=True)


class Benchmark(abc.ABC):
    """
    Benchmark class for evaluating models.

    Internally, class should handle init and manage own cache (if needed).
    """

    @abc.abstractmethod
    def evaluate(self, router, threshold: float) -> tuple[str, dict[str, int], str]:
        """Takes in a router and threshold and returns a tuple of weighted accuracy, model counts, and number of requests."""
        pass

    @abc.abstractmethod
    def get_optimal_accuracy(self, gpt4_percent: float) -> float:
        """Takes in % gpt4 calls and returns the optimal score for the benchmark given these % of calls."""
        pass

    @abc.abstractmethod
    def get_model_accuracy(self, model: str) -> float:
        """Takes in a model name and returns the accuracy of that model on the benchmark."""
        pass


class MMLUOffline(Benchmark):
    def __init__(self, domains, overwrite_cache):
        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/mmlu/thresholds_cache.npy"

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            self.cache = {}

        all_data = pd.DataFrame()
        for domain in tqdm(domains, desc="Loading domain data"):
            all_data = pd.concat(
                [all_data, self.load_domain_data(domain)], ignore_index=True
            )
        original_length = len(all_data)

        # Generated using contamination_check.py
        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/mmlu/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        self.all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"Remaining {len(self.all_data)}/{original_length} prompts for MMLU after decontamination"
        )

    def evaluate(self, router, num_results):
        router_name = str(router)

        if router_name not in self.cache or router_name in self.overwrite_cache:
            if router.NO_PARALLEL:
                thresholds = self.all_data["prompt"].progress_apply(
                    router.calculate_threshold
                )
            else:
                thresholds = self.all_data["prompt"].parallel_map(
                    router.calculate_threshold
                )
            self.cache[router_name] = thresholds
            np.save(self.cache_path, self.cache)
        else:
            thresholds = self.cache[router_name]

        # Choose cutoffs split into 10 equally sized bins (including duplicates)
        _, cutoffs = pd.qcut(thresholds, num_results, retbins=True)
        print(f"Calculated cutoffs for {router_name}: {cutoffs}")
        self.all_data["threshold"] = thresholds

        for i, cutoff in enumerate(cutoffs):
            selection = (
                self.all_data["threshold"] >= cutoff
                if i != len(cutoffs) - 1
                else self.all_data["threshold"] > cutoff
            )
            results = np.where(
                selection,
                self.all_data[MODEL_LIST[1]],
                self.all_data[MODEL_LIST[0]],
            )
            models = np.where(
                selection,
                MODEL_LIST[1],
                MODEL_LIST[0],
            )
            model_counts = Counter(models)
            yield cutoff, sum(results) / len(results) * 100, model_counts, len(results)

    def get_optimal_accuracy(self, gpt4_percent):
        df = self.all_data
        total = len(df)

        gpt4_calls = total * gpt4_percent
        mixtral_correct = len(df[df[MODEL_LIST[0]] == True])

        df_sub = df[df[MODEL_LIST[0]] == False]
        df_sub = df_sub[df_sub[MODEL_LIST[1]] == True]

        gpt4_bonus = min(gpt4_calls, len(df_sub))
        opt_correct = mixtral_correct + gpt4_bonus
        opt_accuracy = opt_correct / total * 100

        return opt_accuracy

    def get_model_accuracy(self, model):
        df = self.all_data
        return len(df[df[model] == True]) / len(df) * 100


class MTBenchOffline(Benchmark):
    def __init__(self, overwrite_cache):
        self.judgements = pd.read_json(
            f"{CURRENT_DIR}/mt_bench/judgements.jsonl", lines=True
        )
        self.questions = pd.read_json(
            f"{CURRENT_DIR}/mt_bench/judgements.jsonl", lines=True
        )
        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/mt_bench/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()

        self.questions["turn1"] = self.questions["turns"].apply(lambda x: x[0])
        self.questions["turn2"] = self.questions["turns"].apply(lambda x: x[1])
        self.questions = self.questions[
            ~(
                self.questions["turn1"].isin(contaminated_prompts)
                | self.questions["turn2"].isin(contaminated_prompts)
            )
        ]
        print(f"{len(self.questions)} questions for MT bench after decontamination.")

        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/mt_bench/thresholds_cache.npy"

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            print("Error loading MT Bench cache, starting fresh.")
            self.cache = {}

    def evaluate(self, router, num_results):
        router_name = str(router)

        if router_name not in self.cache or router_name in self.overwrite_cache:
            if router.NO_PARALLEL:
                thresholds = self.questions["turns"].progress_apply(
                    # Only use first turn for routing
                    lambda turn: router.calculate_threshold(turn[0])
                )
            else:
                thresholds = self.questions["turns"].parallel_apply(
                    lambda turn: router.calculate_threshold(turn[0])
                )
            self.cache[router_name] = thresholds
            np.save(self.cache_path, self.cache)
        else:
            thresholds = self.cache[router_name]

        _, cutoffs = pd.qcut(thresholds, num_results, retbins=True)
        print(f"Calculated cutoffs for {router_name}: {cutoffs}")
        questions = self.questions[["question_id", "turns"]]
        questions["threshold"] = thresholds

        for i, cutoff in enumerate(cutoffs):
            questions["routed_model"] = np.where(
                (
                    questions["threshold"] >= cutoff
                    if i != len(cutoffs) - 1
                    else questions["threshold"] > cutoff
                ),
                MODEL_LIST[1],
                MODEL_LIST[0],
            )

            results = questions.merge(
                self.judgements,
                left_on=["question_id", "routed_model"],
                right_on=["question_id", "model"],
                how="left",
            )[["question_id", "model", "score"]]

            score = results["score"].mean()

            model_counts = results["model"].value_counts().to_dict()
            if MODEL_LIST[0] not in model_counts:
                model_counts[MODEL_LIST[0]] = 0
            if MODEL_LIST[1] not in model_counts:
                model_counts[MODEL_LIST[1]] = 0

            total = len(results)

            assert total == sum(model_counts.values()) == len(self.questions) * 2

            yield cutoff, score, model_counts, total

    def get_model_accuracy(self, model):
        questions = self.questions[["question_id"]]
        questions["routed_model"] = model

        results = questions.merge(
            self.judgements,
            left_on=["question_id", "routed_model"],
            right_on=["question_id", "model"],
            how="left",
        )[["question_id", "model", "score"]]

        return results["score"].mean()

    def get_optimal_accuracy(self, gpt4_percent):
        max_gpt4_calls = int(len(self.questions) * gpt4_percent)

        gpt4_judgements = (
            self.judgements[self.judgements["model"] == "gpt-4-1106-preview"][
                ["question_id", "model", "score"]
            ]
            .groupby(by=["model", "question_id"], as_index=False)
            .mean()
        )

        mixtral_judgements = (
            self.judgements[self.judgements["model"] == MODEL_LIST[0]][
                [
                    "question_id",
                    "model",
                    "score",
                ]
            ]
            .groupby(by=["model", "question_id"], as_index=False)
            .mean()
        )

        combined_judgements = gpt4_judgements.merge(
            mixtral_judgements,
            on=["question_id"],
            how="left",
            suffixes=("_gpt4", "_mixtral"),
        )
        combined_judgements["diff"] = (
            combined_judgements["score_gpt4"] - combined_judgements["score_mixtral"]
        )
        combined_judgements = combined_judgements.sort_values(
            by=["diff"], ascending=False
        ).reset_index(drop=True)

        if len(combined_judgements[combined_judgements["diff"] > 0]) > max_gpt4_calls:
            combined_judgements.loc[:max_gpt4_calls, "score_optimal"] = (
                combined_judgements.loc[:max_gpt4_calls, "score_gpt4"]
            )
            combined_judgements.loc[max_gpt4_calls:, "score_optimal"] = (
                combined_judgements.loc[max_gpt4_calls:, "score_mixtral"]
            )
        else:
            combined_judgements["score_optimal"] = combined_judgements[
                "score_gpt4"
            ].where(
                combined_judgements["diff"] > 0, combined_judgements["score_mixtral"]
            )

        assert (
            len(gpt4_judgements) == len(mixtral_judgements) == len(combined_judgements)
        )

        return combined_judgements["score_optimal"].mean()


class GSM8KOffline(Benchmark):
    def __init__(self, overwrite_cache):
        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/gsm8k/thresholds_cache.npy"

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            self.cache = {}

        all_data = pd.read_csv(f"{CURRENT_DIR}/gsm8k/gsm8k_responses.csv")
        original_len = len(all_data)

        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/gsm8k/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        self.all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"{len(self.all_data)}/{original_len} questions for GSM8K after decontamination."
        )

    def evaluate(self, router, num_results):
        router_name = str(router)

        if router_name not in self.cache or router_name in self.overwrite_cache:
            if router.NO_PARALLEL:
                thresholds = self.all_data["prompt"].progress_apply(
                    router.calculate_threshold
                )
            else:
                thresholds = self.all_data["prompt"].parallel_map(
                    router.calculate_threshold
                )
            self.cache[router_name] = thresholds
            np.save(self.cache_path, self.cache)
        else:
            thresholds = self.cache[router_name]

        # Choose cutoffs split into 10 equally sized bins (including duplicates)
        _, cutoffs = pd.qcut(thresholds, num_results, retbins=True)
        print(f"Calculated cutoffs for {router_name}: {cutoffs}")
        self.all_data["threshold"] = thresholds

        for i, cutoff in enumerate(cutoffs):
            selection = (
                self.all_data["threshold"] >= cutoff
                if i != len(cutoffs) - 1
                else self.all_data["threshold"] > cutoff
            )
            results = np.where(
                selection,
                self.all_data[MODEL_LIST[1]],
                self.all_data[MODEL_LIST[0]],
            )
            models = np.where(selection, MODEL_LIST[1], MODEL_LIST[0])
            model_counts = Counter(models)
            yield cutoff, sum(results) / len(results) * 100, model_counts, len(results)

    def get_model_accuracy(self, model):
        df = self.all_data
        return len(df[df[model] == True]) / len(df) * 100

    def get_optimal_accuracy(self, gpt4_percent):
        df = self.all_data
        total = len(df)

        gpt4_calls = total * gpt4_percent
        mixtral_correct = len(df[df[MODEL_LIST[0]] == True])

        df_sub = df[df[MODEL_LIST[0]] == False]
        df_sub = df_sub[df_sub[MODEL_LIST[1]] == True]

        gpt4_bonus = min(gpt4_calls, len(df_sub))
        opt_correct = mixtral_correct + gpt4_bonus
        opt_accuracy = opt_correct / total * 100

        return opt_accuracy
