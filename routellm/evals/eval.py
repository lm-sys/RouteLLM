import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from benchmarks import GSM8K, MMLU, MTBench

from routellm.constants import ALL_MMLU_DOMAINS, MODEL_LIST
from routellm.routers.routers import ROUTER_CLS

ANYSCALE_API_KEY = os.environ.get("ANYSCALE_API_KEY")


def generate_results(
    result_list, benchmark, benchmark_name, output, plot_optimal=False
):
    plt.figure(figsize=(10, 6))
    df_router_result = pd.DataFrame(result_list)
    for method in df_router_result["method"].unique():
        df_per_method = df_router_result[
            df_router_result["method"] == method
        ].sort_values(by=["gpt4_percentage"])

        plt.plot(
            df_per_method["gpt4_percentage"],
            df_per_method["accuracy"],
            label=f"{method}_accuracy",
            marker="x",
            linestyle="-",
        )

    if plot_optimal:
        optimal_accs = []
        optimal_range = range(0, 101, 10)
        for gpt4_percent in optimal_range:
            optimal_accs.append(benchmark.get_optimal_accuracy(gpt4_percent / 100))
        plt.plot(
            optimal_range,
            optimal_accs,
            label="Optimal Accuracy",
            marker="o",
            linestyle="-",
        )

    mixtral_accuracy = benchmark.get_model_accuracy(MODEL_LIST[0])
    print("Mixtral model accuracy:", mixtral_accuracy)

    gpt4_accuracy = benchmark.get_model_accuracy(MODEL_LIST[1])
    print("GPT-4 model accuracy:", gpt4_accuracy)

    plt.axhline(
        y=mixtral_accuracy,
        color="purple",
        linestyle="--",
        label="Mixtral Accuracy",
    )
    plt.axhline(y=gpt4_accuracy, color="orange", linestyle="--", label="GPT-4 Accuracy")
    plt.xlabel("GPT-4 Calls (%)")
    plt.ylabel("Count / Accuracy")
    plt.title(f"Model Calls and Accuracy ({benchmark_name})")
    plt.legend()

    file_name = f"{output}/{benchmark_name}.png"
    print("Saving plot to", file_name)
    plt.savefig(file_name)

    def pct_call_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["gpt4_percentage"])
        pct_calls = []

        for pct in [0.2, 0.5, 0.8]:
            pct_call = np.interp(
                pct * (gpt4_accuracy - mixtral_accuracy) + mixtral_accuracy,
                df_per_method["accuracy"],
                df_per_method["gpt4_percentage"],
            )
            pct_calls.append(f"{pct_call:.2f}%")

        return pd.Series(pct_calls)

    def auc_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["gpt4_percentage"])
        return np.trapz(
            df_per_method["accuracy"], df_per_method["gpt4_percentage"] / 100
        )

    def apgr_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["gpt4_percentage"])

        mixtral_auc = np.zeros([len(df_per_method)], dtype=float)
        mixtral_auc.fill(mixtral_accuracy)
        mixtral_auc = np.trapz(mixtral_auc, df_per_method["gpt4_percentage"] / 100)

        gpt4_auc = np.zeros([len(df_per_method)], dtype=float)
        gpt4_auc.fill(gpt4_accuracy)
        gpt4_auc = np.trapz(gpt4_auc, df_per_method["gpt4_percentage"] / 100)

        return (row["AUC"] - mixtral_auc) / (gpt4_auc - mixtral_auc)

    metrics = pd.DataFrame({"method": df_router_result["method"].unique()})
    metrics[["20% qual", "50% qual", "80% qual"]] = metrics.apply(
        pct_call_metric, axis=1
    )
    metrics["AUC"] = metrics.apply(auc_metric, axis=1)
    metrics["APGR"] = metrics.apply(apgr_metric, axis=1)
    metrics = metrics.sort_values(by=["APGR"], ascending=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Metrics:\n", metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate routers on various benchmarks."
    )
    parser.add_argument(
        "--router-list",
        nargs="+",
        type=str,
        default=["random"],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "mmlu",
            "mt-bench",
            "gsm8k",
        ],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--overwrite-cache",
        nargs="*",
        type=str,
        default=[],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()
    print(args)

    if args.benchmark == "mmlu":
        print("Running eval for full MMLU.")
        mmlu_domains = ALL_MMLU_DOMAINS
        benchmark = MMLU(mmlu_domains, args.overwrite_cache)
    elif args.benchmark == "mt-bench":
        print("Running eval for MT Bench.")
        benchmark = MTBench(args.overwrite_cache)
    elif args.benchmark == "gsm8k":
        print("Running eval for GSM8k.")
        benchmark = GSM8K(args.overwrite_cache)
    else:
        raise ValueError(f"Invalid benchmark {args.benchmark}")

    config = yaml.safe_load(open(args.config, "r"))

    all_results = []
    for router in args.router_list:
        # Ensure reproducibility on a per-router basis
        random.seed(0)
        router_config = config.get(router, {})
        router_results = benchmark.evaluate(
            ROUTER_CLS[router](router_config, MODEL_LIST), args.num_results
        )
        for threshold, score, model_counts, total in router_results:
            print(f"Evaluating router: {router} with threshold {threshold}...")
            # Pretty print stats
            header = (
                "=" * 15
                + f" {router} with threshold {threshold} on {args.benchmark} "
                + "=" * 15
            )
            print("\n" + header)
            print("Average accuracy: {:.3f}".format(score))
            print(
                f"Model counts: {', '.join([f'{k}: {v}' for k, v in model_counts.items()])}"
            )
            print(
                f"Model %: {', '.join([f'{k}: {v / total * 100:.3f}%' for k, v in model_counts.items()])}"
            )
            print("=" * len(header) + "\n")

            result = {
                "method": str(router),
                "threshold": threshold,
                "gpt4_percentage": model_counts[MODEL_LIST[1]] / total * 100,
                "accuracy": score,
            }
            all_results.append(result)

    generate_results(all_results, benchmark, args.benchmark, output=args.output)
