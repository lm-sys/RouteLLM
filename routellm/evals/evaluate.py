import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from pandarallel import pandarallel

from routellm.evals.benchmarks import GSM8K, MMLU, MTBench
from routellm.evals.mmlu.domains import ALL_MMLU_DOMAINS
from routellm.model_pair import ROUTED_PAIR
from routellm.routers.routers import ROUTER_CLS

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_results(
    result_list, benchmark, benchmark_name, output, plot_optimal=False
):
    plt.figure(figsize=(6, 5))
    df_router_result = pd.DataFrame(result_list)
    for method in df_router_result["method"].unique():
        df_per_method = df_router_result[
            df_router_result["method"] == method
        ].sort_values(by=["strong_percentage"])

        plt.plot(
            df_per_method["strong_percentage"],
            df_per_method["accuracy"],
            label=f"{method}",
            marker=".",
            linestyle="-",
        )

    weak_accuracy = benchmark.get_model_accuracy(ROUTED_PAIR.weak)
    print(ROUTED_PAIR.weak, weak_accuracy)

    strong_accuracy = benchmark.get_model_accuracy(ROUTED_PAIR.strong)
    print(ROUTED_PAIR.strong, strong_accuracy)

    plt.axhline(
        y=weak_accuracy,
        color="grey",
        linestyle="--",
        label=ROUTED_PAIR.weak,
    )
    plt.axhline(
        y=strong_accuracy,
        color="red",
        linestyle="--",
        label=ROUTED_PAIR.strong,
    )

    if plot_optimal:
        optimal_accs = []
        optimal_range = range(0, 101, 10)
        for strong_percent in optimal_range:
            optimal_accs.append(benchmark.get_optimal_accuracy(strong_percent / 100))
        plt.plot(
            optimal_range,
            optimal_accs,
            label="Optimal",
            marker="x",
            linestyle="-",
        )

    plt.xlabel("Strong Model Calls (%)")
    plt.ylabel("Performance")
    plt.title(f"Router Performance ({benchmark_name})")
    plt.legend()

    file_name = f"{output}/{benchmark_name}.png"
    print("Saving plot to", file_name)
    plt.savefig(file_name, bbox_inches="tight")

    def pct_call_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        pct_calls = []

        for pct in [0.2, 0.5, 0.8]:
            pct_call = np.interp(
                pct * (strong_accuracy - weak_accuracy) + weak_accuracy,
                df_per_method["accuracy"],
                df_per_method["strong_percentage"],
            )
            pct_calls.append(f"{pct_call:.2f}%")

        return pd.Series(pct_calls)

    def auc_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        return np.trapz(
            df_per_method["accuracy"], df_per_method["strong_percentage"] / 100
        )

    def apgr_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])

        weak_auc = np.zeros([len(df_per_method)], dtype=float)
        weak_auc.fill(weak_accuracy)
        weak_auc = np.trapz(weak_auc, df_per_method["strong_percentage"] / 100)

        strong_auc = np.zeros([len(df_per_method)], dtype=float)
        strong_auc.fill(strong_accuracy)
        strong_auc = np.trapz(strong_auc, df_per_method["strong_percentage"] / 100)

        return (row["AUC"] - weak_auc) / (strong_auc - weak_auc)

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
        "--routers",
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
    parser.add_argument(
        "--parallel",
        type=int,
        default=psutil.cpu_count(logical=False),
        help="Number of cores to use, all by default.",
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--num-results", type=int, default=10)

    args = parser.parse_args()
    print(args)

    pandarallel.initialize(progress_bar=True, nb_workers=args.parallel)

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
    for router in args.routers:
        # Ensure reproducibility on a per-router basis
        random.seed(0)
        router_config = config.get(router, {})
        router_results = benchmark.evaluate(
            ROUTER_CLS[router](**router_config), args.num_results
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
                "strong_percentage": model_counts[ROUTED_PAIR.strong] / total * 100,
                "accuracy": score,
            }
            all_results.append(result)

    generate_results(all_results, benchmark, args.benchmark, output=args.output)
