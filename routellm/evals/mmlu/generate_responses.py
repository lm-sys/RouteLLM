import argparse
import os
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tiktoken
import tqdm
from openai import OpenAI

from routellm.evals.mmlu.domains import ALL_MMLU_DOMAINS
from routellm.model_pair import ModelPair

ROUTED_PAIR = ModelPair(
    strong="gpt-4-1106-preview", weak="mistralai/Mixtral-8x7B-Instruct-v0.1"
)

choices = ["A", "B", "C", "D"]
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


"""
The core code is adapted from the original SGLang implementation.
https://github.com/sgl-project/sglang/blob/main/benchmark/mmlu/bench_sglang.py
"""


def select_sglang_backend(args):
    if args.backend.startswith("gpt") or args.backend.startswith("router-"):
        backend = OpenAI(args.backend, base_url=f"{args.host}:{args.port}/v1")
    else:
        raise ValueError(f"Invalid backend: {args.backend}")
    return backend


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate(args, subject, dev_df, test_df):
    prompts = []
    labels = []

    k = args.ntrain
    few_shot_examples = gen_prompt(dev_df, subject, k)
    while len(tokenizer.encode(few_shot_examples)) > 1536:
        k -= 1
        few_shot_examples = gen_prompt(dev_df, subject, k)

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        prompts.append(prompt_end)

        label = test_df.iloc[i, test_df.shape[1] - 1]
        labels.append(label)

    arguments = [{"question": p} for p in prompts]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    if args.backend.startswith("gpt-") or args.backend.startswith("router-"):

        @sgl.function
        def few_shot_mmlu(s, examples, question):
            s += sgl.user(examples + question)
            s += sgl.assistant(sgl.gen("answer"))

    else:

        @sgl.function
        def few_shot_mmlu(s, examples, question):
            s += examples + question + sgl.gen("answer")

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Select backend
    backend = select_sglang_backend(args)

    tic = time.time()
    states = few_shot_mmlu.bind(examples=few_shot_examples).run_batch(
        arguments,
        temperature=0,
        max_new_tokens=1,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )
    preds = [
        s["answer"].strip()[0] if len(s["answer"].strip()) > 0 else "" for s in states
    ]
    models = [s["model"] for s in states]
    latency = time.time() - tic

    cors = [pred == label for pred, label in zip(preds, labels)]
    acc = np.mean(cors)
    cors = np.array(cors)
    model_counts = Counter(models)

    print(
        "Average accuracy {:.3f}, latency {:.2f}, #q: {} - {}, routing: {}".format(
            acc,
            latency,
            len(prompts),
            subject,
            ", ".join([f"{k}: {v}" for k, v in model_counts.items()]),
        )
    )

    return cors, acc, latency, model_counts, prompts


def generate_domain_data(args, domain):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_key = f"{current_dir}/responses/mmlu_{domain}.csv"
    if os.path.exists(cache_key):
        return pd.read_csv(cache_key)

    dev_df = pd.read_csv(f"{current_dir}/data/dev/{domain}_dev.csv", header=None)[
        : args.ntrain
    ]
    test_df = pd.read_csv(f"{current_dir}/data/test/{domain}_test.csv", header=None)

    # Dummy router just to get the results
    weak_cors, _, _, _, prompts = evaluate(
        SimpleNamespace(**vars(args), backend="router-random-1.0"),
        domain,
        dev_df,
        test_df,
    )

    strong_cors, _, _, _, _ = evaluate(
        SimpleNamespace(**vars(args), backend="router-random-0.0"),
        domain,
        dev_df,
        test_df,
    )

    assert len(weak_cors) == len(strong_cors)

    result_df = pd.DataFrame(
        zip(prompts, weak_cors, strong_cors),
        columns=["prompt", ROUTED_PAIR.weak, ROUTED_PAIR.strong],
    )

    result_df.to_csv(cache_key, index=False)
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--port", type=str, default="6060")
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    args = parser.parse_args()

    for domain in tqdm.tqdm(ALL_MMLU_DOMAINS, desc="Loading MMLU data"):
        generate_domain_data(args, domain)
