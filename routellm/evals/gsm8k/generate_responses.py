import ast
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from openai import OpenAI

from routellm.constants import MODEL_LIST

"""
The core code is based heavily on the original SGLang implementation.
https://github.com/sgl-project/sglang/blob/main/benchmark/gsm8k/bench_sglang.py
"""

INVALID = -9999999


def select_sglang_backend(args):
    if args.backend.startswith("gpt") or args.backend.startswith("router-"):
        backend = OpenAI(args.backend, base_url=f"{args.host}:{args.port}/v1")
    else:
        raise ValueError(f"Invalid backend: {args.backend}")
    return backend


def read_jsonl(filename: str):
    """Read a JSONL file."""
    rets = []
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            rets.append(json.loads(line))
    return rets


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lines = read_jsonl(f"{current_dir}/test.jsonl")
    train = read_jsonl(f"{current_dir}/train.jsonl")

    # Construct prompts
    k = 8
    few_shot_examples = get_few_shot_examples(train, k)

    questions = []
    labels = []
    for i in range(len(lines)):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += sgl.user(few_shot_examples + question)
        s += sgl.assistant(sgl.gen("answer", max_tokens=1024, stop=["Question"]))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )

    preds = []
    responses = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))
        responses.append(states[i]["answer"])

    # Compute accuracy
    return np.array(preds) == np.array(labels), responses


evaluate_args_base = {
    "parallel": 64,
    "host": "http://localhost",
    "port": "6060",
}
mixtral_cors, mixtral_responses = main(
    SimpleNamespace(**evaluate_args_base, backend="router-random-1.0"),
)
gpt4_cors, gpt4_responses = main(
    SimpleNamespace(**evaluate_args_base, backend="router-random-0.0"),
)
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts = pd.read_json(f"{current_dir}/test.jsonl", lines=True)["question"].tolist()

assert len(mixtral_cors) == len(gpt4_cors)

result_df = pd.DataFrame(
    zip(prompts, mixtral_cors, gpt4_cors, mixtral_responses, gpt4_responses),
    columns=[
        "prompt",
        MODEL_LIST[0],
        MODEL_LIST[1],
        f"{MODEL_LIST[0]}_response",
        f"{MODEL_LIST[1]}_response",
    ],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
result_df.to_csv(
    f"{current_dir}/gsm8k_responses.csv",
    index=False,
)
