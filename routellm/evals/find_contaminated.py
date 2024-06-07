import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from openai import OpenAI

from routellm.constants import ALL_MMLU_DOMAINS

THRESHOLD = 0.95
client = OpenAI()


def check_data_contamination_similarity(train_embeddings, eval_prompts):
    eval_embeddings = []

    batch_size = 2000
    for eval_idx in tqdm.tqdm(range(0, len(eval_prompts), batch_size)):
        prompts = eval_prompts[eval_idx : eval_idx + batch_size]
        responses = client.embeddings.create(
            input=prompts, model="text-embedding-3-small"
        ).data
        eval_embeddings.extend([data.embedding for data in responses])

    eval_embeddings = torch.tensor(eval_embeddings)
    eval_embeddings = eval_embeddings.numpy()

    similarities = np.dot(eval_embeddings, train_embeddings.T) / (
        np.linalg.norm(eval_embeddings, axis=1)[:, np.newaxis]
        * np.linalg.norm(train_embeddings, axis=1)
    )

    contaminated_prompts = []
    for eval_idx in range(len(eval_prompts)):
        max_similarity_idx = np.argmax(similarities[eval_idx])
        max_similarity = similarities[eval_idx][max_similarity_idx]
        if max_similarity >= THRESHOLD:
            contaminated_prompts.append((eval_idx, max_similarity_idx))

    return contaminated_prompts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["mt-bench", "mmlu", "gsm8k"],
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
    )
    parser.add_argument(
        "--battles-path",
        type=str,
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.benchmark == "mt-bench":
        questions = pd.read_json(f"{current_dir}/mt_bench/question.jsonl", lines=True)
        eval_prompts = []
        eval_prompts.extend(questions["turns"].apply(lambda x: x[0]))
        eval_prompts.extend(questions["turns"].apply(lambda x: x[1]))
    elif args.benchmark == "mmlu":
        eval_prompts = []
        for domain in ALL_MMLU_DOMAINS:
            eval_prompts.extend(
                pd.read_csv(f"{current_dir}/mmlu/responses/mmlu_{domain}.csv")[
                    "prompt"
                ].tolist()
            )
    elif args.benchmark == "gsm8k":
        eval_prompts = pd.read_csv(f"{current_dir}/gsm8k/gsm8k_responses.csv")[
            "prompt"
        ].tolist()

    print(
        f"Checking data contamination for {len(eval_prompts)} prompts from {args.benchmark}"
    )

    train_prompts = (
        pd.read_json(args.battles_path)["conversation_a"]
        .apply(lambda x: x[0]["content"])
        .tolist()
    )
    train_embeddings = np.load(args.embeddings_path)

    contaminated_prompts = check_data_contamination_similarity(
        train_embeddings, eval_prompts
    )

    battles = pd.read_json(args.battles_path)

    contamination_df = []
    for eval_idx, train_idx in contaminated_prompts:
        contamination_df.append(
            {
                "train_prompt": train_prompts[train_idx],
                "eval_prompt": eval_prompts[eval_idx],
                "train_idx": train_idx,
                "eval_idx": eval_idx,
            }
        )
    contamination_df = pd.DataFrame(contamination_df)
    print(contamination_df)
    contamination_df.to_json(args.output, lines=True, orient="records")
