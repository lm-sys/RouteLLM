import argparse
import json
import os

import openai
import torch
import tqdm
from datasets import Dataset, load_dataset

from routellm.routers.similarity_weighted.utils import preprocess_battles


def get_embeddings(battles_df):
    battles_df = preprocess_battles(battles_df)
    print(f"Battles after preprocessing: {battles_df.shape[0]}")
    battles_df["first_turn"] = battles_df["prompt"].apply(
        lambda s: json.loads(s)[0].strip()
    )

    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1"
    )

    batch_size = 2000
    embeddings = []
    user_prompts = battles_df["first_turn"].tolist()

    for i in tqdm.tqdm(range(0, len(user_prompts), batch_size)):
        battles = user_prompts[i : i + batch_size]
        responses = client.embeddings.create(
            input=battles, model="text-embedding-3-small"
        ).data
        embeddings.extend([data.embedding for data in responses])
    embeddings = torch.tensor(embeddings)
    embeddings = embeddings.numpy()

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Arena battles after preprocessing"
    )
    parser.add_argument("--battles-dataset", type=str)
    parser.add_argument("--output-dataset", type=str)

    args = parser.parse_args()
    battles_df = load_dataset(args.battles_dataset, split="train").to_pandas()

    embeddings = get_embeddings(battles_df)

    embeddings_dataset = Dataset.from_dict({"embeddings": embeddings})
    embeddings_dataset.push_to_hub(args.output_dataset)
