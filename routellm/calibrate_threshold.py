import argparse
import json

import yaml
from datasets import Dataset, load_dataset
from pandarallel import pandarallel
from tqdm import tqdm

from routellm.controller import Controller
from routellm.routers.routers import ROUTER_CLS

tqdm.pandas()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--battles_dataset", type=str, default="lmsys/lmsys-arena-human-preference-55k"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--routers",
        nargs="+",
        type=str,
        default=["random"],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument("--strong-model-pct", type=float)
    parser.add_argument(
        "--task", type=str, choices=["generate", "calibrate"], default="calibrate"
    )
    args = parser.parse_args()

    if args.task == "generate":
        pandarallel.initialize(progress_bar=True)
        battles_df = load_dataset(args.battles_dataset, split="train").to_pandas()
        controller = Controller(
            routers=args.routers,
            config=yaml.safe_load(open(args.config, "r")) if args.config else None,
            # This is not needed since we only calculate the win rate
            routed_pair=None,
            progress_bar=True,
        )

        for router in args.routers:
            win_rates = controller.batch_calculate_win_rate(
                battles_df["prompt"].apply(lambda x: json.loads(x)[0]), router
            )
            battles_df[str(router)] = win_rates
            Dataset.from_pandas(battles_df).push_to_hub(
                "routellm/lmsys-arena-human-preference-55k-thresholds"
            )
    elif args.task == "calibrate":
        thresholds_df = load_dataset(
            "routellm/lmsys-arena-human-preference-55k-thresholds", split="train"
        ).to_pandas()
        for router in args.routers:
            threshold = thresholds_df[router].quantile(q=1 - args.strong_model_pct)
            print(
                f"For {args.strong_model_pct * 100}% strong model calls for {router}: threshold = {round(threshold, 5)}"
            )
