import argparse

from routellm.controller import Controller
from routellm.model_pair import ModelPair
from routellm.routers.routers import ROUTER_CLS

system_content = (
    "You are a helpful assistant. Respond to the questions as best as you can."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router",
        type=str,
        default="random",
        choices=ROUTER_CLS.keys(),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for the router",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is heavier, a pound of feathers or a kilogram of steel?",
    )
    args = parser.parse_args()
    print(args)

    client = Controller(
        routers=["mf"],
        config={
            "mf": {
                "checkpoint_path": "routellm/mf_gpt4_augmented",
            }
        },
        routed_pair=ModelPair(
            strong="gpt-4-1106-preview",
            weak="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
        ),
    )

    chat_completion = client.chat.completions.create(
        # Or, you can specify these in the model e.g. f"router-{args.router}-{args.threshold}"
        router=args.router,
        threshold=args.threshold,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": args.prompt},
        ],
        temperature=0.7,
    )

    response = chat_completion.choices[0].message.content
    print(f"Router used {chat_completion.model} and received: {response}")
