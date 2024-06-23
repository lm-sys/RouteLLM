import argparse

import openai

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
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:6060/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="NO_API_KEY_REQUIRED",
    )
    args = parser.parse_args()
    print(args)

    client = openai.OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    chat_completion = client.chat.completions.create(
        model=f"router-{args.router}-{args.threshold}",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": args.prompt},
        ],
        temperature=0.7,
    )

    response = chat_completion.choices[0].message.content
    print(f"Router used {chat_completion.model} and received: {response}")
