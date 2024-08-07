"""
Based on https://github.com/vllm-project/vllm/blob/main/examples/gradio_openai_chatbot_webserver.py
"""

import argparse
import re

import gradio as gr
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(description="Chatbot Interface for RouteLLM")
parser.add_argument(
    "--model-url", type=str, default="http://localhost:6060/v1", help="Model URL"
)
parser.add_argument(
    "-r", "--router", type=str, required=True, help="Router name to use for the chatbot"
)
parser.add_argument(
    "--threshold", type=float, required=True, help="Cost threshold to use for routing"
)
parser.add_argument(
    "--temp", type=float, default=0.8, help="Temperature for text generation"
)
parser.add_argument(
    "--stop-token-ids", type=str, default="", help="Comma-separated stop token IDs"
)
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def predict(message, history, threshold, router, temperature):
    # Convert chat history to OpenAI format
    history_openai_format = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append(
            {
                "role": "assistant",
                # Remove model name from response
                "content": re.sub(r"^\[.*?\]\s*", "", assistant),
            }
        )
    history_openai_format.append({"role": "user", "content": message})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=f"router-{router}-{threshold}",  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=temperature,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            "repetition_penalty": 1,
            "stop_token_ids": (
                [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
                if args.stop_token_ids
                else []
            ),
        },
    )
    print(stream)

    # Read and return generated text from response stream
    partial_message = ""
    for i, chunk in enumerate(stream):
        print(chunk)
        if i == 0:
            model_prefix = f"[{chunk.model}]\n"
            yield model_prefix
            partial_message += model_prefix
        partial_message += chunk.choices[0].delta.content or ""
        yield partial_message


# Create and launch a chat interface with Gradio
gr.ChatInterface(
    predict,
    additional_inputs=[
        gr.Number(label="Threshold", value=args.threshold),
        gr.Textbox(label="Router", value=args.router),
        gr.Slider(label="Temperature", minimum=0, maximum=1, value=args.temp, step=0.1),
    ],
).queue().launch(server_name=args.host, server_port=args.port, share=True)
