# Routing to Local Models with RouteLLM and Ollama

A common use-case is routing between GPT-4 as the strong model and a local model as the weak model. This allows you to only use GPT-4 for queries that require it, saving costs while maintaining response quality.

Let's route between GPT-4 and a local Llama 3 8B as an example. Make sure Ollama is [installed](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) beforehand.

1. Run Llama 3 8B locally using Ollama:
```
ollama run llama3
```
Now, the Ollama server will be running at `http://localhost:11434/v1`.

2. Launch RouteLLM server with the `mf` router (recommended):
```
> export OPENAI_API_KEY=sk-...
> python -m routellm.openai_server --routers mf --weak-model ollama/llama3 --config.example.yaml
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:6060 (Press CTRL+C to quit)
```
The server is now listening on `http://0.0.0.0:6060`. We use the `--weak-model` flag to use point to the Llama 3 instance that is running locally on our machine.

3. Point your OpenAI client to the RouteLLM server:
```python
import openai

client = openai.OpenAI(
  base_url="https://localhost:6060/v1",
  # Required but ignored
  api_key="no_api_key"
)
...
response = client.chat.completions.create(
  # "Use the MF router with a threshold of 0.116"
  model="router-mf-0.116",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)
```
In the [Quickstart](../README.md#quickstart) section, we calibrated the threshold to be `0.116` for `mf` so that we get approximately 50% of queries routed to GPT-4, which we set in the `model` field here.

And that's it! Now, our requests will be routed between GPT-4 for more difficult queries and our local Llama-3 8B model for simpler queries.
