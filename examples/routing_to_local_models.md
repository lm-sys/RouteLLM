# Routing to Local Models with RouteLLM and Ollama

A common use-case is routing between GPT-4 as the strong model and a local model as the weak model. This allows you to only use GPT-4 for queries that require it, saving costs while maintaining response quality.

Let's route between GPT-4 and a local Llama 3 8B as an example. Make sure Ollama is [installed](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) beforehand.

1. Run Llama 3 8B locally using Ollama:
```
ollama run llama3
```
Now, the Ollama server will be running at `http://localhost:11434/v1`.

Next, you have 2 options depending on your use case: either replace the OpenAI client in your Python code, or launch an OpenAI-compatible server.

## Option A: Replace existing OpenAI client

2. Create a RouteLLM controller with the `mf` router, specifying the local Llama 3 8B as the weak model:
```python
os.environ["OPENAI_API_KEY"] = "sk-XXXXXX"

client = Controller(
  routers=["mf"],
  strong_model="gpt-4-1106-preview",
  weak_model="ollama_chat/llama3",
)
```

3. Update the `model` field in when generating completions:
```python
response = client.chat.completions.create(
  model="router-mf-0.11593",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)
```
In the [Quickstart](../README.md#quickstart) section, we calibrated the threshold to be `0.11593` for `mf` so that we get approximately 50% of queries routed to GPT-4, which we set in the `model` field here.

And that's it! Now, our requests will be routed between GPT-4 for more difficult queries and our local Llama-3 8B model for simpler queries.

## Option B: Launch an OpenAI-compatible Server

2. Launch a server with the `mf` router:
```
> export OPENAI_API_KEY=sk-...
> python -m routellm.openai_server --routers mf --weak-model ollama_chat/llama3 --config.example.yaml
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:6060 (Press CTRL+C to quit)
```
The server is now listening on `http://0.0.0.0:6060`. We use the `--weak-model` flag to use point to the Llama 3 model that is running locally on our machine.

3. Point your OpenAI client to the RouteLLM server:
```python
import openai

client = openai.OpenAI(
  base_url="https://localhost:6060/v1",
  api_key="no_api_key"
)
...
response = client.chat.completions.create(
  model="router-mf-0.11593",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)
```
In the [Quickstart](../README.md#quickstart) section, we calibrated the threshold to be `0.11593` for `mf` so that we get approximately 50% of queries routed to GPT-4, which we set in the `model` field here.

And that's it! Now, our requests will be routed between GPT-4 for more difficult queries and our local Llama-3 8B model for simpler queries.
