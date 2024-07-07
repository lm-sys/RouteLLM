# RouteLLM Python SDK

You can interface with RouteLLM in Python via the `Controller` class.

```python
import os
from routellm.controller import Controller

os.environ["OPENAI_API_KEY"] = "sk-XXXXXX"
os.environ["ANYSCALE_API_KEY"] = "esecret_XXXXXX"

client = Controller(
  # List of routers to initialize
  routers=["mf"],
  # The pair of strong and weak models to route to
  strong_model="gpt-4-1106-preview",
  weak_model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
  # The config for the router (best-performing config by default)
  config = {
    "mf": {
      "checkpoint_path": "routellm/mf_gpt4_augmented"
    }
  },
  # Override API base and key for LLM calls
  api_base=None,
  api_key=None,
  # Display a progress bar for operations
  progress_bar=False,
)
```

The controller is a drop-in replacement for OpenAI's client and supports the same chat completions interface. You can call `acreate` for the async equivalent.

```python
response = client.chat.completions.create(
  # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
  model="router-mf-0.11593",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)
print(response.choices[0]["message"]["content"])
```
You can also access these methods at `controller.completion` and `controller.acompletion`.


In addition, the controller also supports a `route` method that returns the best model for a given prompt.
```python
routed_model = client.route(
	prompt="What's the squareroot of 144?",
	router="mf",
	threshold=0.11593,
)
print(f"Prompt should be routed to {routed_model}")
```

Finally, the controller also supports a `batch_calculate_win_rate` method that takes in a `Series` of prompts and return the of win rate for the strong model on each prompt as calculated by the specified router. This is mainly used for offline evaluations and will parallelize operations wherever possible.
```python
import pandas as pd

prompts = pd.Series(["What's the squareroot of 144?", "Who's the last president of the US?", "Is the sun a star?"])
win_rates = client.batch_calculate_win_rate(prompts=prompts, router="mf")

print(f"Calculated win rate for prompts:\n{win_rates.describe()}")
```
