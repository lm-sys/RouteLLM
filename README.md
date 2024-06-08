# RouteLLM

RouteLLM is a framework for serving and evaluating routers for large language models.

The core setup is routing between a pair of LLMs - a strong model and a weak model. Each router takes in **only** the user prompt, and decides which LLM to route that request to using any strategy. Each routing request is also associated with a _cost threshold_, which is a user-specified value between 0 and 1 that determines the cost-quality tradeoff of that request. A higher cost threshold corresponds to a stronger restriction on the cost of the request, reducing cost but leading to a looser restriction on quality as well.

## Installation

**From source:**

```
git clone https://github.com/lm-sys/RouteLLM.git
cd RouteLLM
pip install -e .[serve]
```

## Server

RouteLLM offers a lightweight OpenAI-compatible server for routing requests between two LLMs based on different routing strategies. The server can be started with the following command:

```
python -m routellm.openai_server --config config.yaml --workers 1 --routers random sw_ranking
```

- `--routers` specifies the list of routers available to the server. For instance, here, the server is started with two available routers: `random` and `sw_ranking`.
- `--config` specifies the path to the configuration file, which contains the paths and settings required by each router.
- `--workers` specifies the number of workers to use with FastAPI.

## Evaluation

RouteLLM also includes a evaluation framework to measure the performance of different routing strategies on specific benchmarks. We currently support the following benchmarks: [MMLU](https://arxiv.org/abs/2009.03300), [GSM8K](https://arxiv.org/abs/2110.14168), and [MT Bench](https://arxiv.org/abs/2306.05685).

To evaluate a router on a benchmark, you can use the following command:

```
python -m routellm.evals.evaluate --routers random sw_ranking bert --benchmark gsm8k
```

- `--routers` specifies the list of routers to evaluate.
- `--benchmark` specifies the specific benchmark to evaluate the routers on.

By default, the evaluation results will be printed to the console. A plot of router performance will also be generated in the current directory (override using `--output`).

To avoid recomputing the results, the results for a router on a given benchmark is cached by default. This behavior can be overridden by using the `--overwrite-cache` flag, which takes in a list of routers to overwrite the cache for.

## Routers

Out of the box, RouteLLM supports the following routers trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair:

- `random`: Routes to a random LLM.
- `sw_ranking`
- `bert`
- `causal_llm`
- `matrix_factorization`

While these routers have been trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair, we have found that these routers generalize well to other strong and weak model pairs as well.

### Adding a new router

Adding a new router to RouteLLM is straightforward. You need to implement the abstract `Router` class in `routers.py` and add the new router to the `ROUTER_CLS` dictionary. Then, you can use immediately the new router in the server or evaluation framework.

There is only a single method to implement: `calculate_strong_win_rate`, which takes in the user prompt and returns the win rate for the strong model conditioned on that given prompt - if this win rate is great than user-specified cost threshold, then the request is routed to the strong model. Otherwise, it is routed to the weak model.
