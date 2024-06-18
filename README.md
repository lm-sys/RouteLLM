# RouteLLM

RouteLLM is a framework for serving and evaluating routers for large language models.

The core setup is routing between a pair of LLMs - a strong model and a weak model. Each router takes in **only** the user prompt, and decides which LLM to route that request to using any strategy. Each routing request is also associated with a _cost threshold_, which is a user-specified value between 0 and 1 that determines the cost-quality tradeoff of that request. A higher cost threshold corresponds to a stronger restriction on the cost of the request, reducing cost but leading to a looser restriction on quality as well.

<p float="left">
  <img src="assets/gsm8k.png" width="40%" />
  <img src="assets/mt-bench.png" width="40%" />
</p>

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

This means that each server can support multiple routing strategies - users can specify which router and what cost threshold to use for each request using the model name with the following format `router-[ROUTER NAME]-[THRESHOLD]`. For instance, using model name of `router-bert-0.5` specifies that the request should be routed using the BERT router with a cost threshold of 0.5.

For OpenAI models, the server will route requests to the official OpenAI client. For all other models, the server will route to an alternate OpenAI compatible server, which can be configured with the `--alt-base-url` and `alt-api-key` flags (uses Anyscale by default).

## Evaluation

RouteLLM also includes a evaluation framework to measure the performance of different routing strategies on specific benchmarks. We currently support the following benchmarks: [MMLU](https://arxiv.org/abs/2009.03300), [GSM8K](https://arxiv.org/abs/2110.14168), and [MT Bench](https://arxiv.org/abs/2306.05685).

To evaluate a router on a benchmark, you can use the following command:

```
python -m routellm.evals.evaluate --routers random sw_ranking bert --benchmark gsm8k
```

- `--routers` specifies the list of routers to evaluate.
- `--benchmark` specifies the specific benchmark to evaluate the routers on.

By default, the evaluation results will be printed to the console. A plot of router performance will also be generated in the current directory (override using `--output`). To avoid recomputing results, the results for a router on a given benchmark is cached by default. This behavior can be overridden by using the `--overwrite-cache` flag, which takes in a list of routers to overwrite the cache for.

The results for all our benchmarks are cached for speed. For MT Bench, we use the precomputed judgements for the desired model pair. For MMLU and GSM8K, we utilized [SGLang](https://github.com/sgl-project/sglang) to efficiently compute the results for the desired model pair and stored these results - the full code for this can be found in the respective benchmark directories.

## Routers

Out of the box, RouteLLM supports the following routers trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair:

- `random`: Randomly routes to either model.
- `sw_ranking`: Uses Elo calculations for routing, weighted according to the similarity of the prompt to the preference data.
- `bert`: Uses a BERT classifier trained on the preference data.
- `causal_llm`: Uses a LLM-based classifier tuned on the preference data.
- `matrix_factorization`: Uses a matrix factorization model trained on the preference data.

For the full details of how these routers were trained, please refer to our paper.

While these routers have been trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair, we have found that these routers generalize well to other strong and weak model pairs as well (see Section 4.4 of our paper).

## Configuration

The configuration for all routers is contained in single YAML file, which is a top-level mapping from router name to the keyword arguments used for router initialization. An example configuration is provided in the `config.example.yaml` file - it provides the configurations for routers that have trained on Arena data augmented using GPT-4 as a judge, as discussed in our paper. The models and datasets used are all hosted on Hugging Face under the [RouteLLM](https://huggingface.co/routellm) and [LMSYS](https://huggingface.co/lmsys) organizations.

```yaml
sw_ranking:
    arena_battle_datasets:
      - lmsys/lmsys-arena-human-preference-55k
      - routellm/gpt4_judge_battles
    arena_embedding_datasets:
      - routellm/arena_battles_embeddings
      - routellm/gpt4_judge_battles_embeddings
    strong_model: gpt-4-1106-preview
    weak_model: mixtral-8x7b-instruct-v0.1
causal_llm:
    checkpoint_path: routellm/causal_llm_augmented
    system_message: routellm/routers/causal_llm/system_ft_v5.txt
    classifier_message: routellm/routers/causal_llm/classifier_ft_v5.txt
bert:
    checkpoint_path: routellm/bert_gpt4_augmented
matrix_factorization:
    checkpoint_path: routellm/matrix_factorization_gpt4_augmented
    hidden_size: 128
    strong_model: gpt-4-1106-preview
    weak_model: mixtral-8x7b-instruct-v0.1
```

## Contribution

We welcome contributions! Please feel free to open an issue or a pull request if you have any suggestions or improvements.

### Adding a new router

To add a new router to RouteLLM, implement the abstract `Router` class in `routers.py` and add the new router to the `ROUTER_CLS` dictionary. Then, you can use immediately the new router in the server or evaluation framework.

There is only a single method to implement: `calculate_strong_win_rate`, which takes in the user prompt and returns the win rate for the strong model conditioned on that given prompt - if this win rate is great than user-specified cost threshold, then the request is routed to the strong model. Otherwise, it is routed to the weak model.

### Adding a new benchmark

To add a new benchmark to RouteLLM, implement the abstract `Benchmark` class in `benchmarks.py` and update the `evaluate.py` module to properly initialize the new benchmark class. Ideally, the results for the benchmark should be precomputed to avoid having to regenerate the results for each evaluation run -- see the existing benchmarks for examples on how to do this.
