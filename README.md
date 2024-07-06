# RouteLLM

RouteLLM is a framework for serving and evaluating LLM routers.  
[ [Blog](http://lmsys.org/blog/2024-07-01-routellm/) ] [ [Paper](https://arxiv.org/abs/2406.18665) ]

<p align="center">
  <img src="assets/router.png" width="50%" />
</p>

Our core features include:

- Launch an OpenAI-compatible API that takes in user requests and routes them to the best model for that specific request using a single command.
- Trained routers are provided out of the box, which we have shown to **reduce costs by up to 85%** on widely-used benchmarks such as MT Bench while maintaining **95% GPT-4 performance**.
- Easily extend the framework to include new routers and benchmarks, and compare the performance of all routers with a single command.

## Installation

**From PyPI**
```
pip install "routellm[serve,eval]"
```


**From source**

```
git clone https://github.com/lm-sys/RouteLLM.git
cd RouteLLM
pip install -e .[serve,eval]
```

## Quickstart

Let's walkthrough setting up a RouteLLM server and pointing our existing OpenAI client to it.

1. First, launch the RouteLLM server with the `mf` router:
```
> export OPENAI_API_KEY=sk-XXXXXX
> export ANYSCALE_API_KEY=esecret_XXXXXX
> python -m routellm.openai_server --routers mf --weak-model anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1 ---config config.example.yaml
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:6060 (Press CTRL+C to quit)
```
The server is now listening on `http://0.0.0.0:6060`. By default, the router will route between GPT-4 and Mixtral 8x7B, so you'll need to configure your API keys for OpenAI and a model provider for Mixtral 8x7B beforehand (we use Anyscale by setting the API key and pointing our weak model to it above).

You can also route between a different model pair by specifying the `--strong-model` and `--weak-model` flags (see [Model Support](#model-support) and [Routing to Local Models](examples/routing_to_local_models.md)).

2. The *cost threshold* controls the tradeoff between cost and quality for routing, and depends on both the router and dataset. Let's calibrate our threshold for 50% GPT-4 calls using public Chatbot Arena data:
```
> python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5 --config config.example.yaml
For 50.0% strong model calls, calibrated threshold for mf: 0.11592505872249603
```
This means that I'll want to use `0.116` as my cost threshold to get approximately 50% of queries routed to GPT-4 (see [Threshold Calibration](#threshold-calibration)).

3. Now, let's point our existing OpenAI client to RouteLLM and specify the router:
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
That's it! Now, requests with be routed between the strong and weak model depending on what is required, **saving costs while maintaining a high quality of responses**.

Depending on your use case, you might want to consider hosting the server on the cloud, using a different model pair, and calibrating the thresholds based on the types of queries you will receive to improve performance.

### Demo

Once the server is launched, you can also launch a local chat interface to experiment with the router to see how different requests are routed.
```
python -m examples.router_chat --router mf --threshold 0.116
```

<p align="center">
  <img src="assets/chat-interface.png" width="50%" />
</p>

### Model Support

By default, GPT-4 and Mixtral 8x7B are used as the model pair for serving. To modify the model pair used, set them using the `--strong-model` and `--weak-model` flags. However, regardless of the model pair, an `OPENAI_API_KEY` is required for generating embeddings.

We leverage [LiteLLM](https://github.com/BerriAI/litellm) to support chat completions from a wide-range of open-source and closed models. In general, you need a setup an API key and point to the provider with the appropriate model name using the `--strong-model` or `--weak-model` flag. Alternatively, you can also use **any OpenAI-compatible endpoint** by prefixing the model name with `openai/` using the `--alt-base-url` and `--alt-api-key` flags to point to the server.

See [Routing to Local Models](examples/routing_to_local_models.md) for a walkthrough of routing to local models using Ollama.

Instructions for other popular providers:
- [Anthropic](https://litellm.vercel.app/docs/providers/anthropic#api-keys)
- [Gemini - Google AI Studio](https://litellm.vercel.app/docs/providers/gemini#sample-usage)
- [Amazon Bedrock](https://litellm.vercel.app/docs/providers/bedrock#required-environment-variables)
- [Together AI](https://litellm.vercel.app/docs/providers/togetherai#api-keys)
- [Anyscale Endpoints](https://litellm.vercel.app/docs/providers/anyscale#api-key)

For other model providers, view the instructions [here](https://litellm.vercel.app/docs/providers). 

## Motivation

Different LLMs vary widely in their costs and capabilities, which leads to a dilemma when deploying them: routing all queries to the most capable model leads to the highest-quality responses but can be very expensive, while routing queries to smaller models can save costs but may result in lower-quality responses. 

*LLM routing* offers a solution to this. We introduce a router that looks at queries and routes simpler queries to smaller, cheaper models, saving costs while maintaining quality. We focus on routing between 2 models: a stronger, more expensive model and a cheaper but weaker model. Each request is also associated with a _cost threshold_ that determines the cost-quality tradeoff of that request - a higher cost threshold leads to lower cost but may lead to lower-quality responses.

## Server

RouteLLM offers a lightweight OpenAI-compatible server for routing requests based on different routing strategies:

```
python -m routellm.openai_server --routers mf --config config.example.yaml 
```

- `--routers` specifies the list of routers available to the server. For instance, here, the server is started with one available router: `mf` (see below for the list of routers).
- `--config` specifies the path to the configuration file for the routers (see Configuration section)

For most use-cases, **we recommend the `mf` router** as we have evaluated it to be very strong and lightweight.

When making a request to the server, clients specify which router and what cost threshold to use for each request using the `model` field in the following format `router-[ROUTER NAME]-[THRESHOLD]`. For instance, using a `model` of `router-mf-0.5` specifies that the request should be routed using the `mf` router with a cost threshold of 0.5.

### Threshold Calibration

The threshold used for routing controls the cost-quality tradeoff. The range of meaningful thresholds varies depending on the type of router and the queries you receive. Therefore, we recommend calibrating thresholds using a sample of your incoming queries, as well as the % of queries you'd like to route to the stronger model.

By default, we support calibrating thresholds based on the public [Chatbot Arena dataset](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k). For example, to calibrate the threshold for the `mf` router such that 50% of calls are routed to the stronger model:

```
> python -m routellm.calibrate_threshold --task calibrate --routers mf --strong-model-pct 0.5 --config config.example.yaml
For 50.0% strong model calls, calibrated threshold for mf: 0.11592505872249603
```

This means that the threshold should be set to 0.1881 for the `mf` router so that approximately 50% of calls are routed to the strong model i.e. using a `model` field of `router-mf-0.1159`.

However, note that because we calibrate the thresholds based on an existing dataset, the % of calls routed to each model will differ based on the actual queries received. Therefore, we recommend calibrating on a dataset that closely resembles the types of queries you receive.

## Evaluation

RouteLLM also includes a evaluation framework to measure the performance of different routing strategies on benchmarks.

To evaluate a router on a benchmark, you can use the following command:
```
python -m routellm.evals.evaluate --routers random sw_ranking bert --benchmark gsm8k --config config.example.yaml 
```

- `--routers` specifies the list of routers to evaluate, for instance, `random` and `bert` in this case.
- `--benchmark` specifies the specific benchmark to evaluate the routers on. We currently support: `mmlu`, `gsm8k`, and `mt-bench`.

Evaluation results will be printed to the console. A plot of router performance will also be generated in the current directory (override the path using `--output`). To avoid recomputing results, the results for a router on a given benchmark is cached by default. This behavior can be overridden by using the `--overwrite-cache` flag, which takes in a list of routers to overwrite the cache for.

The results for all our benchmarks have been cached. For MT Bench, we use the precomputed judgements for the desired model pair. For MMLU and GSM8K, we utilized [SGLang](https://github.com/sgl-project/sglang) to compute the results for the desired model pair - the full code for this can be found in the benchmark directories if you would like to evaluate a different model pair.

By default, GPT-4 and Mixtral are used as the model pair for evaluation. To modify the model pair used, set them using the `--strong-model` and `--weak-model` flags.

## Routers

Out of the box, RouteLLM supports 4 routers trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair.

The full list of routers:
1. `mf`: Uses a matrix factorization model trained on the preference data. (recommended)
2. `sw_ranking`: Uses a weighted Elo calculation for routing, where each vote is weighted according to how similar it is to the user's prompt.
3. `bert`: Uses a BERT classifier trained on the preference data.
4. `causal_llm`: Uses a LLM-based classifier tuned on the preference data.
5. `random`: Randomly routes to either model.

While these routers have been trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair, we have found that these routers generalize well to other strong and weak model pairs as well. Therefore, you can replace the model pair used for routing without having to retrain these models!

 For the full details, refer to our [paper](https://arxiv.org/abs/2406.18665).

## Configuration

The configuration for all routers is specified in single YAML file, which is a top-level mapping from router name to the keyword arguments used for router initialization.

An example configuration is provided in the `config.example.yaml` file - it provides the configurations for routers that have trained on Arena data augmented using GPT-4 as a judge. The models and datasets used are all hosted on Hugging Face under the [RouteLLM](https://huggingface.co/routellm) and [LMSYS](https://huggingface.co/lmsys) organizations.

## Contribution

We welcome contributions! Please feel free to open an issue or a pull request if you have any suggestions or improvements.

### Adding a new router

To add a new router to RouteLLM, implement the abstract `Router` class in `routers.py` and add the new router to the `ROUTER_CLS` dictionary. Then, you can use immediately the new router in the server or evaluation framework.

There is only a single method to implement: `calculate_strong_win_rate`, which takes in the user prompt and returns the win rate for the strong model conditioned on that given prompt - if this win rate is great than user-specified cost threshold, then the request is routed to the strong model. Otherwise, it is routed to the weak model.

### Adding a new benchmark

To add a new benchmark to RouteLLM, implement the abstract `Benchmark` class in `benchmarks.py` and update the `evaluate.py` module to properly initialize the new benchmark class. Ideally, the results for the benchmark should be precomputed to avoid having to regenerate the results for each evaluation run -- see the existing benchmarks for examples on how to do this.

# Citation

The code in this repository is based on the research from the [paper](https://arxiv.org/abs/2406.18665). Please cite if you find the repository helpful.

```
@misc{ong2024routellmlearningroutellms,
      title={RouteLLM: Learning to Route LLMs with Preference Data},
      author={Isaac Ong and Amjad Almahairi and Vincent Wu and Wei-Lin Chiang and Tianhao Wu and Joseph E. Gonzalez and M Waleed Kadous and Ion Stoica},
      year={2024},
      eprint={2406.18665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.18665},
}
```
