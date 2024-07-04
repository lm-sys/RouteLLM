The main way to interface with RouteLLM in your applications is through our OpenAI-compatible server, which you can run locally or in the cloud. Let's walkthrough a minimal example, which you can modify to suit your needs.

1. Say I want to run the MF router (recommended) locally using the default model pair (GPT-4 / Mixtral 8x7B), I first launch the server:
```
> python -m routellm.openai_server --routers mf --config config.example.yaml 
Launching server with routers: ['mf']
INFO:     Started server process [92737]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:6060 (Press CTRL+C to quit)
```
The server is now listening on `http://0.0.0.0:6060`

2. Next, I want to calibrate my threshold for this router so I know what threshold to use for routing. The threshold is what controls the cost-quality tradeoff of routing. If you have some knowledge about the type of queries you are going to serve, you can get a more accurate threshold by calibrating on that dataset using the `calibrate_threshold` script. In this case, I'm going to calibrate based on the publicly-available LMSYS chat dataset (https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k). Say I want approximately 50% of my calls to be routed to GPT-4, managing my cost while maximizing quality within this cost range.
```
> python -m routellm.calibrate_threshold --task calibrate --routers mf causal_llm bert --strong-model-pct 0.5 --config config.example.yaml
For 50.0% strong model calls, calibrated threshold for mf: 0.11592505872249603
```

This means that I'll want to use `0.116` as my threshold to get approximately 50% of queries routed to GPT-4. (Note that if your input queries differ a lot from the dataset used to calibrate, then the % of calls routed to GPT-4 can differ, so it's recommended to calibrate on a dataset closest to the type of queries you will receive).

3. Now, I can use it in my Python application to generate completions just like I would any OpenAI model.
```
import openai
client = openai.OpenAI(
	base_url="https://localhost:6060/v1",
	api_key="no_api_key"
)
response = client.chat.completions.create(
	model="router-mf-0.116",
	messages=[
		{"role": "user", "content": "What is a roux?"}
	]
)
print(response.choices[0].message.content)
```
Here, I'm setting the `model` to `router-mf-0.116` to mean: "I want to use the MF router with a threshold of `0.116`".
Importantly, the server that we have launched will also work with any other application that uses the OpenAI endpoint - you just need to update the base URL.

That's it! Depending on your needs, you would want to consider hosting the server on the cloud, using a different strong / weak model pair for routing, and also calibrating the thresholds differently using knowledge about the types of queries you will receive.