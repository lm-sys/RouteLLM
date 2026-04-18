"""
Quick smoke-test for the RouteLLM setup.
Run AFTER start.py is running.

  python test_routing.py
"""

import openai
import time

client = openai.OpenAI(
    base_url="http://localhost:6060/v1",
    api_key="no-key",
)

ROUTER = "mf"
THRESHOLD = 0.11593
MODEL = f"router-{ROUTER}-{THRESHOLD}"

TESTS = [
    ("easy",   "What is 2+2?"),
    ("medium", "Explain what a REST API is in one sentence."),
    ("hard",   "Describe three subtle differences between Python's GIL and Java's memory model in terms of concurrency guarantees."),
]

print(f"Testing RouteLLM at http://localhost:6060 with model={MODEL}\n")
for label, prompt in TESTS:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0
    answer = resp.choices[0].message.content.strip()
    routed_to = getattr(resp, "model", "unknown")
    print(f"[{label:6s}] ({elapsed:.1f}s) → {routed_to}")
    print(f"  Q: {prompt[:60]}")
    print(f"  A: {answer[:80]}\n")
