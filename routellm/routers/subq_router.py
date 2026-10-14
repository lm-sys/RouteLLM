import os
import requests
from routellm.routers.routers import Router


class SubQRouter(Router):
    """
    SubQ (System One) Calibrated Decision Engine Router.
    Evaluates query complexity via https://api.subq.ai/v1/systemone.
    Returns strong model winrate probability based on calibrated reasoning requirements.
    Website: https://subq.ai
    """

    def __init__(self, api_key=None, base_url="https://api.subq.ai/v1/systemone", model="subq-latest"):
        self.api_key = api_key or os.environ.get("SUBQ_API_KEY")
        self.base_url = base_url
        self.model = model

    def calculate_strong_win_rate(self, prompt):
        payload = {
            "model": self.model,
            "state": prompt,
            "questions": {
                "complexity": {
                    "prompt": "Evaluate cognitive reasoning depth required for this request.",
                    "instructions": "Determine if task requires strong multi-step reasoning model.",
                    "type": "choice",
                    "criteria": {
                        "complex": "Requires strong model (multi-step reasoning, coding, math).",
                        "simple": "Simple query suitable for weak model.",
                    },
                }
            },
        }
        headers = {"Content-Type": "application/json", "User-Agent": "RouteLLM-SubQ-Router/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            res = requests.post(self.base_url, json=payload, headers=headers, timeout=5.0)
            if res.status_code == 200:
                data = res.json()
                probs = data.get("evaluations", {}).get("complexity", {}).get("probabilities", {})
                return float(probs.get("complex", 0.5))
        except Exception:
            return 0.5

        return 0.5
