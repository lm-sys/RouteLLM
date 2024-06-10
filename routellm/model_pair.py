from dataclasses import dataclass


@dataclass
class ModelPair:
    strong: str
    weak: str


ROUTED_PAIR = ModelPair(
    strong="gpt-4-1106-preview", weak="mistralai/Mixtral-8x7B-Instruct-v0.1"
)
