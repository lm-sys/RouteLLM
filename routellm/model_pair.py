from dataclasses import dataclass


@dataclass
class ModelPair:
    strong: str
    weak: str


DEFAULT_PAIR = ModelPair(strong="gpt-4-1106-preview", weak="mixtral-8x7b-instruct-v0.1")
