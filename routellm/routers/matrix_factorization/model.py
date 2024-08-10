import torch
from huggingface_hub import PyTorchModelHubMixin

from routellm.routers.similarity_weighted.utils import OPENAI_CLIENT

MODEL_IDS = {
    "RWKV-4-Raven-14B": 0,
    "alpaca-13b": 1,
    "chatglm-6b": 2,
    "chatglm2-6b": 3,
    "chatglm3-6b": 4,
    "claude-1": 5,
    "claude-2.0": 6,
    "claude-2.1": 7,
    "claude-instant-1": 8,
    "codellama-34b-instruct": 9,
    "deepseek-llm-67b-chat": 10,
    "dolly-v2-12b": 11,
    "dolphin-2.2.1-mistral-7b": 12,
    "falcon-180b-chat": 13,
    "fastchat-t5-3b": 14,
    "gemini-pro": 15,
    "gemini-pro-dev-api": 16,
    "gpt-3.5-turbo-0125": 17,
    "gpt-3.5-turbo-0314": 18,
    "gpt-3.5-turbo-0613": 19,
    "gpt-3.5-turbo-1106": 20,
    "gpt-4-0125-preview": 21,
    "gpt-4-0314": 22,
    "gpt-4-0613": 23,
    "gpt-4-1106-preview": 24,
    "gpt4all-13b-snoozy": 25,
    "guanaco-33b": 26,
    "koala-13b": 27,
    "llama-13b": 28,
    "llama-2-13b-chat": 29,
    "llama-2-70b-chat": 30,
    "llama-2-7b-chat": 31,
    "llama2-70b-steerlm-chat": 32,
    "mistral-7b-instruct": 33,
    "mistral-7b-instruct-v0.2": 34,
    "mistral-medium": 35,
    "mixtral-8x7b-instruct-v0.1": 36,
    "mpt-30b-chat": 37,
    "mpt-7b-chat": 38,
    "nous-hermes-2-mixtral-8x7b-dpo": 39,
    "oasst-pythia-12b": 40,
    "openchat-3.5": 41,
    "openchat-3.5-0106": 42,
    "openhermes-2.5-mistral-7b": 43,
    "palm-2": 44,
    "pplx-70b-online": 45,
    "pplx-7b-online": 46,
    "qwen-14b-chat": 47,
    "qwen1.5-4b-chat": 48,
    "qwen1.5-72b-chat": 49,
    "qwen1.5-7b-chat": 50,
    "solar-10.7b-instruct-v1.0": 51,
    "stablelm-tuned-alpha-7b": 52,
    "starling-lm-7b-alpha": 53,
    "stripedhyena-nous-7b": 54,
    "tulu-2-dpo-70b": 55,
    "vicuna-13b": 56,
    "vicuna-33b": 57,
    "vicuna-7b": 58,
    "wizardlm-13b": 59,
    "wizardlm-70b": 60,
    "yi-34b-chat": 61,
    "zephyr-7b-alpha": 62,
    "zephyr-7b-beta": 63,
}


class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim=128,
        num_models=64,
        text_dim=768,
        num_classes=1,
        use_proj=True,
        collapse_linear=False,
        embedding_model="all-mpnet-base-v2",
    ):
        """
        Args:
            dim:
                Dimension of the model embeddings, default to 128
            num_models:
                Number of models, default to 64
            text_dim:
                Dimension of the text embeddings
                1536 for OpenAI's text-embedding-3-small
                768 for all-mpnet-base-v2
                1024 for infgrad/stella_en_400M_v5
            num_classes:
                Number of classes, default to 1, output a scalar
            use_proj:
                Whether to use projection for the text embeddings
                This is set to be True in our pretrained models for better performance
            collapse_linear:
                Whether to collapse the linear transformations into a single linear layer
                Since the current pretrained models only consist of Linear layers,
                we can collapse them into a single layer for faster inference
                See https://github.com/lm-sys/RouteLLM/issues/9
            embedding_model:
                Text embedding model for the prompt, should be the same as the one used in training
                Use all-mpnet-base-v2 to avoid OpenAI's key, however, slightly worse performance
                Use OpenAI's text-embedding-3-small for better performance
        """
        super().__init__()
        self.use_proj = use_proj
        self.collapse_linear = collapse_linear  # collapse the linear transformations into a single linear layer
        self.P = torch.nn.Embedding(num_models, dim)

        self.embedding_model = embedding_model

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes, bias=False)
        )

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_id, prompt):
        if self.embedding_model == "text-embedding-3-small":
            prompt_embed = (
                OPENAI_CLIENT.embeddings.create(
                    input=[prompt], model=self.embedding_model
                )
                .data[0]
                .embedding
            )
        elif self.embedding_model == "all-mpnet-base-v2":
            prompt_embed = self._embedding_model.encode([prompt])
        elif self.embedding_model == "infgrad/stella_en_400M_v5":
            prompt_embed = self._embedding_model.encode(
                [prompt], prompt_name="s2s_query"
            )
        else:
            raise ValueError(
                f"Unsupported embedding model {self.embedding_model}, "
                "should be one of text-embedding-3-small, all-mpnet-base-v2, infgrad/stella_en_400M_v5"
            )

        prompt_embed = torch.tensor(prompt_embed, device=self.get_device())
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())

        if self.collapse_linear:
            upscaled_model_embed = self.precompute_upscaled_embedding(model_id)
            return upscaled_model_embed @ prompt_embed.squeeze(-1)

        model_embed = self.P(model_id)
        prompt_embed = self.text_proj(prompt_embed)
        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def post_process_weight(self):
        # since the current model consist of only linear transformations
        # we can collapse the linear transformations into a single linear layer
        # https://github.com/lm-sys/RouteLLM/issues/9
        num_models = self.P.weight.shape[0]
        text_dim = self.text_proj[0].weight.shape[1]

        self.P.weight.data = torch.nn.functional.normalize(
            self.P.weight.data, p=2, dim=1
        )

        if (
            self.embedding_model == "all-mpnet-base-v2"
            or self.embedding_model == "infgrad/stella_en_400M_v5"
        ):
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.embedding_model, trust_remote_code=True
            ).to("cuda")

        if self.collapse_linear:
            self.precompute_upscaled_embedding = torch.nn.Embedding(
                num_models, text_dim
            )
            self.precompute_upscaled_embedding.weight.data = (
                self.P.weight * self.classifier[0].weight.data
            ) @ self.text_proj[0].weight.data
