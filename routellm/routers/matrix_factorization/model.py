import torch
from openai import OpenAI


class MFModel(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    ):
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)

        self.client = OpenAI()
        self.embedding_model = "text-embedding-3-small"

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
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())

        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)

        prompt_embed = (
            self.client.embeddings.create(input=[prompt], model=self.embedding_model)
            .data[0]
            .embedding
        )
        prompt_embed = torch.tensor(prompt_embed, device=self.get_device())
        prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))
