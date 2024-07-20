import torch.nn as nn
import torch


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, hidden_size: int = 200):   # TODO: configuration
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, hidden_size)   # hidden_size 768
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.functional.gelu(x)  # type: ignore
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias
