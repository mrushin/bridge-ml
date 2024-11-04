import torch
import torch.nn as nn

class SemanticDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.alignment_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x1, x2):
        return self.alignment_layer(torch.cat([x1, x2], dim=-1))
