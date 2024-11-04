import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)

    def forward(self, x):
        return self.self_attn(x, x, x)[0]
