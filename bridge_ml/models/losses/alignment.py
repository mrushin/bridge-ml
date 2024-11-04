import torch
import torch.nn as nn

class AlignmentLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, similarities, labels):
        return nn.functional.margin_ranking_loss(
            similarities, labels, margin=self.margin
        )
