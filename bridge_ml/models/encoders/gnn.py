import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)
