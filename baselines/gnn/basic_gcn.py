import torch
import torch.nn as nn
from torch_geometric.nn.models import GCN
from baselines.config import INPUT_SIZE, FH

N = 5


class BasicGCN(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_dim):
        super(BasicGCN, self).__init__()
        self.mlp_embedder = nn.Linear(INPUT_SIZE * input_features, hidden_dim)
        self.gcn = GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=N)
        self.mlp_decoder = nn.Linear(hidden_dim, output_features * FH)

    def forward(self, x, edge_index, edge_weights):
        x = x.view(-1, x.size(-1) * x.size(-2))
        x = self.mlp_embedder(x).relu()
        x = self.gcn(x, edge_index, edge_weights).relu()
        x = self.mlp_decoder(x)
        return x.view(x.size(0), x.size(1) // FH, FH)
