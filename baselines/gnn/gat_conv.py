import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv
from baselines.config import INPUT_SIZE, FH

N = 5


class GATConvNN(torch.nn.Module):
    def __init__(self, input_features, output_features, edge_dim, hidden_dim):
        super(GATConvNN, self).__init__()
        self.mlp_embedder = nn.Linear(input_features * INPUT_SIZE, hidden_dim)
        self.layer_norm_embed = nn.LayerNorm(hidden_dim)
        self.gatnns = nn.ModuleList(
            [
                GATConv(
                    in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=edge_dim
                )
                for _ in range(N)
            ]
        )
        self.mlp_decoder = nn.Linear(hidden_dim, output_features * FH)
        self.layer_norm_decoder = nn.LayerNorm(output_features * FH)

    def forward(self, x, edge_index, edge_attr):
        x = x.view(-1, x.size(-2) * x.size(-1))
        x = self.mlp_embedder(x)
        x = self.layer_norm_embed(x).relu()
        for gatnn in self.gatnns:
            x = gatnn(x, edge_index, edge_attr).relu()
        x = self.mlp_decoder(x)
        return x.view(x.size(0), x.size(1) // FH, FH)
