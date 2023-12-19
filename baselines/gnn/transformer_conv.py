import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from baselines.config import INPUT_SIZE, FH
from torch_geometric.nn.dense import DenseGCNConv, DenseSAGEConv

N = 5


class TransformerGNN(torch.nn.Module):
    def __init__(self, input_features, output_features, edge_dim, hidden_dim):
        super(TransformerGNN, self).__init__()
        # self.mlp_embedder = nn.Linear(input_features * INPUT_SIZE, hidden_dim)
        # self.mlp_embedder = DenseGCNConv(input_features * INPUT_SIZE, hidden_dim)
        self.mlp_embedder = DenseSAGEConv(input_features * INPUT_SIZE, hidden_dim)

        self.layer_norm_embed = nn.LayerNorm(hidden_dim)
        self.transgnns = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=edge_dim
                )
                for _ in range(N)
            ]
        )
        # self.mlp_decoder = nn.Linear(hidden_dim, output_features * FH)
        # self.mlp_decoder = DenseGCNConv(hidden_dim, output_features * FH)
        self.mlp_decoder = DenseSAGEConv(hidden_dim, output_features * FH)

    def initialize_last_layer_bias(self, mean_features):
        self.mlp_decoder.bias.data.zero_()
        self.mlp_decoder.bias.data.add_(mean_features)

    def forward(self, x, edge_index, edge_attr, adj):
        x = x.view(-1, x.size(-2) * x.size(-1))
        x = self.mlp_embedder(x, adj)
        # x = self.mlp_embedder(x)
        x = self.layer_norm_embed(x).relu()
        for transgnn in self.transgnns:
            x = transgnn(x, edge_index, edge_attr).relu()
        x = self.mlp_decoder(x, adj)
        # x = self.mlp_decoder(x)
        x = x.permute((1, 2, 0))
        x = x.view(x.size(0), x.size(1) * x.size(2), FH)
        # x = x.view(x.size(0), x.size(1) // FH, FH)
        return x
