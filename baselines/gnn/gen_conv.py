import torch
import torch.nn as nn
from torch_geometric.nn.conv import GENConv
from baselines.config import INPUT_SIZE, FH
from baselines.gnn.st_encoder_module import SpatioTemporalEncoder

N = 5


class GENConvNN(torch.nn.Module):
    def __init__(self, input_features, output_features, edge_dim, hidden_dim, input_t_dim=4, input_s_dim=6):
        super(GENConvNN, self).__init__()
        self.mlp_embedder = nn.Linear(input_features * INPUT_SIZE, hidden_dim)
        self.st_encoder = SpatioTemporalEncoder(hidden_dim, input_t_dim, input_s_dim, hidden_dim)
        self.layer_norm_embed = nn.LayerNorm(hidden_dim)
        self.gengnn = GENConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            num_layers=N,
        )
        self.mlp_decoder = nn.Linear(hidden_dim, output_features * FH)

    def forward(self, x, edge_index, edge_attr, t, s):
        x = x.view(-1, x.size(-2) * x.size(-1))
        x = self.mlp_embedder(x)
        x = self.st_encoder(x, t, s)
        x = self.layer_norm_embed(x).relu()
        x = self.gengnn(x, edge_index, edge_attr).relu()
        x = self.mlp_decoder(x)
        return x.view(x.size(0), x.size(1) // FH, FH)
