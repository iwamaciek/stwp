import torch
import torch.nn as nn
from models.config import config as cfg


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, input_X_dim, input_t_dim, input_s_dim, output_dim, hidden=16):
        self.input_X_dim = input_X_dim
        super(SpatioTemporalEncoder, self).__init__()
        self.lat, self.lon = cfg.INPUT_DIMS
        self.mlp_embedder = nn.Linear(input_X_dim, hidden)
        self.temporal_embedder = nn.Linear(input_t_dim, self.lat * self.lon)
        self.mlp_decoder = nn.Linear(hidden + 1 + input_s_dim, output_dim)

    def forward(self, X, t, s):
        batch_size = s.shape[0]
        X = X.reshape((batch_size, self.lat * self.lon, self.input_X_dim))
        X = self.mlp_embedder(X).relu()
        t = (
            self.temporal_embedder(t.reshape(batch_size, 1, -1))
            .relu()
            .permute((0, 2, 1))
        )
        concat = torch.cat((X, t, s), dim=-1)
        output = self.mlp_decoder(concat)
        output = output.reshape((-1, self.input_X_dim))
        return output
