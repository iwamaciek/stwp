import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from baselines.config import FH


class TemporalGNN(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_dim):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(
            in_channels=input_features, out_channels=hidden_dim, periods=FH
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        # self.linear3 = torch.nn.Linear(hidden_dim // 2, hidden_dim // 4)
        # self.linear4 = torch.nn.Linear(hidden_dim // 4, hidden_dim // 8)
        # self.linear5 = torch.nn.Linear(hidden_dim // 8, FH * input_features)
        self.linear = torch.nn.Linear(hidden_dim, FH * input_features)

    def forward(self, x, edge_index, edge_weights):
        # torch.Size([2250, 6, 3])
        h = self.tgnn(x, edge_index, edge_weights)
        # torch.Size([2250, hidden_dim])
        h = F.relu(h)
        # h = self.batch_norm(h)
        h = self.linear1(h)
        h = self.linear2(h)
        h = self.linear3(h)
        h = self.linear4(h)
        h = self.linear(h)
        return h.view(-1, x.size(1), FH)
