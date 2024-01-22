import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv, CGConv, GATConv, GENConv, PDNConv
from models.config import config as cfg
from models.gnn.st_encoder_module import SpatioTemporalEncoder


class GNNModule(torch.nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        edge_dim,
        hidden_dim,
        input_t_dim=4,
        input_s_dim=6,
        input_size=cfg.INPUT_SIZE,
        fh=cfg.FH,
        arch="trans",
        num_graph_cells=cfg.GRAPH_CELLS,
    ):
        super(GNNModule, self).__init__()
        self.mlp_embedder = nn.Linear(input_features * input_size, hidden_dim)
        self.st_encoder = SpatioTemporalEncoder(
            hidden_dim, input_t_dim, input_s_dim, hidden_dim
        )
        self.layer_norm_embed = nn.LayerNorm(hidden_dim)
        self.gnns = None
        self.choose_graph_cells(arch, hidden_dim, edge_dim, num_graph_cells)
        self.mlp_decoder = nn.Linear(hidden_dim, output_features * fh)
        self.fh = fh

    def choose_graph_cells(self, arch, hidden_dim, edge_dim, num_graph_cells):
        if arch == "trans":
            self.gnns = nn.ModuleList(
                [
                    TransformerConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,  # , dropout=0.1
                    )
                    for _ in range(num_graph_cells)
                ]
            )
        elif arch == "cgc":
            self.gnns = nn.ModuleList(
                [
                    CGConv(hidden_dim, edge_dim, aggr="mean")
                    for _ in range(num_graph_cells)
                ]
            )
        elif arch == "gat":
            self.gnns = nn.ModuleList(
                [
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,
                    )
                    for _ in range(num_graph_cells)
                ]
            )
        elif arch == "gen":
            self.gnns = nn.ModuleList(
                [
                    GENConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,
                        num_layers=num_graph_cells,
                    )
                ]
            )
        elif arch == "pdn":
            self.gnns = nn.ModuleList(
                [
                    PDNConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        edge_dim=edge_dim,
                        hidden_channels=4 * edge_dim,
                    )
                    for _ in range(num_graph_cells)
                ]
            )
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, t, s):
        x = x.view(-1, x.size(-2) * x.size(-1))
        x = self.mlp_embedder(x)
        x = self.st_encoder(x, t, s)
        x = self.layer_norm_embed(x).relu()
        for gnn in self.gnns:
            x = x + gnn(x, edge_index, edge_attr).relu()  # option A
            # x = (x + gnn(x, edge_index, edge_attr)).relu()       # option B
            # x = gnn(x, edge_index, edge_attr).relu()             # option C
        x = self.mlp_decoder(x)
        return x.view(x.size(0), x.size(1) // self.fh, self.fh)
