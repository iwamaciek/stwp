import cfgrib
import numpy as np
import torch
import torch_geometric.data as data
import copy
import sys
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import MinMaxScaler
from baselines.gnn.config import DEVICE, FH, INPUT_SIZE, TRAIN_RATIO, DATAPATH

sys.path.append("..")
from baselines.data_processor import DataProcessor


def preprocess():
    grib_data = cfgrib.open_datasets(DATAPATH)
    surface = grib_data[0]
    hybrid = grib_data[1]
    t2m_numpy = surface.t2m.to_numpy()
    sp_numpy = surface.sp.to_numpy()
    dataset = np.stack((t2m_numpy, sp_numpy), axis=-1)
    _, num_latitudes, num_longitudes, num_features = dataset.shape

    processor = DataProcessor(dataset)
    X, y = processor.preprocess(INPUT_SIZE)

    num_samples = X.shape[0]
    train_size = int(num_samples * TRAIN_RATIO)
    val_size = num_samples - train_size
    X = X.reshape(-1, num_latitudes * num_longitudes * INPUT_SIZE, num_features)
    y = y.reshape(-1, num_latitudes * num_longitudes * FH, num_features)

    X_train, X_test = X[:train_size], X[-val_size:]
    y_train, y_test = y[:train_size], y[-val_size:]

    scaler = MinMaxScaler()
    scalers = [copy.deepcopy(scaler) for _ in range(num_features)]

    Xi_shape = num_latitudes * num_longitudes * INPUT_SIZE
    yi_shape = num_latitudes * num_longitudes * FH

    for i in range(num_features):
        X_train_i = X_train[..., i].reshape(-1, 1)
        X_test_i = X_test[..., i].reshape(-1, 1)
        y_train_i = y_train[..., i].reshape(-1, 1)
        y_test_i = y_test[..., i].reshape(-1, 1)

        scalers[i].fit(X_train_i)
        X_train[..., i] = (
            scalers[i].transform(X_train_i).reshape((train_size, Xi_shape))
        )
        X_test[..., i] = scalers[i].transform(X_test_i).reshape((val_size, Xi_shape))
        y_train[..., i] = (
            scalers[i].transform(y_train_i).reshape((train_size, yi_shape))
        )
        y_test[..., i] = scalers[i].transform(y_test_i).reshape((val_size, yi_shape))

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    X = X.reshape(-1, num_latitudes * num_longitudes, INPUT_SIZE, num_features)
    y = y.reshape(-1, num_latitudes * num_longitudes, FH, num_features)
    X = X.transpose((0, 1, 3, 2))
    y = y.transpose((0, 1, 3, 2))

    def node_index(i, j, num_cols):
        return i * num_cols + j

    edge_index = []
    for i in range(num_latitudes):
        for j in range(num_longitudes):
            if i > 0:
                edge_index.append(
                    [
                        node_index(i, j, num_longitudes),
                        node_index(i - 1, j, num_longitudes),
                    ]
                )
            if j > 0:
                edge_index.append(
                    [
                        node_index(i, j, num_longitudes),
                        node_index(i, j - 1, num_longitudes),
                    ]
                )

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_index = to_undirected(edge_index)

    dataset = []
    for i in range(X.shape[0]):
        Xi = torch.from_numpy(X[i].astype("float32")).to(DEVICE)
        yi = torch.from_numpy(y[i].astype("float32")).to(DEVICE)
        g = data.Data(x=Xi, edge_index=edge_index, y=yi)
        g = g.to(DEVICE)
        dataset.append(g)

    shapes = (num_samples, num_latitudes, num_longitudes, num_features)
    return dataset, scalers, shapes
