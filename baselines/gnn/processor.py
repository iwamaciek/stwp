import cfgrib
import numpy as np
import torch
import torch_geometric.data as data
import copy
import sys
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
from baselines.gnn.config import (
    DEVICE,
    FH,
    INPUT_SIZE,
    TRAIN_RATIO,
    DATA_PATH,
    BATCH_SIZE,
)

sys.path.append("..")
from baselines.data_processor import DataProcessor


class NNDataProcessor:
    def __init__(self):
        self.dataset = self.load_data()
        (
            self.num_samples,
            self.num_latitudes,
            self.num_longitudes,
            self.num_features,
        ) = self.dataset.shape

        self.train_loader = None
        self.test_loader = None

        self.train_size = None
        self.test_size = None
        self.scalers = None
        self.edge_weights = None
        self.edge_index = None

    def preprocess(self, subset=None):
        self.edge_index, self.edge_weights = self.create_edges()
        X_train, X_test, y_train, y_test = self.train_test_split()
        X, y = self.get_scalers(X_train, X_test, y_train, y_test)
        self.train_loader, self.test_loader = self.get_loaders(X, y, subset)

    @staticmethod
    def load_data(data=DATA_PATH):
        grib_data = cfgrib.open_datasets(data)
        surface = grib_data[0]
        hybrid = grib_data[1]
        t2m = surface.t2m.to_numpy() - 273.15  # -> C
        sp = surface.sp.to_numpy() / 100  # -> hPa
        tcc = surface.tcc.to_numpy()
        u10 = surface.u10.to_numpy()
        v10 = surface.v10.to_numpy()
        tp = hybrid.tp.to_numpy()
        if(tp.ndim >= 4):
            tp = tp.reshape((-1,) + hybrid.tp.shape[2:])
        return np.stack((t2m, sp, tcc, u10, v10, tp), axis=-1)

    def train_test_split(self):
        processor = DataProcessor(self.dataset)
        X, y = processor.preprocess(INPUT_SIZE)

        self.num_samples = X.shape[0]
        self.train_size = int(self.num_samples * TRAIN_RATIO)
        self.test_size = self.num_samples - self.train_size

        X = X.reshape(
            -1, self.num_latitudes * self.num_longitudes * INPUT_SIZE, self.num_features
        )
        y = y.reshape(
            -1, self.num_latitudes * self.num_longitudes * FH, self.num_features
        )

        X_train, X_test = X[: self.train_size], X[-self.test_size :]
        y_train, y_test = y[: self.train_size], y[-self.test_size :]

        return X_train, X_test, y_train, y_test

    def get_scalers(self, X_train, X_test, y_train, y_test):
        self.train_size = len(X_train)
        self.test_size = len(X_test)

        scaler = MinMaxScaler()
        self.scalers = [copy.deepcopy(scaler) for _ in range(self.num_features)]

        Xi_shape = self.num_latitudes * self.num_longitudes * INPUT_SIZE
        yi_shape = self.num_latitudes * self.num_longitudes * FH

        for i in range(self.num_features):
            X_train_i = X_train[..., i].reshape(-1, 1)
            X_test_i = X_test[..., i].reshape(-1, 1)
            y_train_i = y_train[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)

            self.scalers[i].fit(X_train_i)
            X_train[..., i] = (
                self.scalers[i]
                .transform(X_train_i)
                .reshape((self.train_size, Xi_shape))
            )
            X_test[..., i] = (
                self.scalers[i].transform(X_test_i).reshape((self.test_size, Xi_shape))
            )
            y_train[..., i] = (
                self.scalers[i]
                .transform(y_train_i)
                .reshape((self.train_size, yi_shape))
            )
            y_test[..., i] = (
                self.scalers[i].transform(y_test_i).reshape((self.test_size, yi_shape))
            )

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        X = X.reshape(
            -1, self.num_latitudes * self.num_longitudes, INPUT_SIZE, self.num_features
        )
        y = y.reshape(
            -1, self.num_latitudes * self.num_longitudes, FH, self.num_features
        )
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))

        return X, y

    def create_edges(self):
        def node_index(i, j, num_cols):
            return i * num_cols + j

        edge_index = []
        edge_weights = []
        for i in range(self.num_latitudes):
            for j in range(self.num_longitudes):
                if i > 0:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i - 1, j, self.num_longitudes),
                        ]
                    )
                    edge_index.append(
                        [
                            node_index(i - 1, j, self.num_longitudes),
                            node_index(i, j, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(0.5)
                    edge_weights.append(-0.5)

                if j > 0:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i, j - 1, self.num_longitudes),
                        ]
                    )
                    edge_index.append(
                        [
                            node_index(i, j - 1, self.num_longitudes),
                            node_index(i, j, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(0.5)
                    edge_weights.append(-0.5)

        edge_index = torch.tensor(edge_index, dtype=torch.int).t().to(DEVICE)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32).to(DEVICE)

        return edge_index, edge_weights

    def get_loaders(self, X, y, subset=None):
        dataset = []
        for i in range(X.shape[0]):
            Xi = torch.from_numpy(X[i].astype("float32")).to(DEVICE)
            yi = torch.from_numpy(y[i].astype("float32")).to(DEVICE)
            g = data.Data(
                x=Xi, edge_index=self.edge_index, edge_attr=self.edge_weights, y=yi
            )
            g = g.to(DEVICE)
            dataset.append(g)

        if subset is None:
            train_dataset = dataset[: self.train_size]
            test_dataset = dataset[-self.test_size :]
        else:
            train_dataset = dataset[: subset * BATCH_SIZE]
            test_dataset = dataset[subset * BATCH_SIZE : subset * BATCH_SIZE * 2]

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        return train_loader, test_loader

    def get_shapes(self):
        return (
            self.num_samples,
            self.num_latitudes,
            self.num_longitudes,
            self.num_features,
        )
