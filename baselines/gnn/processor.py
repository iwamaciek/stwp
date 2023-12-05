import numpy as np
import torch
import torch_geometric.data as data
import copy
import sys
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import shuffle
from baselines.config import (
    DEVICE,
    FH,
    INPUT_SIZE,
    TRAIN_RATIO,
    BATCH_SIZE,
)

sys.path.append("..")
from baselines.data_processor import DataProcessor


class NNDataProcessor:
    def __init__(self):
        self.dataset, self.feature_list = DataProcessor.load_data(
            spatial_encoding=False
        )
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
        self.edge_attr = None

    def preprocess(self, subset=None):
        self.edge_index, self.edge_weights, self.edge_attr = self.create_edges()
        X_train, X_test, y_train, y_test = self.train_test_split()
        X, y = self.get_scalers(X_train, X_test, y_train, y_test)
        self.train_loader, self.test_loader = self.get_loaders(X, y, subset)

    def train_test_split(self):
        processor = DataProcessor(self.dataset)
        X, y = processor.preprocess(INPUT_SIZE, FH)

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

        scaler = StandardScaler()
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

        # edge aggregation unit
        u = 0.5
        edge_index = []
        edge_weights = []
        edge_attr = []
        for i in range(self.num_latitudes):
            for j in range(self.num_longitudes):

                # Up
                if i > 0:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i - 1, j, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u)
                    edge_attr.append([u, 0, u])

                # Left
                if j > 0:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i, j - 1, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u)
                    edge_attr.append([0, -u, u])

                # Down
                if i < self.num_latitudes - 1:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i + 1, j, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u)
                    edge_attr.append([-u, 0, u])

                # Right
                if j < self.num_longitudes - 1:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i, j + 1, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u)
                    edge_attr.append([0, u, u])

                # Up-Left
                if i > 0 and j > 0:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i - 1, j - 1, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u / np.sqrt(2 * u))
                    edge_attr.append([u, -u, np.sqrt(2 * u)])

                # Up-Right
                if i > 0 and j < self.num_longitudes - 1:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i - 1, j + 1, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u / np.sqrt(2 * u))
                    edge_attr.append([u, u, np.sqrt(2 * u)])

                # Down-Left
                if i < self.num_latitudes - 1 and j > 0:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i + 1, j - 1, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u / np.sqrt(2 * u))
                    edge_attr.append([-u, -u, np.sqrt(2 * u)])

                # Down-Right
                if i < self.num_latitudes - 1 and j < self.num_longitudes - 1:
                    edge_index.append(
                        [
                            node_index(i, j, self.num_longitudes),
                            node_index(i + 1, j + 1, self.num_longitudes),
                        ]
                    )
                    edge_weights.append(u / np.sqrt(2 * u))
                    edge_attr.append([-u, u, np.sqrt(2 * u)])

        edge_index = torch.tensor(edge_index, dtype=torch.int64).t().to(DEVICE)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32).to(DEVICE)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(DEVICE)  # (e, 2)

        return edge_index, edge_weights, edge_attr

    def get_loaders(self, X, y, subset=None):
        dataset = []
        for i in range(X.shape[0]):
            Xi = torch.from_numpy(X[i].astype("float32")).to(DEVICE)
            yi = torch.from_numpy(y[i].astype("float32")).to(DEVICE)
            g = data.Data(
                x=Xi, edge_index=self.edge_index, edge_attr=self.edge_attr, y=yi
            )
            g = g.to(DEVICE)
            dataset.append(g)

        train_dataset = dataset[: self.train_size]
        test_dataset = dataset[-self.test_size :]
        # random state for reproduction
        train_dataset = shuffle(train_dataset, random_state=42)
        test_dataset = shuffle(test_dataset, random_state=42)

        if subset is not None:
            train_dataset = train_dataset[: subset * BATCH_SIZE]
            test_dataset = test_dataset[: subset * BATCH_SIZE]

        # if subset is None:
        #     train_sampler, test_sampler = None, None
        # else:
        #     train_sampler = RandomSampler(train_dataset, num_samples=subset)
        #     test_sampler = RandomSampler(test_dataset, num_samples=subset)

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
