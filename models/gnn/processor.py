import numpy as np
import torch
import torch_geometric.data as data
import copy
import sys
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
)
from sklearn.utils import shuffle
from models.config import config as cfg

sys.path.append("..")
from models.data_processor import DataProcessor


class NNDataProcessor:
    def __init__(
        self,
        spatial_encoding=False,
        temporal_encoding=False,
        additional_encodings=False,
        test_shuffle=True,
        path=cfg.DATA_PATH,
    ):
        self.data_proc = DataProcessor(
            path=path,
            spatial_encoding=spatial_encoding,
            temporal_encoding=temporal_encoding,
            additional_encodings=additional_encodings,
        )
        self.raw_data = self.data_proc.data
        self.temporal_data = self.data_proc.temporal_data
        self.spatial_data = self.data_proc.spatial_data
        self.feature_list = self.data_proc.feature_list
        (
            self.num_samples,
            self.num_latitudes,
            self.num_longitudes,
            self.num_features,
        ) = self.raw_data.shape

        self.spatial_encoding = spatial_encoding or additional_encodings
        self.temporal_encoding = temporal_encoding or additional_encodings
        self.num_spatial_constants = self.data_proc.num_spatial_constants
        self.num_temporal_constants = self.data_proc.num_temporal_constants

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.test_shuffle = test_shuffle

        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.scaler = None
        self.scalers = None
        self.edge_weights = None
        self.edge_index = None
        self.edge_attr = None
        self.cfg = cfg

    def update(self, c):
        self.data_proc.upload_data(self.raw_data)
        self.cfg = c

    def preprocess(self, subset=None):
        self.edge_index, self.edge_weights, self.edge_attr = self.create_edges()
        X_train, X_test, y_train, y_test = self.train_val_test_split()
        X, y = self.fit_transform_scalers(
            X_train, X_test, y_train, y_test, scaler_type=self.cfg.SCALER_TYPE
        )
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(
            X, y, subset, test_shuffle=self.test_shuffle
        )

    def train_val_test_split(self):
        X, y = self.data_proc.preprocess(self.cfg.INPUT_SIZE, self.cfg.FH)

        self.num_samples = X.shape[0]
        self.train_size = int(self.num_samples * self.cfg.TRAIN_RATIO)
        self.val_size = self.train_size
        self.test_size = self.num_samples - self.train_size - self.val_size

        X = X.reshape(
            -1,
            self.num_latitudes * self.num_longitudes * self.cfg.INPUT_SIZE,
            self.num_features,
        )
        y = y.reshape(
            -1,
            self.num_latitudes * self.num_longitudes * self.cfg.FH,
            self.num_features,
        )

        return self.data_proc.train_val_test_split(X, y, split_type=3)

    def fit_transform_scalers(
        self, X_train, X_test, y_train, y_test, scaler_type="standard"
    ):
        if scaler_type == "min_max":
            self.scaler = MinMaxScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "max_abs":
            self.scaler = MaxAbsScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            print(f"{scaler_type} scaler not implemented")
            raise ValueError

        self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.num_features)]

        Xi_shape = self.num_latitudes * self.num_longitudes * self.cfg.INPUT_SIZE
        yi_shape = self.num_latitudes * self.num_longitudes * self.cfg.FH

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
                self.scalers[i]
                .transform(X_test_i)
                .reshape((self.test_size + self.val_size, Xi_shape))
            )
            y_train[..., i] = (
                self.scalers[i]
                .transform(y_train_i)
                .reshape((self.train_size, yi_shape))
            )
            y_test[..., i] = (
                self.scalers[i]
                .transform(y_test_i)
                .reshape((self.test_size + self.val_size, yi_shape))
            )

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        X = X.reshape(
            -1,
            self.num_latitudes * self.num_longitudes,
            self.cfg.INPUT_SIZE,
            self.num_features,
        )
        y = y.reshape(
            -1, self.num_latitudes * self.num_longitudes, self.cfg.FH, self.num_features
        )
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))

        return X, y

    def inverse_transform_scalers(self, X, y):
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))
        for i in range(self.num_features):
            X_i = X[..., i].reshape(-1, 1)
            y_i = y[..., i].reshape(-1, 1)
            X[..., i] = (
                self.scalers[i]
                .inverse_transform(X_i)
                .reshape(
                    (
                        X.shape[0],
                        self.num_latitudes * self.num_longitudes,
                        self.cfg.INPUT_SIZE,
                    )
                )
            )
            y[..., i] = (
                self.scalers[i]
                .inverse_transform(y_i)
                .reshape(
                    (y.shape[0], self.num_latitudes * self.num_longitudes, self.cfg.FH)
                )
            )
        X = X.reshape(
            -1,
            self.num_latitudes * self.num_longitudes,
            self.cfg.INPUT_SIZE,
            self.num_features,
        )
        y = y.reshape(
            -1, self.num_latitudes * self.num_longitudes, self.cfg.FH, self.num_features
        )
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))
        return X, y

    def create_edges(self, r=None):
        if r is None:
            r = self.cfg.R

        def node_index(i, j, num_cols):
            return i * num_cols + j

        # edge aggregation unit
        u = 0.5
        edge_index = []
        edge_attr = []
        _, indices = DataProcessor.count_neighbours(radius=r)

        for la in range(self.num_latitudes):
            for lo in range(self.num_longitudes):
                for i, j in indices:
                    if (
                        -1 < la + i < self.num_latitudes
                        and -1 < lo + j < self.num_longitudes
                    ):
                        edge_index.append(
                            [
                                node_index(la, lo, self.num_longitudes),
                                node_index(la + i, lo + j, self.num_longitudes),
                            ]
                        )
                        edge_attr.append(
                            [u * i, u * j, np.sqrt((u * i) ** 2 + (u * j) ** 2)]
                        )

        edge_index = torch.tensor(edge_index, dtype=torch.int64).t().to(self.cfg.DEVICE)
        edge_weights = None
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.cfg.DEVICE)

        return edge_index, edge_weights, edge_attr

    def get_loaders(self, X, y, subset=None, test_shuffle=True):
        dataset = []
        for i in range(X.shape[0]):
            Xi = torch.from_numpy(X[i].astype("float32")).to(self.cfg.DEVICE)
            yi = torch.from_numpy(y[i].astype("float32")).to(self.cfg.DEVICE)
            if self.temporal_encoding:
                ti = torch.from_numpy(self.temporal_data[i].astype("float32")).to(
                    self.cfg.DEVICE
                )
            else:
                ti = None
            if self.spatial_encoding:
                si = torch.from_numpy(self.spatial_data.astype("float32")).to(
                    self.cfg.DEVICE
                )
            else:
                si = None
            g = data.Data(
                x=Xi,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=yi,
                pos=si,
                time=ti,
            )
            g = g.to(self.cfg.DEVICE)
            dataset.append(g)

        train_dataset = dataset[: self.train_size]
        val_dataset = dataset[self.train_size : self.train_size + self.val_size]
        test_dataset = dataset[-self.test_size :]
        # random state for reproduction
        train_dataset = shuffle(train_dataset, random_state=self.cfg.RANDOM_STATE)
        val_dataset = shuffle(val_dataset, random_state=self.cfg.RANDOM_STATE)
        if test_shuffle:
            test_dataset = shuffle(test_dataset, random_state=self.cfg.RANDOM_STATE)

        if subset is not None:
            train_dataset = train_dataset[: subset * self.cfg.BATCH_SIZE]
            val_dataset = val_dataset[: subset * self.cfg.BATCH_SIZE]
            test_dataset = test_dataset[: subset * self.cfg.BATCH_SIZE]

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.BATCH_SIZE)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.BATCH_SIZE)
        return train_loader, val_loader, test_loader

    def get_shapes(self):
        return (
            self.num_samples,
            self.num_latitudes,
            self.num_longitudes,
            self.num_features,
        )

    def map_latitude_longitude_span(
        self, input_tensor, old_span=(32, 48), new_span=(25, 45), flat=True
    ):
        """
        Maps latitude-longitude span e.g. (32,48) -> (25,45)
        """
        if flat:
            batch_size = int(
                input_tensor.shape[0] / self.num_latitudes / self.num_longitudes
            )
            input_tensor = input_tensor.reshape(
                (
                    batch_size,
                    self.num_latitudes,
                    self.num_longitudes,
                    self.num_features,
                    input_tensor.shape[-1],
                )
            )

        old_lat, old_lon = old_span
        new_lat, new_lon = new_span

        lat_diff = old_lat - new_lat
        left_lat = lat_diff // 2
        right_lat = new_lat + lat_diff - left_lat - 1

        lon_diff = old_lon - new_lon

        mapped_tensor = input_tensor[:, left_lat:right_lat, lon_diff:]
        mapped_tensor.reshape(-1, self.num_features, mapped_tensor.shape[4])

        return mapped_tensor
