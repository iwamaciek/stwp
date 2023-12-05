#!/usr/bin/env python3
import cfgrib
import numpy as np
from baselines.config import DATA_PATH, TRAIN_RATIO, INPUT_SIZE, FH, R


class DataProcessor:
    def __init__(self, spatial_encoding=False):
        self.data, self.feature_list = self.load_data(spatial_encoding=spatial_encoding)
        self.samples, self.latitude, self.longitude, self.num_features = self.data.shape
        self.num_spatial_constants = self.num_features - len(self.feature_list)
        self.num_features = self.num_features - self.num_spatial_constants
        self.neighbours, self.input_size = None, None

    def upload_data(self, data: np.array):
        self.data = data

    def create_autoregressive_sequences(self, sequence_length=INPUT_SIZE + FH):
        self.input_size = sequence_length
        sequences = np.empty(
            (
                self.samples - self.input_size + 1,
                self.input_size,
                self.latitude,
                self.longitude,
                self.num_features + self.num_spatial_constants,
            )
        )
        for i in range(self.samples - sequence_length + 1):
            sequences[i] = self.data[i : i + sequence_length]
        sequences = sequences.transpose((0, 2, 3, 1, 4))
        self.samples = sequences.shape[0]
        self.data = sequences

    def create_neighbours(self, radius):
        self.neighbours, indices = self.count_neighbours(radius=radius)
        neigh_data = np.empty(
            (
                self.samples,
                self.latitude,
                self.longitude,
                self.neighbours + 1,
                self.input_size,
                self.num_features + self.num_spatial_constants,
            )
        )
        neigh_data[..., 0, :, :] = self.data

        for n in range(1, self.neighbours + 1):
            i, j = indices[n - 1]
            for s in range(self.samples):
                for la in range(self.latitude):
                    for lo in range(self.longitude):
                        if -1 < la + i < self.latitude and -1 < lo + j < self.longitude:
                            neigh_data[s, la, lo, n] = self.data[s, la + i, lo + j]
                        else:
                            neigh_data[s, la, lo, n] = self.data[s, la, lo]

        self.data = neigh_data

    def preprocess(self, input_size=INPUT_SIZE, fh=FH, r=R, use_neighbours=False):
        self.create_autoregressive_sequences(sequence_length=input_size + fh)
        if use_neighbours:
            self.create_neighbours(radius=r)
            y = self.data[..., 0, -fh:, : self.num_features]
        else:
            y = self.data[..., -fh:, : self.num_features]
        X = self.data[..., :input_size, :]
        return X, y

    @staticmethod
    def load_data(path=DATA_PATH, spatial_encoding=False):
        grib_data = cfgrib.open_datasets(path)
        surface = grib_data[0]
        hybrid = grib_data[1]
        t2m = surface.t2m.to_numpy() - 273.15  # -> C
        sp = surface.sp.to_numpy() / 100  # -> hPa
        tcc = surface.tcc.to_numpy()
        u10 = surface.u10.to_numpy()
        v10 = surface.v10.to_numpy()
        tp = hybrid.tp.to_numpy()
        if tp.ndim >= 4:
            tp = tp.reshape((-1,) + hybrid.tp.shape[2:])
        data = np.stack((t2m, sp, tcc, u10, v10, tp), axis=-1)
        feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]

        if spatial_encoding:

            def spatial_encode(v, norm_v, trig_func="sin"):
                if trig_func == "sin":
                    v_encoded = np.sin(2 * np.pi * v / norm_v)
                elif trig_func == "cos":
                    v_encoded = np.cos(2 * np.pi * v / norm_v)
                else:
                    print("Function not implemented")
                    return None
                return v_encoded

            spatial_encodings = np.empty(data.shape[:-1] + (4,))

            latitudes = np.array(surface.latitude)
            longitudes = np.array(surface.longitude)
            for i, lat in enumerate(latitudes):
                for j, lon in enumerate(longitudes):
                    for idx, v in enumerate(
                        [
                            spatial_encode(lat, 180, "sin"),
                            spatial_encode(lat, 180, "cos"),
                            spatial_encode(lon, 360, "sin"),
                            spatial_encode(lon, 360, "cos"),
                        ]
                    ):
                        spatial_encodings[:, i, j, idx] = np.repeat(v, data.shape[0])

            data = np.concatenate((data, spatial_encodings), axis=-1)

        return data, feature_list

    @staticmethod
    def train_test_split(X, y, train_split=TRAIN_RATIO):
        train_samples = int(train_split * len(X))
        # randomness might influence the score !!!
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]
        return X_train, X_test, y_train, y_test

    @staticmethod
    def count_neighbours(radius: int):
        count, indices = 0, []
        if radius < 0:
            return count, indices

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x == 0 and y == 0:
                    continue
                distance = (x**2 + y**2) ** 0.5
                if distance <= radius:
                    count += 1
                    indices.append((x, y))
        return count, indices
