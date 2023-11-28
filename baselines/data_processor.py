#!/usr/bin/env python3
import cfgrib
import numpy as np
from baselines.config import DATA_PATH, TRAIN_RATIO
from random import sample


class DataProcessor:
    def __init__(self, data: np.array):
        self.data = data
        self.samples, self.latitude, self.longitude, self.features = data.shape
        self.neighbours, self.input_size = None, None

    @staticmethod
    def load_data(path=DATA_PATH):
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
        feature_list = ["t2m", "tcc", "u10", "v10", "tp", "sp"]
        return data, feature_list

    def flatten(self):
        self.data = self.data.reshape(-1, self.latitude * self.longitude, self.features)

    def create_autoregressive_sequences(self, sequence_length):
        self.input_size = sequence_length
        sequences = np.empty(
            (
                self.samples - self.input_size + 1,
                self.input_size,
                self.latitude,
                self.longitude,
                self.features,
            )
        )
        for i in range(self.samples - sequence_length + 1):
            sequences[i] = self.data[i : i + sequence_length]
        sequences = sequences.transpose((0, 2, 3, 1, 4))
        self.samples = sequences.shape[0]
        self.data = sequences

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

    def create_neighbours(self, radius):
        # TODO make it more efficient
        self.neighbours, indices = self.count_neighbours(radius=radius)
        neigh_data = np.empty(
            (
                self.samples,
                self.latitude,
                self.longitude,
                self.neighbours + 1,
                self.input_size,
                self.features,
            )
        )
        neigh_data[..., 0, :, :] = self.data

        for n in range(1, self.neighbours + 1):
            i, j = indices[n - 1]
            for s in range(self.samples):
                for la in range(self.latitude):
                    for lo in range(self.longitude):
                        if 0 < la + i < self.latitude and 0 < lo + j < self.longitude:
                            neigh_data[s, la, lo, n] = self.data[s, la + i, lo + j]
                        else:
                            neigh_data[s, la, lo, n] = self.data[s, la, lo]

        self.data = neigh_data

    def preprocess(self, input_size, fh=1, r=1, use_neighbours=False):
        self.create_autoregressive_sequences(sequence_length=input_size + fh)
        if use_neighbours:
            self.create_neighbours(radius=r)
            y = self.data[..., 0, -fh:, :]
        else:
            y = self.data[..., -fh:, :]
        X = self.data[..., :input_size, :]
        return X, y

    @staticmethod
    def train_test_split(X, y, train_split=TRAIN_RATIO):
        n = len(X)
        train_samples = int(train_split * len(X))
        indices = sample(range(len(X)), train_samples)
        X_train, y_train = X[indices], y[indices]
        X_test = np.delete(X, indices, axis=0).reshape((n-train_samples,)+X_train.shape[1:])
        y_test = np.delete(y, indices, axis=0).reshape((n-train_samples,)+y_train.shape[1:])
        # X_train, X_test = X[:train_samples], X[train_samples:]
        # y_train, y_test = y[:train_samples], y[train_samples:]
        return X_train, X_test, y_train, y_test

    def get_latitude_longitude(self):
        return self.latitude, self.longitude
