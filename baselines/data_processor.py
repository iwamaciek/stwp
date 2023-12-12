#!/usr/bin/env python3
import cfgrib
import numpy as np
from datetime import datetime
from utils.datetime_operations import datetime64_to_datetime, get_day_of_year
from utils.trig_encode import trig_encode
from sklearn.utils import shuffle
from baselines.config import DATA_PATH, TRAIN_RATIO, INPUT_SIZE, FH, R, RANDOM_STATE
from utils.get_data import BIG_AREA, SMALL_AREA


class DataProcessor:
    def __init__(
        self,
        spatial_encoding=False,
        temporal_encoding=False,
        additional_encodings=False,
    ):
        self.data, self.feature_list = self.load_data(
            spatial_encoding=(spatial_encoding or additional_encodings),
            temporal_encoding=(temporal_encoding or additional_encodings),
        )
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
    def load_data(path=DATA_PATH, spatial_encoding=False, temporal_encoding=False):
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

            lsm = surface.lsm.to_numpy()
            z = surface.z.to_numpy()
            z = (z - z.mean()) / z.std()
            data = np.concatenate((data, np.stack((lsm, z), axis=-1)), axis=-1)

            spatial_encodings = np.empty(data.shape[:-1] + (4,))

            latitudes = np.array(surface.latitude)
            longitudes = np.array(surface.longitude)
            for i, lat in enumerate(latitudes):
                for j, lon in enumerate(longitudes):
                    for idx, v in enumerate(
                        [
                            trig_encode(lat, 180, "sin"),
                            trig_encode(lat, 180, "cos"),
                            trig_encode(lon, 360, "sin"),
                            trig_encode(lon, 360, "cos"),
                        ]
                    ):
                        spatial_encodings[:, i, j, idx] = np.repeat(v, data.shape[0])

            data = np.concatenate((data, spatial_encodings), axis=-1)

        if temporal_encoding:

            dt = surface.time.to_numpy()
            dt = np.fromiter((datetime64_to_datetime(ti) for ti in dt), dtype=datetime)

            temporal_encodings = np.empty(data.shape[:-1] + (4,))

            for t in range(data.shape[0]):
                for i in range(data.shape[1]):
                    for j in range(data.shape[2]):
                        for idx, v in enumerate(
                            [
                                trig_encode(get_day_of_year(dt[t]), 365, "sin"),
                                trig_encode(get_day_of_year(dt[t]), 365, "cos"),
                                trig_encode(dt[t].hour, 24, "sin"),
                                trig_encode(dt[t].hour, 24, "cos"),
                            ]
                        ):
                            temporal_encodings[t, i, j, idx] = v

            data = np.concatenate((data, temporal_encodings), axis=-1)

        return data, feature_list

    @staticmethod
    def train_test_split(X, y, split_ratio=TRAIN_RATIO):
        return DataProcessor.train_val_test_split(X, y, split_ratio)

    @staticmethod
    def train_val_test_split(X, y, split_ratio=TRAIN_RATIO, split_type=1):
        """
        split_type=0: X_train (2020), X_val (2021), X_test (2022)
        split_type=1: X_train (2020), X_test (2021)
        split_type=2: X_train (2020), X_test (2022)
        split_type=3: X_train (2020), X_test (2021-2022)
        """
        # randomness might influence the score !!!
        train_samples = int(split_ratio * len(X))

        if split_type == 0:
            X_train, X_val, X_test = (
                X[:train_samples],
                X[train_samples : 2 * train_samples],
                X[2 * train_samples :],
            )
            y_train, y_val, y_test = (
                y[:train_samples],
                y[train_samples : 2 * train_samples],
                y[2 * train_samples :],
            )
            X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
            X_val, y_val = shuffle(X_val, y_val, random_state=RANDOM_STATE)
            X_test, y_test = shuffle(X_test, y_test, random_state=RANDOM_STATE)

            return X_train, X_val, X_test, y_train, y_val, y_test

        elif split_type == 1:
            X_train, X_test = X[:train_samples], X[train_samples : 2 * train_samples]
            y_train, y_test = y[:train_samples], y[train_samples : 2 * train_samples]

        elif split_type == 2:
            X_train, X_test = X[:train_samples], X[2 * train_samples :]
            y_train, y_test = y[:train_samples], y[2 * train_samples :]

        else:
            X_train, X_test = X[:train_samples], X[train_samples:]
            y_train, y_test = y[:train_samples], y[train_samples:]

        X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
        X_test, y_test = shuffle(X_test, y_test, random_state=RANDOM_STATE)

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

    @staticmethod
    def get_spatial_info():
        res = 0.25
        north, west, south, east = SMALL_AREA
        spatial_limits = [west, east, south, north]
        we_span_1d = np.arange(west, east + res, res)
        ns_span_1d = np.arange(north, south - res, -res)
        lon_span = np.array([we_span_1d for _ in range(len(ns_span_1d))])
        lat_span = np.array([ns_span_1d for _ in range(len(we_span_1d))]).T
        return lat_span, lon_span, spatial_limits
