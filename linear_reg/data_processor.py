import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, data: np.array):
        self.data = data
        _, self.latitude, self.longitude = data.shape

    def flatten(self):
        self.data = self.data.reshape(-1, self.latitude * self.longitude)

    def create_autoregressive_sequences(self, sequence_length):
        samples = self.data.shape[0]
        features = self.longitude * self.latitude
        sequences = np.zeros((samples - sequence_length + 1, sequence_length, features))
        for i in range(samples - sequence_length + 1):
            sequences[i] = self.data[i : i + sequence_length, :]
        self.data = sequences

    def preprocess(self, sequence_length):
        self.flatten()
        self.create_autoregressive_sequences(sequence_length=sequence_length)
        X = self.data[:, : sequence_length - 1, :]
        X = X.transpose((0, 2, 1))
        y = self.data[:, -1, :]
        return X, y

    def train_test_split(self, X, y, train_percentage=0.8):
        train_samples = int(train_percentage * len(X))
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]
        return X_train, X_test, y_train, y_test

    def get_latitude_longitude(self):
        return self.latitude, self.longitude
