import numpy as np


class DataProcessor:
    def __init__(self, data: np.array):
        self.data = data
        self.samples, self.latitude, self.longitude, self.features = data.shape

    def flatten(self):
        self.data = self.data.reshape(-1, self.latitude * self.longitude, self.features)

    def create_autoregressive_sequences(self, sequence_length):
        sequences = np.zeros(
            (
                self.samples - sequence_length + 1,
                sequence_length,
                self.latitude,
                self.longitude,
                self.features,
            )
        )
        for i in range(self.samples - sequence_length + 1):
            sequences[i] = self.data[i : i + sequence_length, :, :, :]
        sequences = sequences.transpose((0, 2, 3, 1, 4))
        self.data = sequences

    def preprocess(self, input_size, fh=1):
        self.create_autoregressive_sequences(sequence_length=input_size + fh)
        X = self.data[:, :, :, :input_size, :]
        y = self.data[:, :, :, -fh:, :]
        return X, y

    @staticmethod
    def train_test_split(X, y, train_split=0.8):
        train_samples = int(train_split * len(X))
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]
        return X_train, X_test, y_train, y_test

    def get_latitude_longitude(self):
        return self.latitude, self.longitude
