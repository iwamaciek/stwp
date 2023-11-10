import numpy as np


class DataProcessor:
    def __init__(self, data: np.array):
        self.data = data
        self.samples, self.latitude, self.longitude, self.features = data.shape
        self.neighbours, self.input_size = None, None

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
        # TODO make it more efficient:
        # shifted_data = np.roll(self.data, (0, ii, ij, 0, 0), axis=(0, 1, 2, 3, 4))
        # mask = (ii > 0) & (ii < self.latitude) & (ij > 0) & (ij < self.longitude)
        # mask = mask[..., np.newaxis, np.newaxis]
        # neigh_data[:, :, :, i, :, :] = shifted_data * mask

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
    def train_test_split(X, y, train_split=0.8):
        train_samples = int(train_split * len(X))
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]
        return X_train, X_test, y_train, y_test

    def get_latitude_longitude(self):
        return self.latitude, self.longitude
