import cfgrib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from random import sample
from torch.utils.data import DataLoader
import copy
from baselines.cnn.config import (
    DEVICE,
    FH,
    INPUT_SIZE,
    TRAIN_RATIO,
    DATA_PATH,
    BATCH_SIZE,
)

sys.path.append("..")
from baselines.data_processor import DataProcessor

class NNDataProcessor():
    def __init__(self) -> None:
        self.dataset = self.load_data()

        self.samples, self.latitudes, self.longitudes, self.features = self.dataset.shape

        self.train_loader = None
        self.test_loader = None

        self.train_size = None
        self.test_size = None
        self.scalers = None

    def preprocess(self, subset=None):
        X, y = self.get_scalers()
        X_train, X_test, y_train, y_test = self.train_test_split()
        self.train_loader, self.test_loader = self.get_loaders(X_train, y_train, X_test, y_test, subset)

    @staticmethod
    def load_data():
        grib_data = cfgrib.open_datasets(DATA_PATH)
        surface = grib_data[0]
        hybrid = grib_data[1]
        t2m = surface.t2m.to_numpy() - 273.15  # -> C
        sp = surface.sp.to_numpy() / 100  # -> hPa
        tcc = surface.tcc.to_numpy()
        u10 = surface.u10.to_numpy()
        v10 = surface.v10.to_numpy()
        tp = hybrid.tp.to_numpy().reshape((-1,) + hybrid.tp.shape[2:])
        return np.stack((t2m, sp, tcc, u10, v10, tp), axis=-1)
    
    def train_test_split(self):
        processor = DataProcessor(self.dataset)
        X, y = processor.preprocess(INPUT_SIZE, FH)

        self.num_samples = X.shape[0]
        self.train_size = int(self.num_samples * TRAIN_RATIO)
        self.test_size = self.num_samples - self.train_size

        indices = sample(range(self.num_samples), self.train_size)
        X_train, y_train = X[indices], y[indices]
        X_test = np.delete(X, indices, axis=0).reshape((self.num_samples-self.train_size,)+X_train.shape[1:])
        y_test = np.delete(y, indices, axis=0).reshape((self.num_samples-self.train_size,)+y_train.shape[1:])

        return X_train, X_test, y_train, y_test
    
    def get_scalers(self):
        self.scalers = []
        self.num_samples = self.dataset.shape[0]
        self.train_size = int(self.num_samples * TRAIN_RATIO)
        self.test_size = self.num_samples - self.train_size

        for i in range(self.features):
            scaler = MinMaxScaler()
            og_shape = self.dataset[..., i].shape
            self.dataset[..., i] = scaler.fit_transform(self.dataset[..., i].reshape((-1, 1))).reshape(og_shape)
            self.scalers.append(scaler)
        
        processor = DataProcessor(self.dataset)
        X, y = processor.preprocess(INPUT_SIZE, fh=FH)

        return X, y
    
    def get_loaders(self, X_train, y_train, X_test, y_test, subset=None):
        train = np.concatenate([X_train, y_train], axis=3)
        test = np.concatenate([X_test, y_test], axis=3)

        train = train.transpose(0, 3, 4, 1, 2)
        test = test.transpose(0, 3, 4, 1, 2)

        train = np.float32(train)
        test = np.float32(test)

        if subset is not None:
            train = train[: subset * BATCH_SIZE]
            test = test[subset * BATCH_SIZE : subset * BATCH_SIZE * 2]

        train_loader = DataLoader(train, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE)
        return train_loader, test_loader
    
    def get_shapes(self):
        return (
            self.samples,
            self.latitudes,
            self.longitudes,
            self.features,
        )