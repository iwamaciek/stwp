#!/usr/bin/env python3
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor
from baselines.data_processor import DataProcessor
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
)


class BaselineRegressor:
    def __init__(
        self, X_shape: tuple, fh: int, feature_list: list, scaler_type="robust"
    ):
        if len(X_shape) > 5:
            (
                _,
                self.latitude,
                self.longitude,
                self.neighbours,
                self.input_state,
                self.features,
            ) = X_shape
        else:
            (
                _,
                self.latitude,
                self.longitude,
                self.input_state,
                self.features,
            ) = X_shape
            self.neighbours = 1

        self.fh = fh
        self.feature_list = feature_list

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

        self.model = DummyRegressor()
        self.models = [copy.deepcopy(self.model) for _ in range(self.features)]
        self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.features)]

    def train(self, X_train, y_train, normalize=False):
        X = X_train.reshape(-1, self.neighbours * self.input_state * self.features)
        for i in range(self.features):
            yi = y_train[..., 0, i].reshape(-1, 1)
            if normalize:
                self.scalers[i].fit(yi)
            self.models[i].fit(X, yi)

    def get_rmse(self, y_hat, y_test, normalize=False):
        rmse_features = []
        for i in range(self.features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            if normalize:
                y_test_i = self.scalers[i].transform(y_test_i)
                y_hat_i = self.scalers[i].transform(y_hat_i)
            err = round(np.sqrt(mean_squared_error(y_hat_i, y_test_i)), 3)
            rmse_features.append(err)
        return rmse_features

    def get_mae(self, y_hat, y_test, normalize=False):
        mae_features = []
        for i in range(self.features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            if normalize:
                y_test_i = self.scalers[i].transform(y_test_i)
                y_hat_i = self.scalers[i].transform(y_hat_i)
            # err = round(np.sqrt(mean_squared_error(y_hat_i, y_test_i)), 3)
            err = round(mean_absolute_error(y_hat_i, y_test_i), 3)
            mae_features.append(err)
        return mae_features

    def evaluate(self, y_hat, y_test):
        return self.get_rmse(y_hat, y_test)

    def plot_predictions(self, y_hat, y_test, max_samples):
        for i in range(max_samples):
            y_test_sample, y_hat_sample = y_test[i], y_hat[i]
            fig, ax = plt.subplots(
                self.features, 3 * self.fh, figsize=(10 * self.fh, 3 * self.features)
            )

            for j in range(self.features):
                cur_feature = self.feature_list[j]
                y_test_sample_feature_j = y_test_sample[..., j].reshape(-1, 1)
                y_hat_sample_feature_j = y_hat_sample[..., j].reshape(-1, 1)
                mse = mean_squared_error(
                    y_test_sample_feature_j, y_hat_sample_feature_j
                )
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(
                    y_test_sample_feature_j, y_hat_sample_feature_j
                )
                std = np.std(y_test_sample_feature_j)
                sqrt_n = np.sqrt(y_test_sample_feature_j.shape[0])
                print(
                    f"{cur_feature} => RMSE:  {round(rmse,5)}; MAE: {mae}; SE: {std / sqrt_n}"
                )

                for k in range(3 * self.fh):
                    ts = k // 3
                    if k % 3 == 0:
                        title = rf"$X_{{{cur_feature},t+{ts+1}}}$"
                        value = y_test[i, ..., ts, j]
                        cmap = plt.cm.coolwarm
                    elif k % 3 == 1:
                        title = rf"$\hat{{X}}_{{{cur_feature},t+{ts+1}}}$"
                        value = y_hat[i, ..., ts, j]
                        cmap = plt.cm.coolwarm
                    else:
                        title = rf"$|X - \hat{{X}}|_{{{cur_feature},t+{ts+1}}}$"
                        value = np.abs(y_test[i, ..., ts, j] - y_hat[i, ..., ts, j])
                        cmap = "binary"
                    pl = ax[j, k].imshow(value, cmap=cmap)
                    ax[j, k].set_title(title)
                    ax[j, k].axis("off")
                    _ = fig.colorbar(pl, ax=ax[j, k], fraction=0.15)
            plt.show()

    def predict_(self, X_test, y_test):
        X = X_test.reshape(-1, self.neighbours * self.input_state * self.features)
        if self.fh == 1:
            y_hat = []
            for i in range(self.features):
                y_hat_i = (
                    self.models[i]
                    .predict(X)
                    .reshape(-1, self.latitude, self.longitude, self.fh)
                )
                y_hat.append(y_hat_i)
            y_hat = np.array(y_hat).transpose((1, 2, 3, 4, 0))
        else:
            y_hat = self.predict_autoreg(X_test, y_test)
        return y_hat

    def predict_and_evaluate(self, X_test, y_test, max_samples=5):
        y_hat = self.predict_(X_test, y_test)
        self.plot_predictions(y_hat, y_test, max_samples=max_samples)
        eval_scores = self.evaluate(y_hat, y_test)
        mae_scores = self.get_mae(y_hat, y_test)
        print("=======================================")
        print("Evaluation metrics for entire test set:")
        print("=======================================")

        sqrt_n = np.sqrt(y_test.shape[0] * self.latitude * self.longitude * self.fh)
        for i in range(self.features):
            print(
                f"{self.feature_list[i]} => RMSE: {eval_scores[i]};  MAE: {mae_scores[i]}; SE: {np.std(y_test[...,i]) / sqrt_n}"
            )

        return y_hat

    def predict_autoreg(self, X_test, y_test):
        """
        Prediction in an autoregressive manner:
        Depends on forecasting horizon parameter, each model prediction
        becomes a part of an input for next timestamp prediction.

        self.input_size -> n
        self.fh -> k

        (Xi,Xi+1, ..., Xi+n) -> Yi+n+1
        (Xi+1, Xi+2, ..., Xi+n, Yi+n+1) -> Yi+n+2
        ...
        (Xi+k-1, ..., Yi+n+k-2 ,Yi+n+k-1) -> Yi+n+k

        """
        y_hat = np.empty(y_test.shape)
        num_samples = X_test.shape[0]
        for i in range(num_samples):
            Xi = X_test[i]
            Yik = np.empty(
                (self.latitude, self.longitude, self.neighbours, self.fh, self.features)
            )
            for k in range(-1, self.fh - 1):
                Xik = Xi
                if k > -1:
                    if self.neighbours > 1:
                        Yik[..., k, :] = self.extend(y_hat[i, ..., k, :])
                        Xik = np.concatenate(
                            (Xi[..., k + 1 :, :], Yik[..., : k + 1, :]), axis=-2
                        )
                    else:
                        Xik = np.concatenate(
                            (Xi[..., k + 1 :, :], y_hat[i, ..., : k + 1, :]), axis=-2
                        )
                for j in range(self.features):
                    y_hat[i, ..., k + 1, j] = (
                        self.models[j]
                        .predict(
                            Xik.reshape(
                                -1, self.neighbours * self.input_state * self.features
                            )
                        )
                        .reshape(1, self.latitude, self.longitude)
                    )
        return y_hat

    def extend(self, Y):
        """
        Extend data sample such that it will use neighbours
        shape: (latitude, longitude, neighbours, features)
        It might be in data_processor
        """
        # TODO function that maps no. of neighbours -> radius
        if self.neighbours <= 5:
            radius = 1
        elif self.neighbours <= 13:
            radius = 2
        # ...
        else:
            radius = 3

        _, indices = DataProcessor.count_neighbours(radius=radius)
        Y_out = np.empty(
            (self.latitude, self.longitude, self.neighbours, self.features)
        )
        Y_out[..., 0, :] = Y
        for n in range(1, self.neighbours):
            i, j = indices[n - 1]
            for lo in range(self.longitude):
                for la in range(self.latitude):
                    if -1 < la + i < self.latitude and -1 < lo + j < self.longitude:
                        Y_out[la, lo, n] = Y[la + i, lo + j]
                    else:
                        Y_out[la, lo, n] = Y[la, lo]
        return Y_out
