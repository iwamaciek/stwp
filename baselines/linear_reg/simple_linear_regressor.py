#!/usr/bin/env python3
import copy
import numpy as np
from baselines.data_processor import DataProcessor
from baselines.baseline_regressor import BaselineRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


class SimpleLinearRegressor(BaselineRegressor):
    """
    Model M_i takes X_fi as an input - models have no access to different features
    """

    def __init__(
        self,
        X_shape,
        fh,
        feature_list,
        regressor_type="linear",
        alpha=1.0,
    ):
        super().__init__(X_shape, fh, feature_list)

        if regressor_type == "linear":
            self.model = LinearRegression()
        elif regressor_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif regressor_type == "lasso":
            self.model = Lasso(alpha=alpha)
        elif regressor_type == "elastic_net":
            self.model = ElasticNet(alpha=alpha)
        else:
            print(f"{regressor_type} regressor not implemented")
            raise ValueError

        self.models = [copy.deepcopy(self.model) for _ in range(self.num_features)]

    def train(self, X_train, y_train, normalize=False):
        for i in range(self.num_features):
            Xi = X_train[..., i].reshape(-1, self.neighbours * self.input_state)
            yi = y_train[..., 0, i].reshape(-1, 1)
            if normalize:
                self.scalers[i].fit(yi)
            self.models[i].fit(Xi, yi)

    def predict_(self, X_test, y_test):
        if self.fh == 1:
            y_hat = []
            for i in range(self.num_features):
                Xi = X_test[..., i].reshape(-1, self.neighbours * self.input_state)
                y_hat_i = (
                    self.models[i]
                    .predict(Xi)
                    .reshape(-1, self.latitude, self.longitude, self.fh)
                )
                y_hat.append(y_hat_i)
            y_hat = np.array(y_hat).transpose((1, 2, 3, 4, 0))
        else:
            y_hat = self.predict_autoreg(X_test, y_test)
        return y_hat

    def predict_autoreg(self, X_test, y_test):
        y_hat = np.empty(y_test.shape)
        num_samples = X_test.shape[0]
        for i in range(num_samples):
            for j in range(self.num_features):
                y_hat_ij = np.zeros(y_test[i].shape[:-1])
                Xij = X_test[i, ..., j].reshape(-1, self.neighbours * self.input_state)
                y_hat_ij[..., 0] = (
                    self.models[j]
                    .predict(Xij)
                    .reshape(1, self.latitude, self.longitude)
                )
                for k in range(self.fh - 1):
                    # print("Xij before conc", Xij.shape)
                    if self.fh - self.input_state < 2:
                        autoreg_start = 0
                    else:
                        autoreg_start = max(0, k - self.input_state + 1)

                    if self.neighbours > 1:
                        Xij = np.concatenate(
                            (
                                X_test[i, ..., k + 1 :, j],
                                self.extend(y_hat_ij[..., autoreg_start : k + 1]),
                            ),
                            axis=3,
                        )
                    else:
                        Xij = np.concatenate(
                            (
                                X_test[i, ..., k + 1 :, j],
                                y_hat_ij[..., autoreg_start : k + 1],
                            ),
                            axis=2,
                        )
                    # print("Xij after conc", Xij.shape)
                    Xij = Xij.reshape(-1, self.neighbours * self.input_state)
                    # print("Xij before predict", Xij.shape)
                    # print("We need:",  y_hat_ij[..., k + 1].shape)
                    y_hat_ij[..., k + 1] = (
                        self.models[j]
                        .predict(Xij)
                        .reshape(1, self.latitude, self.longitude)
                    )
                y_hat[i, ..., j] = y_hat_ij
        return y_hat

    def extend(self, Y):
        # TODO function that maps no. of neighbours -> radius
        if self.neighbours <= 5:
            radius = 1
        elif self.neighbours <= 13:
            radius = 2
        # ...
        else:
            radius = 3

        _, indices = DataProcessor.count_neighbours(radius=radius)
        Y_out = np.empty((self.latitude, self.longitude, self.neighbours, Y.shape[-1]))
        Y_out[..., 0, :] = Y.reshape((self.latitude, self.longitude, -1))
        for n in range(1, self.neighbours):
            i, j = indices[n - 1]
            for lo in range(self.longitude):
                for la in range(self.latitude):
                    if -1 < la + i < self.latitude and -1 < lo + j < self.longitude:
                        Y_out[la, lo, n] = Y[la + i, lo + j]
                    else:
                        Y_out[la, lo, n] = Y[la, lo]
        return Y_out
