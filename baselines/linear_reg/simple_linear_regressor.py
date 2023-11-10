#!/usr/bin/env python3
import copy
import numpy as np
from baselines.data_processor import DataProcessor
from baselines.baseline_regressor import BaselineRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


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
        scaler_type="min_max",
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

        if scaler_type == "min_max":
            self.scaler = MinMaxScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "max_abs":
            self.scaler = MaxAbsScaler()
        else:
            print(f"{scaler_type} scaler not implemented")
            raise ValueError

        self.models = [copy.deepcopy(self.model) for _ in range(self.features)]
        self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.features)]

    def train(self, X_train, y_train, normalized=False):
        for i in range(self.features):
            Xi = X_train[..., i].reshape(-1, self.neighbours * self.input_state)
            yi = y_train[..., 0, i].reshape(-1, 1)
            if normalized:
                self.scalers[i].fit(Xi, yi)
            self.models[i].fit(Xi, yi)

    def predict_(self, X_test, y_test):
        if self.fh == 1:
            y_hat = []
            for i in range(self.features):
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
            for j in range(self.features):
                y_hat_ij = np.zeros(y_test[i].shape[:-1])
                Xij = X_test[i, :, :, :, j].reshape(
                    -1, self.neighbours * self.input_state
                )
                y_hat_ij[..., 0] = (
                    self.models[j]
                    .predict(Xij)
                    .reshape(1, self.latitude, self.longitude)
                )
                for k in range(self.fh - 1):
                    Xij = np.concatenate(
                        (X_test[i, ..., k + 1 :, j], y_hat_ij[..., : k + 1]), axis=2
                    )
                    Xij = Xij.reshape((-1, self.input_state))
                    y_hat_ij[..., k + 1] = (
                        self.models[j]
                        .predict(Xij)
                        .reshape(1, self.latitude, self.longitude)
                    )
                y_hat[i, :, :, :, j] = y_hat_ij
        return y_hat

    def extend(self, Y):
        pass
        # TODO
