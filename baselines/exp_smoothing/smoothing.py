#!/usr/bin/env python3
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from baselines.baseline_regressor import BaselineRegressor


class SmoothingPredictor(BaselineRegressor):
    def __init__(
        self, X_shape: tuple, fh: int, feature_list: list, smoothing_type="simple"
    ):
        super().__init__(X_shape, fh, feature_list)
        if smoothing_type == "simple":
            self.type = smoothing_type
        elif smoothing_type in ("holt", "seasonal"):
            print("WARNING: not used")
            raise DeprecationWarning
        else:
            print("Not implemented")
            raise ValueError

        self.params = [0.4 if fname == "t2m" else (0.6 if fname == "tcc" else 0.8) for fname in self.feature_list]

    def train(self, X_train, y_train, normalized=False):
        print("Not needed")
        raise KeyError

    def predict_(self, X_test, y_test):
        X = X_test.reshape(
            -1, self.latitude, self.longitude, self.input_state, self.features
        )
        y_hat = []
        for i in range(X.shape[0]):
            y_hat_i = []
            for j in range(self.features):
                ylat = []
                for lat in range(X.shape[1]):
                    ylon = []
                    for lon in range(X.shape[2]):
                        if self.type == "simple":
                            forecast = (
                                SimpleExpSmoothing(
                                    X[i, lat, lon, :, j],
                                    initialization_method="known",
                                    initial_level=X[i, lat, lon, 0, j],
                                )
                                .fit(
                                    smoothing_level=self.params[j],
                                    optimized=False
                                )
                                .forecast(self.fh)
                            )
                        elif self.type == "holt":
                            raise DeprecationWarning("Please use the simple type")
                            forecast = (
                                Holt(
                                    X[i, lat, lon, :, j],
                                    initialization_method="known",
                                    initial_level=X[i, lat, lon, 0, j],
                                    initial_trend=(X[i, lat, lon, -1, j]-X[i, lat, lon, 0, j])/self.input_state,
                                )
                                .fit(
                                    smoothing_level=0.05,
                                    smoothing_trend=(X[i, lat, lon, -1, j]-X[i, lat, lon, 0, j])/self.input_state,
                                    optimized=False,
                                )
                                .forecast(self.fh)
                            )
                        else:
                            raise ValueError
                        ylon.append(forecast)
                    ylat.append(ylon)
                y_hat_i.append(ylat)
            y_hat.append(y_hat_i)
        y_hat = (
            np.array(y_hat)
            .reshape(
                (X.shape[0], self.features, self.latitude, self.longitude, self.fh)
            )
            .transpose((0, 2, 3, 4, 1))
        )
        return y_hat
