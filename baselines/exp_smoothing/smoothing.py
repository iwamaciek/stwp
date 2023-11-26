#!/usr/bin/env python3
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from baselines.baseline_regressor import BaselineRegressor


class SmoothingPredictor(BaselineRegressor):
    def __init__(
        self, X_shape: tuple, fh: int, feature_list: list, smoothing_type="simple"
    ):
        super().__init__(X_shape, fh, feature_list)
        if smoothing_type in ("simple", "holt"):
            self.type = smoothing_type
        else:
            print("Not implemented")
            raise ValueError

        self.models = [None for _ in range(self.features)]
        self.params = [None for _ in range(self.features)]

    def train(self, X_train, y_train, normalized=False):
        X = X_train.reshape(-1, self.features)
        # print(X.shape)
        for i in range(self.features):
            if self.type == "simple":
                self.models[i] = SimpleExpSmoothing(
                    X[:, i], initialization_method="estimated"
                ).fit()
                self.params[i] = self.models[i].params_formatted["param"]
            elif self.type == "holt":
                self.models[i] = Holt(X[:, i], initialization_method="estimated").fit()
                self.params[i] = self.models[i].params_formatted["param"]
        print(self.params[0])
        print()

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
                                    initial_level=self.params[j]["initial_level"],
                                )
                                .fit(smoothing_level=self.params[j]["smoothing_level"])
                                .forecast(self.fh)
                            )
                        elif self.type == "holt":
                            forecast = (
                                Holt(
                                    X[i, lat, lon, :, j],
                                    initialization_method="known",
                                    initial_level=self.params[j]["initial_level"],
                                    initial_trend=self.params[j]["initial_trend"],
                                )
                                .fit(
                                    smoothing_level=self.params[j]["smoothing_level"],
                                    smoothing_trend=self.params[j]["smoothing_trend"],
                                )
                                .forecast(self.fh)
                            )
                        else:
                            raise ValueError
                        ylon.append(forecast)
                    ylat.append(ylon)
                y_hat_i.append(ylat)
            y_hat.append(y_hat_i)
            if i % 50 == 0:
                print(i, "/", X.shape[0])
        y_hat = (
            np.array(y_hat)
            .reshape(
                (X.shape[0], self.features, self.latitude, self.longitude, self.fh)
            )
            .transpose((0, 2, 3, 4, 1))
        )
        return y_hat
