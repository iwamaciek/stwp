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

        self.params = [0.4 if fname == "t2m" else (0.6 if fname == "tcc" else 0.8) for fname in self.feature_list]

    def train(self, X_train, y_train, normalized=False):
        print("Not needed")
        raise KeyError
        # X = X_train.reshape(-1, self.latitude * self.longitude, self.input_state, self.features)
        # for feature in range(self.features):
        #     init_levels = []
        #     smoothing_levels = []
        #     init_trends = []
        #     smoothing_trends = []
        #     for sample in range(X.shape[0]):
        #         for grid in range(X.shape[1]):
        #             if self.type == "simple":
        #                 params = SimpleExpSmoothing(
        #                     X[sample, grid, :, feature], initialization_method="estimated"
        #                 ).fit().params_formatted["param"]
        #                 init_levels.append(params["initial_level"])
        #                 smoothing_levels.append(params["smoothing_level"])
        #             elif self.type == "holt":
        #                 params = Holt(X[sample, grid, :, feature], initialization_method="estimated").fit().params_formatted["param"]
        #                 init_levels.append(params["initial_level"])
        #                 init_trends.apped(params["initial_trend"])
        #                 smoothing_levels.append(params["smoothing_level"])
        #                 smoothing_trends.append(params["smoothing_trend"])
        #             else:
        #                 raise ValueError
        #     if self.type == "simple":
        #         self.params[feature] = {"initial_level": np.mean(np.array(init_levels)), "smoothing_level": np.mean(np.array(smoothing_levels))}
        #     elif self.type == "holt":
        #         self.params[feature] = {"initial_level": np.mean(np.array(init_levels)), "smoothing_level": np.mean(np.array(smoothing_levels)), "initial_trend": np.mean(np.array(init_trends)), "smoothing_trend": np.mean(np.array(smoothing_trends))}
        #     print(self.feature_list[feature], self.params[feature])

    def predict_(self, X_test, y_test, alpha):
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
            # if i % 50 == 0:
            #     print(i, "/", X.shape[0])
        y_hat = (
            np.array(y_hat)
            .reshape(
                (X.shape[0], self.features, self.latitude, self.longitude, self.fh)
            )
            .transpose((0, 2, 3, 4, 1))
        )
        return y_hat
