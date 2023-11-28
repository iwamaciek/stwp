#!/usr/bin/env python3
import copy
from lightgbm import LGBMRegressor
from baselines.baseline_regressor import BaselineRegressor
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
)


class LightGBMRegressor(BaselineRegressor):
    def __init__(self, X_shape, fh, feature_list, scaler_type="robust"):
        super().__init__(X_shape, fh, feature_list)
        self.model = LGBMRegressor(verbose=-1)

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

        self.models = [copy.deepcopy(self.model) for _ in range(self.features)]
        self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.features)]

    def train(self, X_train, y_train, normalize=False):
        X = X_train.reshape(-1, self.neighbours * self.input_state * self.features)
        for i in range(self.features):
            yi = y_train[..., 0, i].reshape(-1)
            if normalize:
                self.scalers[i].fit(yi)
            self.models[i].fit(X, yi)
