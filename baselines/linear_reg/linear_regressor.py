#!/usr/bin/env python3
from baselines.baseline_regressor import BaselineRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import copy


class LinearRegressor(BaselineRegressor):
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
