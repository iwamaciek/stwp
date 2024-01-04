#!/usr/bin/env python3
from models.baseline_regressor import BaselineRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import copy


class LinearRegressor(BaselineRegressor):
    def __init__(
        self,
        X_shape,
        fh,
        feature_list,
        regressor_type="linear",
        alpha=1.0,
        scaler_type="standard",
    ):
        super().__init__(X_shape, fh, feature_list, scaler_type=scaler_type)

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
