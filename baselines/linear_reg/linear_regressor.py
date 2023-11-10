from baselines.baseline_regressor import BaselineRegressor


class LinearRegressor(BaselineRegressor):
    def __init__(self, X_shape, fh, feature_list, regressor_type="linear", alpha=1.0):
        super().__init__(X_shape, fh, feature_list, regressor_type, alpha)
