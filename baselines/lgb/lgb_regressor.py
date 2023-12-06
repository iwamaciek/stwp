#!/usr/bin/env python3
import copy
from lightgbm import LGBMRegressor
from baselines.baseline_regressor import BaselineRegressor


class LightGBMRegressor(BaselineRegressor):
    def __init__(self, X_shape, fh, feature_list, **kwargs):
        super().__init__(X_shape, fh, feature_list)
        self.model = LGBMRegressor(verbose=-1, **kwargs)
        self.models = [copy.deepcopy(self.model) for _ in range(self.num_features)]
