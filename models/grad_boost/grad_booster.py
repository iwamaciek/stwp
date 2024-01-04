#!/usr/bin/env python3
import copy
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from models.baseline_regressor import BaselineRegressor


class GradBooster(BaselineRegressor):
    def __init__(self, X_shape, fh, feature_list, booster="lgb", scaler_type="standard", **kwargs):
        super().__init__(X_shape, fh, feature_list, scaler_type=scaler_type)
        if booster == "lgb":
            self.model = LGBMRegressor(verbose=-1, n_jobs=-1, **kwargs)
        elif booster == "xgb":
            self.model = XGBRegressor(n_jobs=-1, **kwargs)
        elif booster == "cat":
            self.model = CatBoostRegressor(verbose=0, thread_count=-1, **kwargs)
        elif booster == "ada":
            self.model = AdaBoostRegressor(**kwargs)
        else:
            print(f"{booster} booster not implemented")

            raise ValueError

        self.models = [copy.deepcopy(self.model) for _ in range(self.num_features)]
