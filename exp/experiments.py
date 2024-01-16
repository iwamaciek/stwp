import os
import numpy as np
from models.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.tigge.tigge import evaluate_and_compare


class Analyzer:
    def __init__(self):
        self.pred_dir = {}
        self.er_dir = {}
        self.era5 = None

    def init(self):
        self.get_pred_tensors()
        self.get_era5()
        # self.best_with_tigge_approx()
        # self.calculate_errors()

    def get_pred_tensors(self, path="../data/pred/"):
        for model in os.listdir(path):
            p = os.path.join(path, model)
            if os.path.isfile(p):
                pred_tensor = np.load(p)
                self.pred_dir[model.split("_2024")[0]] = pred_tensor

    def get_era5(self):
        processor = DataProcessor(path="../data/input/data2021-small.grib")
        self.era5 = processor.data

    @staticmethod
    def calculate_metrics(y_hat, y, verbose=True):
        rmse_features, mae_features = [], []
        for i in range(6):
            y_fi = y[..., i, :].reshape(-1, 1)
            y_hat_fi = y_hat[..., i, :].reshape(-1, 1)
            rmse = np.sqrt(mean_squared_error(y_hat_fi, y_fi))
            mae = mean_absolute_error(y_hat_fi, y_fi)
            if verbose:
                print(f"RMSE for f{i}: {rmse}; MAE for f{i}: {mae};")
            rmse_features.append(rmse)
            mae_features.append(mae)
        return rmse_features, mae_features

    def calculate_errors(self):
        for model, pred_tensor in self.pred_dir.items():
            self.er_dir[model] = np.abs(pred_tensor - self.era5)

    def calculate_corr_matrix(self):
        # TODO
        pass

    def best_with_tigge_approx(self):
        y_trans = self.pred_dir["trans"][..., 0][1:][1::2]
        y_tigge = self.pred_dir["tigge"]
        for a in np.arange(0.1, 1, 0.1):
            print("Alpha: ", a)
            self.combine_and_evaluate(y_trans, y_tigge, alpha=a)
            print("\n\n")

    def combine_and_evaluate(self, y1, y2, alpha=0.5):
        y = alpha * y1 + (1 - alpha) * y2
        evaluate_and_compare(y, self.era5[1::2], max_samples=0)
