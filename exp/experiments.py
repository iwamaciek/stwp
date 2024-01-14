import os
import numpy as np
from models.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Analyzer:
    def __init__(self):
        self.pred_dir = {}
        self.er_dir = {}
        self.era5 = None

    def err_corr_analysis(self):
        self.get_pred_tensors()
        # self.get_era5()
        # self.calculate_errors()

    def get_pred_tensors(self, path="../data/pred/"):
        for model in os.listdir(path):
            p = os.path.join(path, model)
            if os.path.isfile(p):
                pred_tensor = np.load(p)
                self.pred_dir[model.split("_2024")[0]] = pred_tensor

    def get_era5(self):
        processor = DataProcessor()
        X, y = processor.preprocess()
        _, _, _, data = processor.train_val_test_split(X, y, split_type=2)
        self.era5 = data.transpose((0, 1, 2, 4, 3))

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

    def extract_seasons(self):
        pass
        # month = y_hat.shape[0] // 12
        # winter = np.concatenate((y_hat[:2 * month], y_hat[-month:]), axis=0)
        # spring = y_hat[2 * month:5 * month]
        # summer = y_hat[5 * month:8 * month]
        # autumn = y_hat[8 * month:11 * month]
        # return winter, spring, summer, autumn
