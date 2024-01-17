import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from models.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


class Analyzer:
    def __init__(self):
        self.pred_dir = {}
        self.er_dir = {}
        self.era5 = None
        self.df_err = None
        self.df_pred = None
        self.feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]

    def init(self):
        self.get_pred_tensors()
        self.get_era5()
        self.calculate_errors()

    def get_pred_tensors(self, path="../data/pred/"):
        for model in os.listdir(path):
            p = os.path.join(path, model)
            if os.path.isfile(p):
                pred_tensor = np.load(p)
                self.pred_dir[model.split("_2024")[0]] = pred_tensor

    def get_era5(self):
        processor = DataProcessor(path="../data/input/data2021-small.grib")
        self.era5 = processor.data

    def calculate_errors(self):
        for model, pred_tensor in self.pred_dir.items():
            if "tigge" not in str(model):
                self.er_dir[model] = self.era5 - np.squeeze(pred_tensor, axis=-1)[1:]

    def plot_err_corr_matrix(self, save=False):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten())):
            ax.set_title(feature)
            df_err = pd.DataFrame(
                {key: self.er_dir[key][..., i, :].reshape(-1) for key in self.er_dir}
            )
            sns.heatmap(
                df_err.corr(),
                square=True,
                cmap="RdYlGn",
                annot=True,
                ax=ax,
                annot_kws={"fontsize": 8},
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        if save:
            plt.savefig("../data/analysis/err_corr_matrix.pdf")
        plt.show()

    def plot_pred_corr_matrix(self, save=False):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        filtered_pred_dir = {
            key: value for key, value in self.pred_dir.items() if key != "tigge"
        }
        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten())):
            ax.set_title(feature)
            df_pred = pd.DataFrame(
                {
                    key: filtered_pred_dir[key][..., i, :].reshape(-1)
                    for key in filtered_pred_dir
                }
            )
            sns.heatmap(
                df_pred.corr(),
                square=True,
                cmap="RdYlGn",
                annot=True,
                ax=ax,
                annot_kws={"fontsize": 8},
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        if save:
            plt.savefig("../data/analysis/pred_corr_matrix.pdf")
        plt.show()

    def best_with_tigge_approx(self):
        y_trans = self.pred_dir["trans"][..., 0][1:][1::2]
        y_tigge = self.pred_dir["tigge"]
        for a in np.arange(0.1, 1, 0.1):
            print("Alpha: ", a)
            self.combine_and_evaluate(y_trans, y_tigge, alpha=a)
            print("\n\n")

    def combine_and_evaluate(self, y1, y2, alpha=0.5):
        y = alpha * y1 + (1 - alpha) * y2
        self.calculate_metrics(y, self.era5[1::2])

    def plot_feature_distributions(self, plot_type="dist", save=False, stats=True):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        if stats:
            stats_df = pd.DataFrame(
                index=self.feature_list,
                columns=["Mean", "Variance", "Skewness", "Kurtosis"],
            )
        for i, (feature, ax) in enumerate(zip(self.feature_list, axes.flatten())):
            data_for_feature = self.era5[..., i].flatten()
            if plot_type == "dist":
                sns.distplot(data_for_feature, kde=True, color="skyblue", ax=ax)
            else:
                sns.histplot(data_for_feature, kde=True, color="skyblue", ax=ax)
            ax.set_title(feature)
            if stats:
                mean_val = np.mean(data_for_feature)
                var_val = np.var(data_for_feature)
                skewness_val = pd.Series(data_for_feature).skew()
                kurtosis_val = pd.Series(data_for_feature).kurtosis()
                stats_df.loc[feature] = [mean_val, var_val, skewness_val, kurtosis_val]
        if stats:
            print(stats_df.to_latex())
        plt.tight_layout()
        if save:
            plt.savefig("../data/analysis/feature_dist.pdf")
        plt.show()

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
