import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from models.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.gnn.processor import NNDataProcessor

plt.style.use("ggplot")
warnings.filterwarnings("ignore")


class Analyzer:
    def __init__(self):
        self.pred_dir = {}
        self.er_dir = {}
        self.era5 = None
        self.df_err = None
        self.df_pred = None
        self.scalers = None
        self.nn_proc = None
        self.feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]

    def init(self):
        self.get_pred_tensors()
        self.get_era5()
        self.get_era5()
        self.calculate_errors()

    def get_pred_tensors(self, path="../data/pred/"):
        for model in os.listdir(path):
            p = os.path.join(path, model)
            if os.path.isfile(p):
                pred_tensor = np.load(p)
                if "tigge" in str(model):
                    self.pred_dir[model.split("_2024")[0]] = pred_tensor
                else:
                    self.pred_dir[model.split("_2024")[0]] = pred_tensor[..., 0]  # fh=1

    def get_era5(self):
        processor = DataProcessor(path="../data/input/data2021-small.grib")
        self.era5 = processor.data

    def get_scalers(self):
        if self.nn_proc is None:
            self.nn_proc = NNDataProcessor(
                path="../data/input/data2019-2021-small.grib"
            )
            self.nn_proc.preprocess()
            self.scalers = self.nn_proc.scalers

    def calculate_errors(self):
        for model, pred_tensor in self.pred_dir.items():
            if "tigge" not in str(model):
                self.er_dir[model] = np.zeros_like(
                    self.era5,
                )
                for i in range(len(self.feature_list)):
                    self.er_dir[model][..., i] = (
                        self.era5[..., i] - pred_tensor[1:, ..., i]
                    )
            # else:
            #     self.er_dir[model] = np.zeros_like(self.era5[1::2])
            #     for i in range(len(self.feature_list)):
            #         self.er_dir[model][..., i] = self.era5[1::2][..., i] - pred_tensor[..., i]
            # ???

    def plot_err_corr_matrix(self, save=False):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        divider = fig.add_axes([1.05, 0.15, 0.02, 0.8])  # (left, bottom, width, height)
        vmin, vmax = np.inf, -np.inf  # Initialize vmin and vmax for colorbar scaling
        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten())):
            ax.set_title(feature)
            df_err = pd.DataFrame(
                {key: self.er_dir[key][..., i].reshape(-1) for key in self.er_dir}
            )
            vmin = min(vmin, df_err.corr().values.min())
            vmax = max(vmax, df_err.corr().values.max())
            sns.heatmap(
                df_err.corr(),
                square=True,
                cmap="RdYlGn",
                annot=True,
                ax=ax,
                annot_kws={"fontsize": 8},
                cbar=i == 0,  # Show colorbar only for the first subplot
                cbar_ax=None
                if i
                else divider,  # Use the shared colorbar axis for all subplots
                vmin=vmin,
                vmax=vmax,  # Set colorbar limits
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0
            )  # Added for y-axis label rotation

        cbar = fig.colorbar(
            axs[-1, -1].collections[0], cax=divider, pad=0.02
        )  # Adjust pad as needed
        cbar.set_ticks(
            np.linspace(vmin, vmax, 6)
        )  # Adjust the number of ticks as needed
        fig.subplots_adjust(right=0.8)
        plt.tight_layout()
        if save:
            plt.savefig("../data/analysis/err_corr_matrix.pdf")
        plt.show()

    def plot_pred_corr_matrix(self, save=False):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        filtered_pred_dir = {
            key: value for key, value in self.pred_dir.items() if key != "tigge"
        }
        divider = fig.add_axes([1.05, 0.15, 0.02, 0.8])  # (left, bottom, width, height)
        vmin, vmax = np.inf, -np.inf  # Initialize vmin and vmax for colorbar scaling
        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten())):
            ax.set_title(feature)
            df_pred = pd.DataFrame(
                {
                    key: filtered_pred_dir[key][..., i].reshape(-1)
                    for key in filtered_pred_dir
                }
            )
            vmin = min(vmin, df_pred.corr().values.min())
            vmax = max(vmax, df_pred.corr().values.max())
            sns.heatmap(
                df_pred.corr(),
                square=True,
                cmap="RdYlGn",
                annot=True,
                ax=ax,
                annot_kws={"fontsize": 8},
                cbar=i == 0,
                cbar_ax=None
                if i
                else divider,  # Use the shared colorbar axis for all subplots
                vmin=vmin,
                vmax=vmax,  # Set colorbar limits
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        cbar = fig.colorbar(
            axs[-1, -1].collections[0], cax=divider, pad=0.02
        )  # Adjust pad as needed
        cbar.set_ticks(
            np.linspace(vmin, vmax, 6)
        )  # Adjust the number of ticks as needed
        plt.tight_layout()
        if save:
            plt.savefig("../data/analysis/pred_corr_matrix.pdf")
        plt.show()

    def best_with_tigge_approx(self, verbose=True, plot=False, save=False):
        y_trans = self.pred_dir["trans"][1:][1::2]
        y_tigge = self.pred_dir["tigge"]
        alphas = np.arange(0, 1.1, 0.1)
        if verbose:
            for a in alphas:
                print("Alpha: ", a)
                self.combine_and_evaluate(y_trans, y_tigge, alpha=a)
                print("\n\n")
        if plot:
            losses = np.zeros_like(alphas)
            for i, a in enumerate(alphas):
                losses[i] = self.combine_and_evaluate(
                    y_trans, y_tigge, alpha=a, consolidate=True
                )
            plt.plot(alphas, losses, "-o")
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\overline{\|\mathcal{L}_{RMSE}\|}$")
            if save:
                plt.savefig("../data/analysis/alpha_loss.pdf")
            plt.show()

    def combine_and_evaluate(self, y1, y2, alpha=0.5, consolidate=False):
        y = alpha * y1 + (1 - alpha) * y2
        if consolidate:
            if self.scalers is None:
                self.get_scalers()
            feat = len(self.feature_list)
            losses = np.zeros((feat,))
            for i in range(feat):
                y_normalized_i = self.scalers[i].transform(y[..., i].reshape(-1, 1))
                era5_normalized_i = self.scalers[i].transform(
                    self.era5[1::2][..., i].reshape(-1, 1)
                )
                losses[i] = np.sqrt(
                    mean_squared_error(y_normalized_i, era5_normalized_i)
                )
            return np.mean(losses)
        self.calculate_metrics(y, self.era5[1::2])

    def plot_feature_distributions(self, plot_type="dist", save=False, stats=True):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        if stats:
            stats_df = pd.DataFrame(
                index=self.feature_list,
                columns=["Mean", "Standard Deviation", "Skewness", "Kurtosis"],
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
                var_val = np.std(data_for_feature)
                skewness_val = pd.Series(data_for_feature).skew()
                kurtosis_val = pd.Series(data_for_feature).kurtosis()
                stats_df.loc[feature] = [mean_val, var_val, skewness_val, kurtosis_val]
        if stats:
            print(stats_df.to_latex())
        plt.tight_layout()
        if save:
            plt.savefig("../data/analysis/feature_dist.pdf")
        plt.show()

    def calculate_metrics(self, y_hat, y, verbose=True):
        rmse_features, mae_features = [], []
        for i in range(len(self.feature_list)):
            y_fi = y[..., i].reshape(-1, 1)
            y_hat_fi = y_hat[..., i].reshape(-1, 1)
            rmse = np.sqrt(mean_squared_error(y_hat_fi, y_fi))
            mae = mean_absolute_error(y_hat_fi, y_fi)
            if verbose:
                print(
                    f"RMSE for {self.feature_list[i]}: {rmse}; MAE for {self.feature_list[i]}: {mae};"
                )
            rmse_features.append(rmse)
            mae_features.append(mae)
        return rmse_features, mae_features
