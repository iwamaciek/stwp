import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import cartopy.crs as ccrs
import sys
from models.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.gnn.processor import NNDataProcessor

sys.path.append("..")
from utils.draw_functions import draw_poland

plt.style.use("ggplot")
warnings.filterwarnings("ignore")


class Analyzer:
    def __init__(self):
        self.pred_dict = {}
        self.er_dict = {}
        self.avg_er_dict = {}
        self.era5 = None
        self.df_err = None
        self.df_pred = None
        self.scalers = None
        self.nn_proc = None
        self.min_length = None
        self.feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]
        self.map_dict = {}

    def init(self):
        self.map_dict = {
            "grad_booster": r"$\Phi^{gb}$",
            "simple_linear_regressor": r"$\Phi^{slr}$",
            "linear_regressor": r"$\Phi^{lr}$",
            "unet": r"$\Phi^{unet}$",
            "trans": r"$\Phi^{gnn}$",
            "baseline_regressor": r"$\Phi^{naive}$",
            "tigge": r"$\Phi^{tigge}$",
        }
        self.get_pred_tensors()
        self.get_era5()
        self.calculate_errors()

    def coherent_tensors(self):
        self.min_length = min(
            self.pred_dict[model].shape[0]
            for model in self.pred_dict
            if model != "tigge"
        )
        for model, pred_tensor in self.pred_dict.items():
            if model != "tigge":
                self.pred_dict[model] = pred_tensor[-self.min_length :]
        self.era5 = self.era5[-self.min_length :]

    def get_pred_tensors(self, path="../data/pred/"):
        for model in os.listdir(path):
            p = os.path.join(path, model)
            if os.path.isfile(p):
                pred_tensor = np.load(p)
                if "tigge" in str(model):
                    self.pred_dict[model.split("_2024")[0]] = pred_tensor
                else:
                    self.pred_dict[model.split("_2024")[0]] = pred_tensor[
                        ..., 0
                    ]  # fh=1
        self.min_length = min(
            self.pred_dict[model].shape[0]
            for model in self.pred_dict
            if model != "tigge"
        )

    def get_era5(self):
        processor = DataProcessor(path="../data/input/data2021-small.grib")
        self.era5 = processor.data

    def generate_full_metrics(self, verbose=False, latex=True):
        rmse_results, mae_results = [], []
        models = []
        for model, predictions in self.pred_dict.items():
            if "tigge" in model:
                rmse_per_model, mae_per_model = self.calculate_metrics(
                    predictions, self.era5[1::2], verbose=verbose
                )
            else:
                rmse_per_model, mae_per_model = self.calculate_metrics(
                    predictions[-self.min_length :],
                    self.era5[-self.min_length :],
                    verbose=verbose,
                )
            if verbose:
                print(f"Model: {model}\n\n")
            models.append(self.map_dict[model])
            rmse_results.append(rmse_per_model)
            mae_results.append(mae_per_model)
        rmse_results = np.array(rmse_results)
        mae_results = np.array(mae_results)
        rmse_df = pd.DataFrame(rmse_results, columns=self.feature_list, index=models)
        mae_df = pd.DataFrame(mae_results, columns=self.feature_list, index=models)
        if latex:
            print(
                rmse_df.to_latex(
                    float_format="%.3f", caption="RMSE Results", label="tab:rmse"
                )
            )
            print(
                mae_df.to_latex(
                    float_format="%.3f", caption="MAE Results", label="tab:mae"
                )
            )

    def get_scalers(self):
        if self.nn_proc is None:
            self.nn_proc = NNDataProcessor(
                path="../data/input/data2019-2021-small.grib"
            )
            self.nn_proc.preprocess()
            self.scalers = self.nn_proc.scalers

    def calculate_errors(self):
        for model, pred_tensor in self.pred_dict.items():
            if model != "tigge":
                self.er_dict[model] = np.zeros_like(self.era5[-self.min_length :])
                for i in range(len(self.feature_list)):
                    self.er_dict[model][..., i] = (
                        self.era5[-self.min_length :, ..., i]
                        - pred_tensor[-self.min_length :, ..., i]
                    )
            # else:
            #     self.er_dict[model] = np.zeros_like(self.era5[1::2])
            #     for i in range(len(self.feature_list)):
            #         self.er_dict[model][..., i] = self.era5[1::2][..., i] - pred_tensor[..., i]
            # ???

    def plot_err_corr_matrix(self, save=False):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        divider = fig.add_axes([1.05, 0.15, 0.02, 0.8])  # (left, bottom, width, height)
        vmin, vmax = 1, -1
        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten())):
            ax.set_title(feature)
            df_err = pd.DataFrame(
                {
                    self.map_dict[model]: self.er_dict[model][..., i].reshape(-1)
                    for model in self.er_dict
                }
            )
            vmin = min(vmin, df_err.corr().values.min())
            vmax = max(vmax, df_err.corr().values.max())
            sns.heatmap(
                df_err.corr(),
                square=True,
                cmap="RdYlGn",
                annot=True,
                ax=ax,
                annot_kws={
                    "fontsize": 10
                },  # Set font size for numbers in correlation boxes
                fmt=".2f",  # Format for the numbers (two decimal places)
                cbar=i == 0,
                cbar_ax=None if i else divider,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        cbar = fig.colorbar(
            axs[-1, -1].collections[0], cax=divider, pad=0.02
        )  # Adjust pad as needed
        cbar.set_ticks(
            np.linspace(vmin, vmax, 6)
        )  # Adjust the number of ticks as needed
        fig.subplots_adjust(
            right=0.8, hspace=0.5
        )  # Increase the vertical space between subplots
        plt.tight_layout()
        if save:
            plt.savefig("../data/analysis/err_corr_matrix.pdf")
        plt.show()

    def plot_pred_corr_matrix(self, save=False):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        filtered_pred_dict = {
            key: value for key, value in self.pred_dict.items() if key != "tigge"
        }
        divider = fig.add_axes([1.05, 0.05, 0.02, 0.8])  # (left, bottom, width, height)
        vmin, vmax = 1, -1
        for i, (feature, ax) in enumerate(zip(self.feature_list, axs.flatten())):
            ax.set_title(feature)
            df_pred = pd.DataFrame(
                {
                    self.map_dict[key]: filtered_pred_dict[key][..., i].reshape(-1)
                    for key in filtered_pred_dict
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
        y_trans = self.pred_dict["trans"][1:][1::2]
        y_tigge = self.pred_dict["tigge"]
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
                sns.distplot(data_for_feature, kde=True, color="maroon", ax=ax)
            else:
                sns.histplot(data_for_feature, kde=True, color="maroon", ax=ax)
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

    def generate_error_maps(self, save=False):
        self.calculate_avg_err()
        num_models = len(self.avg_er_dict)
        num_features = len(self.feature_list)
        lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
        spatial = {
            "lat_span": lat_span,
            "lon_span": lon_span,
            "spatial_limits": spatial_limits,
        }
        fig, axes = plt.subplots(
            num_features,
            num_models,
            figsize=(15, 15),
            subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
        )
        # x = 0.7, y = 0.95, weight = "bold"
        for j, model in enumerate(self.avg_er_dict.keys()):
            ax_title = fig.add_subplot(num_features, num_models, j + 1)
            ax_title.set_title(self.map_dict[model], fontsize=12, y=1.05)
            ax_title.axis("off")
            for i, feature in enumerate(self.feature_list):
                error_map = self.avg_er_dict[model][..., i]
                title = rf"$(Y - \hat{{Y}})_{{{feature}}}$"
                axes[i, j].axis("off")
                draw_poland(axes[i, j], error_map, title, "binary", **spatial)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save:
            plt.savefig("../data/analysis/error_maps.pdf")
        plt.show()

    def calculate_avg_err(self):
        for model in self.er_dict.keys():
            avg_tensor = np.zeros((self.er_dict[model].shape[1:]))
            for i in range(len(self.feature_list)):
                avg_tensor[..., i] = np.mean(self.er_dict[model][..., i], axis=0)
            self.avg_er_dict[model] = avg_tensor

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
