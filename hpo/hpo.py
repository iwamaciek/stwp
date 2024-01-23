import numpy as np
import optuna
import matplotlib.pyplot as plt
import torch
import json
import time
import sys
import os


sys.path.append("..")

from models.data_processor import DataProcessor
from models.linear_reg.linear_regressor import LinearRegressor
from models.linear_reg.simple_linear_regressor import SimpleLinearRegressor
from models.grad_boost.grad_booster import GradBooster
from models.gnn.trainer import Trainer
from models.cnn.trainer import Trainer as CNNTrainer
from models.config import config as cfg
from exp.experiments import Analyzer
from utils.draw_functions import draw_poland

from utils.progress_bar import printProgressBar


class InvalidBaselineException(Exception):
    "Raised when baseline type in invalid"
    pass


class HPO:
    def __init__(
        self,
        baseline_type,
        n_trials,
        use_neighbours=False,
        sequence_length=1,
        forcasting_horizon=1,
        sequence_n_trials=15,
        sequence_alpha=5,
        sequence_regressor="ridge",
        fh_n_trials=15,
        max_alpha=10,
        num_epochs=3,
        subset=None,
    ):
        self.baseline_type = baseline_type
        self.n_trials = n_trials
        self.use_neighbours = use_neighbours
        self.sequence_n_trials = sequence_n_trials
        self.sequence_alpha = sequence_alpha
        self.sequence_regressor = sequence_regressor
        self.fh_n_trials = fh_n_trials
        self.processor = DataProcessor()
        self.data, self.feature_list = self.processor.data, self.processor.feature_list
        self.best_s = sequence_length
        self.fh = forcasting_horizon
        self.best_fh = self.fh
        self.regressors = ["lasso", "ridge", "elastic_net"]

        self.subset = subset
        self.num_epochs = num_epochs

        self.scalers = ["standard", "min_max", "max_abs", "robust"]

        self.max_alpha = max_alpha
        self.verbosity = False
        self.params = {}

        self.sequence_plot_x = []
        self.sequence_plot_y = []

        self.sequence_plot_time = []

        self.fh_plot_x = []
        self.fh_plot_y = []
        self.fh_plot_time = []

        self.metrics = []
        self.metrics_mae = []

        self.metrics_for_scalers = {}

        self.not_normalized_plot_sequence = {}
        self.not_normalized_plot_fh = {}

        self.month_error = {}

        self.gnn_verbose = True

        self.best_alpha = 0.1
        self.gnn_alpha_plot_x = []
        self.gnn_alpha_plot_y = []


        self.gnn_cell_plot_x = []
        self.gnn_cell_plot_y = []

        self.best_layer = 5

        self.error_map = []

        self.era_path = "../data/pred/"

    def set_params(self, params):
        self.params = params
    
    def run_hpo(self):
        return -1

    def clear_sequence_plot(self):
        self.sequence_plot_x = []
        self.sequence_plot_y = []

    def clear_alpha_plot(self):
        self.gnn_alpha_plot_x = []
        self.gnn_alpha_plot_y = []

    def clear_fh_plot(self):
        self.fh_plot_x = []
        self.fh_plot_y = []

    def clear_params(self):
        self.params = {}

    def sequence_objective(self, trial, max_sequence_length=15):
        try:
            s = trial.suggest_int("s", 1, max_sequence_length)

            # processor = DataProcessor(self.data)
            X, y = self.processor.preprocess(s, self.fh, self.use_neighbours)
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)
            start_time = time.time()
            if self.baseline_type == "simple-linear":
                linearreg = SimpleLinearRegressor(
                    X.shape,
                    self.fh,
                    self.feature_list,
                    regressor_type=self.sequence_regressor,
                    alpha=self.sequence_alpha,
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                mean_rmse = np.mean(rmse_values)
            elif self.baseline_type == "linear":
                linearreg = LinearRegressor(
                    X.shape,
                    self.fh,
                    self.feature_list,
                    regressor_type=self.sequence_regressor,
                    alpha=self.sequence_alpha,
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                mean_rmse = np.mean(rmse_values)
            elif self.baseline_type == "lgbm":
                regressor = GradBooster(X.shape, self.fh, self.feature_list)
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)

                mean_rmse = np.mean(rmse_values)
            else:
                raise InvalidBaselineException

            end_time = time.time()

            self.sequence_plot_x.append(s)
            self.sequence_plot_y.append(mean_rmse)

            execution_time = end_time - start_time
            self.sequence_plot_time.append(execution_time)

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

        return mean_rmse

    def determine_best_s(self):
        try:
            self.clear_sequence_plot()
            best_s = 0
            max_rmse = np.inf

            printProgressBar(
                0,
                self.sequence_n_trials + 1,
                prefix=" Sequence Progress:",
                suffix="Complete",
                length=50,
            )

            if self.baseline_type == "gnn":
                trainer = Trainer(
                    architecture="trans", hidden_dim=32, lr=1e-3, subset=self.subset
                )
            elif self.baseline_type == "cnn":
                trainer = CNNTrainer(subset=self.subset, test_shuffle=False)

            for s in range(1, self.sequence_n_trials + 1):
                # processor = DataProcessor(self.data)
                self.processor.upload_data(self.data)
                X, y = self.processor.preprocess(s, self.fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = self.processor.train_val_test_split(
                    X, y
                )
                start_time = time.time()
                if self.baseline_type == "simple-linear":
                    linearreg = SimpleLinearRegressor(
                        X.shape,
                        self.fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                    rmse_not_normalized = linearreg.get_rmse(
                        y_hat, y_test, normalize=False
                    )
                    print(rmse_not_normalized)
                    mean_rmse = np.mean(rmse_values)

                    linearreg.save_prediction_tensor(y_hat)

                elif self.baseline_type == "linear":
                    linearreg = LinearRegressor(
                        X.shape,
                        self.fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                    rmse_not_normalized = linearreg.get_rmse(
                        y_hat, y_test, normalize=False
                    )
                    mean_rmse = np.mean(rmse_values)
                    linearreg.save_prediction_tensor(y_hat)
                elif self.baseline_type == "lgbm":
                    regressor = GradBooster(X.shape, self.fh, self.feature_list)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)

                    rmse_not_normalized = regressor.get_rmse(
                        y_hat, y_test, normalize=False
                    )
                    mean_rmse = np.mean(rmse_values)
                    # regressor.save_prediction_tensor(y_hat)
                elif self.baseline_type == "gnn":
                    cfg.FH = self.fh
                    cfg.INPUT_SIZE = s
                    cfg.GRAPH_CELLS = self.best_layer
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    rmse_values, y_hat_normalized = trainer.evaluate(
                        "test", verbose=self.gnn_verbose, inverse_norm=False
                    )
                    rmse_values = rmse_values[0]
                    rmse_not_normalized, y_hat_real = trainer.evaluate(
                        "test", verbose=self.gnn_verbose
                    )
                    rmse_not_normalized = rmse_not_normalized[0]

                    if not(os.path.isdir(f'./{self.baseline_type}')):
                        os.mkdir(f'./{self.baseline_type}')

                    torch.save(
                        trainer.model.state_dict(),
                        f"./{self.baseline_type}/model_state_{self.baseline_type}_s{s}.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_normalized,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_s_{s}_norm.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_real,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_s_{s}_real.pt",
                    )

                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "cnn":
                    cfg.FH = self.fh
                    cfg.INPUT_SIZE = s
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    rmse_values, y_hat_normalized = trainer.evaluate(
                        "test", verbose=self.gnn_verbose, inverse_norm=False
                    )
                    rmse_values = rmse_values[0]
                    rmse_not_normalized, y_hat_real = trainer.evaluate(
                        "test", verbose=self.gnn_verbose
                    )
                    rmse_not_normalized = rmse_not_normalized[0]
                    mean_rmse = np.mean(rmse_values)

                    if not(os.path.isdir(f'./{self.baseline_type}')):
                        os.mkdir(f'./{self.baseline_type}')

                    torch.save(
                        trainer.model.state_dict(),
                        f"./{self.baseline_type}/model_state_{self.baseline_type}_s{s}.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_normalized,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_s_{s}_norm.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_real,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_s_{s}_real.pt",
                    )
                else:
                    raise InvalidBaselineException

                end_time = time.time()

                self.sequence_plot_x.append(s)
                self.sequence_plot_y.append(mean_rmse)

                self.not_normalized_plot_sequence[s] = rmse_not_normalized

                execution_time = end_time - start_time
                self.sequence_plot_time.append(execution_time)

                if mean_rmse < max_rmse:
                    max_rmse = mean_rmse
                    best_s = s

                printProgressBar(
                    s,
                    self.sequence_n_trials + 1,
                    prefix="Sequence Progress:",
                    suffix="Complete",
                    length=50,
                )

            self.best_s = best_s

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

    def objective(self, trial):
        try:
            # processor = DataProcessor(self.data)
            self.processor.upload_data(self.data)
            X, y = self.processor.preprocess(
                input_size=self.best_s, fh=self.fh, use_neighbours=self.use_neighbours
            )
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ) = self.processor.train_val_test_split(X, y, split_type=0)

            if self.baseline_type == "simple-linear":
                alpha = trial.suggest_float("alpha", 0.1, self.max_alpha, log=True)
                regressor_type = trial.suggest_categorical(
                    "regressor_type", self.regressors
                )

                # print(f"alpha: {alpha}")
                linearreg = SimpleLinearRegressor(
                    X.shape,
                    self.fh,
                    self.feature_list,
                    regressor_type=regressor_type,
                    alpha=alpha,
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_val, y_val)
                rmse_values = linearreg.get_rmse(y_hat, y_val, normalize=True)
                mean_rmse = np.mean(rmse_values)
            elif self.baseline_type == "linear":
                alpha = trial.suggest_float("alpha", 0.1, self.max_alpha, log=True)
                regressor_type = trial.suggest_categorical(
                    "regressor_type", self.regressors
                )
                linearreg = LinearRegressor(
                    X.shape,
                    self.fh,
                    self.feature_list,
                    regressor_type=regressor_type,
                    alpha=alpha,
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_val, y_val)
                rmse_values = linearreg.get_rmse(y_hat, y_val, normalize=True)
                mean_rmse = np.mean(rmse_values)
            elif self.baseline_type == "lgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 40),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-4, 0.3, log=True
                    ),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-3, 0.5, log=True
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.5, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 25, 50),
                }
                regressor = GradBooster(X.shape, self.fh, self.feature_list, **params)
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_val, y_val)
                rmse_values = regressor.get_rmse(y_hat, y_val, normalize=True)
                mean_rmse = np.mean(rmse_values)
            elif self.baseline_type == "gnn" or self.baseline_type == "cnn":
                print("HPO not implemented for neural nets")
                return 0
            else:
                raise InvalidBaselineException

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

        return mean_rmse

    def fh_objective(self, trial, max_fh=5):
        try:
            fh = trial.suggest_int("fh", 1, max_fh)

            # processor = DataProcessor(self.data)
            self.processor.upload_data(self.data)
            X, y = self.processor.preprocess(self.best_s, fh, self.use_neighbours)
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)
            start_time = time.time()
            if self.baseline_type == "simple-linear":
                linearreg = SimpleLinearRegressor(
                    X.shape, fh, self.feature_list, **self.params
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                mean_rmse = np.mean(rmse_values)
            elif self.baseline_type == "linear":
                linearreg = LinearRegressor(
                    X.shape, fh, self.feature_list, **self.params
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                mean_rmse = np.mean(rmse_values)
            elif self.baseline_type == "lgbm":
                regressor = GradBooster(X.shape, fh, self.feature_list, **self.params)
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)

                mean_rmse = np.mean(rmse_values)
            else:
                raise InvalidBaselineException

            end_time = time.time()
            self.fh_plot_x.append(fh)
            self.fh_plot_y.append(mean_rmse)
            execution_time = end_time - start_time
            self.fh_plot_time.append(execution_time)

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

        return mean_rmse

    def determine_best_fh(self):
        try:
            self.clear_fh_plot()
            best_fh = 0
            max_rmse = np.inf
            printProgressBar(
                0,
                self.sequence_n_trials + 1,
                prefix=" Forcasting Horizon Progress:",
                suffix="Complete",
                length=50,
            )

            if self.baseline_type == "gnn":
                trainer = Trainer(
                    architecture="trans", hidden_dim=32, lr=1e-3, subset=self.subset
                )
            elif self.baseline_type == "cnn":
                trainer = CNNTrainer(subset=self.subset, test_shuffle=False)

            for fh in range(1, self.fh_n_trials + 1):
                # processor = DataProcessor(self.data)
                self.processor.upload_data(self.data)
                X, y = self.processor.preprocess(self.best_s, fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = self.processor.train_val_test_split(
                    X, y
                )
                start_time = time.time()
                if self.baseline_type == "simple-linear":
                    linearreg = SimpleLinearRegressor(
                        X.shape,
                        fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                    rmse_not_normalized = linearreg.get_rmse(
                        y_hat, y_test, normalize=False
                    )
                    mean_rmse = np.mean(rmse_values)
                    linearreg.save_prediction_tensor(y_hat)

                elif self.baseline_type == "linear":
                    linearreg = LinearRegressor(
                        X.shape,
                        fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
                    rmse_not_normalized = linearreg.get_rmse(
                        y_hat, y_test, normalize=False
                    )
                    mean_rmse = np.mean(rmse_values)
                    linearreg.save_prediction_tensor(y_hat)
                elif self.baseline_type == "lgbm":
                    regressor = GradBooster(X.shape, fh, self.feature_list)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
                    rmse_not_normalized = regressor.get_rmse(
                        y_hat, y_test, normalize=False
                    )
                    mean_rmse = np.mean(rmse_values)
                    # regressor.save_prediction_tensor(y_hat)
                elif self.baseline_type == "gnn":
                    cfg.FH = fh
                    cfg.INPUT_SIZE = self.best_s
                    cfg.GRAPH_CELLS = self.best_layer
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    rmse_values, y_hat_normalized = trainer.evaluate(
                        "test", verbose=self.gnn_verbose, inverse_norm=False
                    )
                    rmse_values = rmse_values[0]
                    # rmse_values, _ = trainer.autoreg_evaluate("test", fh=fh, verbose=False)
                    rmse_not_normalized, y_hat_real = trainer.evaluate(
                        "test", verbose=self.gnn_verbose
                    )
                    rmse_not_normalized = rmse_not_normalized[0]
                    mean_rmse = np.mean(rmse_values)

                    if not(os.path.isdir(f'./{self.baseline_type}')):
                        os.mkdir(f'./{self.baseline_type}')

                    torch.save(
                        trainer.model.state_dict(),
                        f"./{self.baseline_type}/model_state_{self.baseline_type}_fh_{fh}.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_normalized,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_fh_{fh}_norm.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_real,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_fh_{fh}_real.pt",
                    )

                elif self.baseline_type == "cnn":
                    # trainer = CNNTrainer(subset=self.subset)
                    cfg.FH = fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    rmse_values, y_hat_normalized = trainer.evaluate(
                        "test", verbose=self.gnn_verbose, inverse_norm=False
                    )
                    rmse_values = rmse_values[0]
                    rmse_not_normalized, y_hat_real = trainer.evaluate(
                        "test", verbose=self.gnn_verbose
                    )
                    rmse_not_normalized = rmse_not_normalized[0]
                    mean_rmse = np.mean(rmse_values)

                    if not(os.path.isdir(f'./{self.baseline_type}')):
                        os.mkdir(f'./{self.baseline_type}')

                    torch.save(
                        trainer.model.state_dict(),
                        f"./{self.baseline_type}/model_state_{self.baseline_type}_fh_{fh}.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_normalized,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_fh_{fh}_norm.pt",
                    )

                    trainer.save_prediction_tensor(
                        y_hat_real,
                        f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_fh_{fh}_real.pt",
                    )
                else:
                    raise InvalidBaselineException

                end_time = time.time()

                self.fh_plot_x.append(fh)
                self.fh_plot_y.append(mean_rmse)

                self.not_normalized_plot_fh[fh] = rmse_not_normalized

                execution_time = end_time - start_time
                self.fh_plot_time.append(execution_time)

                if mean_rmse < max_rmse:
                    max_rmse = mean_rmse
                    best_fh = fh

                printProgressBar(
                    fh,
                    self.fh_n_trials + 1,
                    prefix="Forcasting Horizon Progress:",
                    suffix="Complete",
                    length=50,
                )

            self.best_fh = best_fh

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

    def run_sequence_study(self):
        self.clear_sequence_plot()
        study = optuna.create_study(direction="minimize")
        study.optimize(self.sequence_objective, n_trials=self.sequence_n_trials)
        if self.verbosity == False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.best_s = study.best_params["s"]
        print("Sequence length study finished.")

    def run_study(self):
        self.clear_params()
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        if self.verbosity == False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.params = study.best_params
        print("Parameter study finished.")

    def run_fh_study(self):
        self.clear_fh_plot()
        study = optuna.create_study(direction="minimize")
        study.optimize(self.fh_objective, n_trials=self.sequence_n_trials)
        if self.verbosity == False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.best_fh = study.best_params["fh"]
        print("Forcasting horizon study finished.")

    def run_full_study(self):
        self.determine_best_s()
        # self.best_s = 3
        if self.baseline_type not in ["gnn", "cnn"]:
            self.run_study()
        self.determine_best_fh()
        self.collect_metrics()
        # self.test_scalers()
        self.write_params_to_json()
        self.write_plots_to_json()

    def report(self):
        self.plot_sequence()
        print(f"Best s => {self.best_s}")
        self.print_parameters()
        self.plot_fh()
        print(f"Best fh=> {self.best_fh}")

    def run_and_report(self):
        self.run_full_study()
        self.report()

    def plot_sequence(self):
        plt.scatter(self.sequence_plot_x, self.sequence_plot_y)
        plt.title("Sequence length")
        plt.xlabel("s")
        plt.ylabel("mean_rmse")
        plt.show()

    def plot_fh(self):
        plt.scatter(self.sequence_plot_x, self.sequence_plot_y)
        plt.title("Forcasting horizon")
        plt.xlabel("fh")
        plt.ylabel("mean_rmse")
        plt.show()

    def print_parameters(self):
        print("Best parameters:")

        for key, value in self.params.items():
            print(key, ":", value)

    def write_params_to_json(self):
        file_name = self.baseline_type + "-params.json"
        with open(file_name, "w") as outfile:
            json.dump(self.params, outfile)

    def collect_metrics(self, model=None):
        try:
            self.processor.upload_data(self.data)
            X, y = self.processor.preprocess(
                input_size=self.best_s,
                fh=self.best_fh,
                use_neighbours=self.use_neighbours,
            )
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(
                X, y, split_type=2
            )
            if self.baseline_type == "simple-linear":
                linearreg = SimpleLinearRegressor(
                    X.shape, self.best_fh, self.feature_list, **self.params
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=False)
                mae_values = linearreg.get_mae(y_hat, y_test, normalize=False)
                if not(os.path.isdir(f'./{self.baseline_type}')):
                    os.mkdir(f'./{self.baseline_type}')
                linearreg.save_prediction_tensor(y_hat, path=f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_best.pt")
            elif self.baseline_type == "linear":
                linearreg = LinearRegressor(
                    X.shape, self.fh, self.feature_list, **self.params
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=False)
                mae_values = linearreg.get_mae(y_hat, y_test, normalize=False)
                if not(os.path.isdir(f'./{self.baseline_type}')):
                    os.mkdir(f'./{self.baseline_type}')
                linearreg.save_prediction_tensor(y_hat, path=f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_best.pt")
            elif self.baseline_type == "lgbm":
                regressor = GradBooster(
                    X.shape, self.best_fh, self.feature_list, **self.params
                )
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                rmse_values = regressor.get_rmse(y_hat, y_test, normalize=False)
                mae_values = regressor.get_mae(y_hat, y_test, normalize=False)
                # if not(os.path.isdir(f'./{self.baseline_type}')):
                #     os.mkdir(f'./{self.baseline_type}')
                regressor.save_prediction_tensor(y_hat, path=f"D:\Piotr\inzynierka\meteoapp-data\data\pred\prediction_tensor_{self.baseline_type}_best.pt")
            elif self.baseline_type == "gnn":
                trainer = Trainer(
                        architecture="trans", hidden_dim=32, lr=1e-3, subset=self.subset
                    )
                if model is None:
                    
                    cfg.FH = self.fh
                    cfg.INPUT_SIZE = self.best_s
                    cfg.GRAPH_CELLS = self.best_layer
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    

                    if not(os.path.isdir(f'./{self.baseline_type}')):
                        os.mkdir(f'./{self.baseline_type}')

                    torch.save(trainer.model.state_dict(), f"./{self.baseline_type}/model_state_{self.baseline_type}_best.pt")

                    trainer.save_prediction_tensor(y_hat_normalized, f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_best_norm.pt")

                    trainer.save_prediction_tensor(y_hat_real, f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_best_real.pt")
                else:
                    trainer.load_model(model)

                rmse_values, y_hat_normalized = trainer.evaluate(
                            "test", verbose=self.gnn_verbose, inverse_norm=False
                        )
                rmse_values = rmse_values[0]
                rmse_not_normalized, y_hat_real = trainer.evaluate(
                    "test", verbose=self.gnn_verbose
                )
                rmse_not_normalized = rmse_not_normalized[0]
                
                    

            elif self.baseline_type == "cnn":
                trainer = CNNTrainer(subset=self.subset, test_shuffle=False)
                if model is None:
                    cfg.FH = self.fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    
                else:
                    trainer.load_model(model)

                rmse_values, _ = trainer.evaluate(
                        "test", verbose=self.gnn_verbose, inverse_norm=True
                    )
                rmse_values = rmse_values[0]
                mae_values = rmse_values[1]

            else:
                raise InvalidBaselineException

            self.metrics = rmse_values
            self.metrics_mae = mae_values

            print("Metrics collected.", rmse_values)

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

    def write_plots_to_json(self):
        file_name = f"modelsplots.json"
        data = {}

        # Load existing data from file
        try:
            with open(file_name, "r") as infile:
                data = json.load(infile)
        except FileNotFoundError:
            pass

        # Update data with new arrays
        data[self.baseline_type] = {
            "sequence_plot_x": self.sequence_plot_x,
            "sequence_plot_y": self.sequence_plot_y,
            "sequence_plot_time": self.sequence_plot_time,
            "fh_plot_x": self.fh_plot_x,
            "fh_plot_y": self.fh_plot_y,
            "fh_plot_time": self.fh_plot_time,
            "metrics": self.metrics,
            "metrics_for_scalers": self.metrics_for_scalers,
            "not_normalized_plot_sequence": self.not_normalized_plot_sequence,
            "not_normalized_plot_fh": self.not_normalized_plot_fh,
            "month_error": self.month_error,
            "gnn_alpha_plot_x": self.gnn_alpha_plot_x,
            "gnn_alpha_plot_y": self.gnn_alpha_plot_y,
            "gnn_cell_plot_x": self.gnn_cell_plot_x,
            "gnn_cell_plot_y": self.gnn_cell_plot_y,
        }

        # Write data to file
        with open(file_name, "w") as outfile:
            json.dump(data, outfile)

        if not(os.path.isdir(f'./{self.baseline_type}')):
                os.mkdir(f'./{self.baseline_type}')

        np.save(f'./{self.baseline_type}/error_maps.npy' ,self.error_map)

    def test_scalers(self, model=None):
        try:
            for scaler in self.scalers:
                self.processor.upload_data(self.data)
                X, y = self.processor.preprocess(
                    self.best_s, self.best_fh, self.use_neighbours
                )
                X_train, X_test, y_train, y_test = self.processor.train_val_test_split(
                    X, y
                )
                start_time = time.time()
                if self.baseline_type == "simple-linear":
                    linearreg = SimpleLinearRegressor(
                        X.shape,
                        self.best_fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                        scaler_type=scaler,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                    mean_rmse = np.mean(rmse_values)

                elif self.baseline_type == "linear":
                    linearreg = LinearRegressor(
                        X.shape,
                        self.best_fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                        scaler_type=scaler,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "lgbm":
                    regressor = GradBooster(
                        X.shape, self.best_fh, self.feature_list, scaler_type=scaler
                    )
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)

                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "gnn":
                    trainer = Trainer(
                        architecture="trans", hidden_dim=32, lr=1e-3, subset=self.subset
                    )
                    if model is None:
                        cfg.FH = self.best_fh
                        cfg.INPUT_SIZE = self.best_s
                        cfg.GRAPH_CELLS = self.best_layer
                        cfg.SCALER_TYPE = scaler
                        trainer.update_config(cfg)
                        trainer.train(num_epochs=self.num_epochs)
                    else:
                        trainer.load_model(model)

                    rmse_values, _ = trainer.evaluate(
                        "test", verbose=self.gnn_verbose, inverse_norm=False
                    )
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "cnn":
                    trainer = CNNTrainer(subset=self.subset, test_shuffle=False)
                    if model is None:
                        cfg.FH = self.best_fh
                        cfg.INPUT_SIZE = self.best_s
                        cfg.SCALER_TYPE = scaler
                        trainer.update_config(cfg)
                        trainer.train(self.num_epochs)
                    else:
                        trainer.load_model(model)
                    rmse_values, _ = trainer.evaluate(
                        "test", verbose=self.gnn_verbose, inverse_norm=False
                    )
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                else:
                    raise InvalidBaselineException

                end_time = time.time()

                execution_time = end_time - start_time

                self.metrics_for_scalers[scaler] = {
                    "rmse": mean_rmse,
                    "execution_time": execution_time,
                }

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

    def monthly_error(self, model=None):
        try:
            months_days = {
                1: (1, 31),
                2: (32, 59),
                3: (60, 90),
                4: (91, 120),
                5: (121, 151),
                6: (152, 181),
                7: (182, 212),
                8: (213, 243),
                9: (244, 273),
                10: (274, 304),
                11: (305, 334),
                12: (335, 365),
            }

            months_names = {
                1: "January",
                2: "February",
                3: "March",
                4: "April",
                5: "May",
                6: "June",
                7: "July",
                8: "August",
                9: "September",
                10: "October",
                11: "November",
                12: "December",
            }

            if self.baseline_type == "gnn":
                trainer = Trainer(
                    architecture="trans",
                    hidden_dim=32,
                    lr=1e-3,
                    subset=self.subset,
                    test_shuffle=False,
                )
            elif self.baseline_type == "cnn":
                trainer = CNNTrainer(subset=self.subset, test_shuffle=False)

            # for month in range(1, 13):
            self.processor.upload_data(self.data)
            X, y = self.processor.preprocess(
                self.best_s, self.best_fh, self.use_neighbours
            )
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(
                X, y, split_type=2, test_shuffle=False
            )
            # start_time = time.time()
            if self.baseline_type == "simple-linear":
                linearreg = SimpleLinearRegressor(
                    X.shape,
                    self.best_fh,
                    self.feature_list,
                    regressor_type=self.sequence_regressor,
                    alpha=self.sequence_alpha,
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                for month in range(1, 13):
                    if month == 1:
                        begin = months_days[month][0] - 1
                        end = months_days[month + 1][0] * 4
                    elif month == 12:
                        begin = months_days[month][0] * 4 + 1
                        end = months_days[month][1] * 4 + 1
                    else:
                        begin = months_days[month][0] * 4 + 1
                        end = months_days[month + 1][0] * 4 + 1
                    rmse_values = linearreg.get_rmse(
                        y_hat,
                        y_test,
                        normalize=True,
                        begin=begin,
                        end=end,
                    )
                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse

            elif self.baseline_type == "linear":
                linearreg = LinearRegressor(
                    X.shape,
                    self.best_fh,
                    self.feature_list,
                    regressor_type=self.sequence_regressor,
                    alpha=self.sequence_alpha,
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                for month in range(1, 13):
                    if month == 1:
                        begin = months_days[month][0] - 1
                        end = months_days[month + 1][0] * 4
                    elif month == 12:
                        begin = months_days[month][0] * 4 + 1
                        end = months_days[month][1] * 4 + 1
                    else:
                        begin = months_days[month][0] * 4 + 1
                        end = months_days[month + 1][0] * 4 + 1
                    rmse_values = linearreg.get_rmse(
                        y_hat,
                        y_test,
                        normalize=True,
                        begin=begin,
                        end=end,
                    )
                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse

            elif self.baseline_type == "lgbm":
                regressor = GradBooster(X.shape, self.best_fh, self.feature_list)
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                for month in range(1, 13):
                    if month == 1:
                        begin = months_days[month][0] - 1
                        end = months_days[month + 1][0] * 4
                    elif month == 12:
                        begin = months_days[month][0] * 4 + 1
                        end = months_days[month][1] * 4 + 1
                    else:
                        begin = months_days[month][0] * 4 + 1
                        end = months_days[month + 1][0] * 4 + 1
                    rmse_values = regressor.get_rmse(
                        y_hat,
                        y_test,
                        normalize=True,
                        begin=begin,
                        end=end,
                    )
                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse

            elif self.baseline_type == "gnn":
                if  model is None:
                    cfg.FH = self.best_fh
                    cfg.INPUT_SIZE = self.best_s
                    cfg.GRAPH_CELLS = self.best_layer
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                else:
                    trainer.load_model(model)
                for month in range(1, 13):
                    rmse_values, _ = trainer.evaluate(
                        "test",
                        verbose=self.gnn_verbose,
                        inverse_norm=False,
                        begin=months_days[month][0],
                        end=months_days[month][1],
                    )
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse

            elif self.baseline_type == "cnn":
                if model is None:
                    cfg.FH = self.best_fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                else:
                    trainer.load_model(model)
                for month in range(1, 13):
                    rmse_values, _ = trainer.evaluate(
                        "test",
                        verbose=self.gnn_verbose,
                        inverse_norm=False,
                        begin=months_days[month][0],
                        end=months_days[month][1],
                    )
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse

            else:
                raise InvalidBaselineException
        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )

    def gnn_alpha_plot(self, model=None):
        self.clear_alpha_plot()
        best_alpha = 0
        max_rmse = np.inf
        alpha = 0
        printProgressBar(
            0, alpha, prefix=" Alpha Progress:", suffix="Complete", length=50
        )

        trainer = Trainer(
            architecture="trans", hidden_dim=32, lr=1e-3, subset=self.subset
        )

        for alpha in range(0.1, 1, 0.1):
            start_time = time.time()
            if model is None:
                cfg.FH = self.best_fh
                cfg.INPUT_SIZE = self.best_s
                cfg.GRAPH_CELLS = self.best_layer
                trainer.update_config(cfg)
                trainer.train(num_epochs=self.num_epochs)
            else:
                trainer.load_model(model)
            # TODO: tigge and gnn mix

            rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
            rmse_values = rmse_values[0]

            mean_rmse = np.mean(rmse_values)

            end_time = time.time()

            self.gnn_alpha_plot_x.append(alpha)
            self.gnn_alpha_plot_y.append(mean_rmse)

            execution_time = end_time - start_time
            print(execution_time)

            if mean_rmse < max_rmse:
                max_rmse = mean_rmse
                best_alpha = alpha

            printProgressBar(
                alpha, 1, prefix="Alpha Progress:", suffix="Complete", length=50
            )

        self.best_alpha = best_alpha



    def gnn_layer(self):
        trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3, subset=self.subset)
        best_layer = 2
        max_rmse = np.inf
        for cell in range(2, 10):
            cfg.FH = self.fh
            cfg.INPUT_SIZE = self.best_s
            cfg.GRAPH_CELLS = cell
            trainer.update_config(cfg)
            trainer.train(num_epochs=self.num_epochs)
            rmse_values, y_hat_normalized = trainer.evaluate("test", verbose=self.gnn_verbose, inverse_norm=False)
            rmse_values = rmse_values[0]
            rmse_not_normalized, y_hat_real = trainer.evaluate("test", verbose=self.gnn_verbose)
            rmse_not_normalized = rmse_not_normalized[0]
            mean_rmse = np.mean(rmse_values)
            self.gnn_cell_plot_x.append(cell)
            self.gnn_cell_plot_y.append(mean_rmse)

            if mean_rmse < max_rmse:
                max_rmse = mean_rmse
                best_layer = cell

            if not(os.path.isdir(f'./{self.baseline_type}')):
                os.mkdir(f'./{self.baseline_type}')

            torch.save(trainer.model.state_dict(), f"./{self.baseline_type}/model_state_{self.baseline_type}_cell_{cell}.pt")

            trainer.save_prediction_tensor(y_hat_normalized, f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_cell_{cell}_norm.pt")

            trainer.save_prediction_tensor(y_hat_real, f"./{self.baseline_type}/prediction_tensor_{self.baseline_type}_cell_{cell}_real.pt")
        
        self.best_layer = best_layer



    def error_maps(self, path = None, model=None):
        try:
            self.processor.upload_data(self.data)
            X, y = self.processor.preprocess(self.best_s, self.fh, self.use_neighbours)
            X_train, X_test, y_train, era_data = self.processor.train_val_test_split(
                    X, y
                )
            if self.baseline_type == 'gnn':
                trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3, subset=self.subset)
            elif self.baseline_type == 'cnn':
                trainer = CNNTrainer(subset=self.subset, test_shuffle=False)

            
            if path is None:
                self.processor.upload_data(self.data)
                X, y = self.processor.preprocess(self.best_s,self.fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)
            start_time = time.time()
            if self.baseline_type == "simple-linear":
                if path is None:
                    linearreg = SimpleLinearRegressor(
                        X.shape,
                        self.best_fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                else:
                    y_hat = np.load(path)
                
                y_true = era_data
                min_length = min(y_hat.shape[0], y_true.shape[0])
                y_hat = y_hat[-min_length:]
                y_true = y_true[-min_length:]
                print(y_hat.shape)
                print(y_true.shape)
                y_diff = y_hat - y_true
                y_diff = y_diff**2
                y_diff = np.mean(y_diff, axis=0)
                for i in range(len(self.feature_list)):
                    self.error_map.append(y_diff[...,i,0])



                
                    
            elif self.baseline_type == "linear":
                if path is None:
                    linearreg = LinearRegressor(
                        X.shape,
                        self.fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                else:
                    y_hat = np.load(path)

                y_true = era_data
                min_length = min(y_hat.shape[0], y_true.shape[0])
                y_hat = y_hat[-min_length:]
                y_true = y_true[-min_length:]
                print(y_hat.shape)
                print(y_true.shape)
                y_diff = y_hat - y_true
                y_diff = y_diff**2
                y_diff = np.mean(y_diff, axis=0)
                for i in range(len(self.feature_list)):
                    self.error_map.append(y_diff[...,i,0])
            elif self.baseline_type == "lgbm":
                if path is None:
                    regressor = GradBooster(X.shape, self.fh, self.feature_list)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                else:
                    y_hat = np.load(path)
                
                y_true = era_data
                min_length = min(y_hat.shape[0], y_true.shape[0])
                y_hat = y_hat[-min_length:]
                y_true = y_true[-min_length:]
                print(y_hat.shape)
                print(y_true.shape)
                y_diff = y_hat - y_true
                y_diff = y_diff**2
                y_diff = np.mean(y_diff, axis=0)
                for i in range(len(self.feature_list)):
                    self.error_map.append(y_diff[...,i,0])
            elif self.baseline_type == "gnn":
                if path is None:
                    if model is None:
                        cfg.FH  = self.fh
                        cfg.INPUT_SIZE = self.best_s
                        cfg.GRAPH_CELLS = self.best_layer
                        trainer.update_config(cfg)
                        trainer.train(num_epochs=self.num_epochs)
                    else:
                        trainer.load_model(model)
                    rmse_values, y_hat_normalized = trainer.evaluate("test", verbose=self.gnn_verbose, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    # rmse_values, _ = trainer.autoreg_evaluate("test", fh=fh, verbose=False)                    
                    rmse_not_normalized, y_hat_real = trainer.evaluate("test", verbose=self.gnn_verbose)
                    rmse_not_normalized = rmse_not_normalized[0]
                else:
                    y_hat_not_normalized = np.load(path)
                
                y_true = era_data
                min_length = min(y_hat.shape[0], y_true.shape[0])
                y_hat = y_hat[-min_length:]
                y_true = y_true[-min_length:]
                print(y_hat.shape)
                print(y_true.shape)
                y_diff = y_hat - y_true
                y_diff = y_diff**2
                y_diff = np.mean(y_diff, axis=0)
                for i in range(len(self.feature_list)):
                    self.error_map.append(y_diff[...,i,0])


            elif self.baseline_type == "cnn":
                # trainer = CNNTrainer(subset=self.subset)
                if path is None:
                    if model is None:
                        cfg.FH  = self.fh
                        cfg.INPUT_SIZE = self.best_s
                        trainer.update_config(cfg)
                        trainer.train(self.num_epochs)
                    else:
                        trainer.load_model(model)
                    rmse_values, y_hat_normalized = trainer.evaluate("test", verbose=self.gnn_verbose, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    rmse_not_normalized, y_hat_real = trainer.evaluate("test", verbose=self.gnn_verbose)
                    rmse_not_normalized = rmse_not_normalized[0]
                    mean_rmse = np.mean(rmse_values)
                
                else:
                    y_hat_not_normalized = np.load(path)
                
                y_true = era_data
                min_length = min(y_hat.shape[0], y_true.shape[0])
                y_hat = y_hat[-min_length:]
                y_true = y_true[-min_length:]
                print(y_hat.shape)
                print(y_true.shape)
                y_diff = y_hat - y_true
                y_diff = y_diff**2
                y_diff = np.mean(y_diff, axis=0)
                for i in range(len(self.feature_list)):
                    self.error_map.append(y_diff[...,i,0])
            else:
                raise InvalidBaselineException
            
            end_time = time.time()

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )   
