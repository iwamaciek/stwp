import xarray as xr
import cfgrib
import numpy as np
import optuna
import sys
import matplotlib.pyplot as plt
from datetime import datetime

from functools import partial
from sklearn.metrics import mean_squared_error
import sys

sys.path.append("..")

import json

from baselines.data_processor import DataProcessor
from baselines.linear_reg.linear_regressor import LinearRegressor
from baselines.linear_reg.simple_linear_regressor import SimpleLinearRegressor
from baselines.lgb.lgb_regressor import LightGBMRegressor


class InvalidBaselineException(Exception):
    "Raised when baseline type in invalid"
    pass


class HPO:
    def __init__(
        self,
        baseline_type,
        n_trials,
        dataset,
        use_neighbours=False,
        sequence_length=1,
        sequence_n_trials=15,
        sequence_alpha=5,
        sequence_regressor="ridge",
        fh_n_trials=15,
        max_alpha=10,
        engine='optuna'
    ):
        self.baseline_type = baseline_type
        self.n_trials = n_trials
        self.use_neighbours = use_neighbours
        # self.max_sequence_length = max_sequence_length
        self.sequence_n_trials = sequence_n_trials
        self.sequence_alpha = sequence_alpha
        self.sequence_regressor = sequence_regressor
        self.fh_n_trials = fh_n_trials
        self.data, self.feature_list = DataProcessor.load_data(dataset)
        self.best_s = sequence_length
        self.fh = 1
        self.best_fh = 10000
        self.regressors = ['lasso', 'ridge', 'elastic_net']
        self.max_alpha = max_alpha
        self.verbosity = True
        self.params = {}
        self.engine = engine

        self.sequence_plot_x = []
        self.sequence_plot_y = []

        self.fh_plot_x = []
        self.fh_plot_y = []

    def run_hpo(self):
        return -1

    def clear_sequence_plot(self):
        self.sequence_plot_x = []
        self.sequence_plot_y = []

    def clear_fh_plot(self):
        self.fh_plot_x = []
        self.fh_plot_y = []

    def clear_params(self):
        self.params = {}

    def sequence_objective(self, trial, max_sequence_length=15):
        try:
            s = trial.suggest_int("s", 1, max_sequence_length)

            processor = DataProcessor(self.data)
            X, y = processor.preprocess(s, self.fh, self.use_neighbours)
            X_train, X_test, y_train, y_test = processor.train_test_split(X, y)

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
                regressor = LightGBMRegressor(X.shape, self.fh, self.feature_list)
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
                mean_rmse = np.mean(rmse_values)
            else:
                raise InvalidBaselineException

            self.sequence_plot_x.append(s)
            self.sequence_plot_y.append(mean_rmse)

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear' and  'lgbm'"
            )

        return mean_rmse

    def determine_best_s(self, max_sequence_lenght=15):
        try:
            self.clear_sequence_plot()
            best_s = 0
            max_rmse = np.inf

            for s in range(1, max_sequence_lenght + 1):
                processor = DataProcessor(self.data)
                X, y = processor.preprocess(s, self.fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = processor.train_test_split(X, y)
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
                    regressor = LightGBMRegressor(X.shape, self.fh, self.feature_list)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
                    mean_rmse = np.mean(rmse_values)
                else:
                    raise InvalidBaselineException



                if(self.verbosity == True):
                    print(f"s: {s} => mean_rmse {mean_rmse}")
                
                self.sequence_plot_x.append(s)
                self.sequence_plot_y.append(mean_rmse)

                if mean_rmse < max_rmse:
                    max_rmse = mean_rmse
                    best_s = s

            self.best_s = best_s
            print(f'best_s : {best_s}')

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear' and  'lgbm'"
            )

    def determine_best_fh(self):
        try:
            max_fh_length = self.best_s + 1
            self.clear_fh_plot()
            best_fh= 0
            max_rmse = np.inf

            for fh in range(1, max_fh_length):
                processor = DataProcessor(self.data)
                X, y = processor.preprocess(self.best_s, fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = processor.train_test_split(X, y)
                if self.baseline_type == "simple-linear":
                    linearreg = SimpleLinearRegressor(
                        X.shape,
                        fh,
                        self.feature_list,
                        **self.params
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "linear":
                    linearreg = LinearRegressor(
                        X.shape,
                        fh,
                        self.feature_list,
                        **self.params
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "lgbm":
                    regressor = LightGBMRegressor(X.shape, fh, self.feature_list, **self.params)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
                    mean_rmse = np.mean(rmse_values)
                else:
                    raise InvalidBaselineException
                

                if(self.verbosity == True):
                    print(f"fh: {fh} => mean_rmse {mean_rmse}")

                self.fh_plot_x.append(fh)
                self.fh_plot_y.append(mean_rmse)

                if mean_rmse < max_rmse:
                    max_rmse = mean_rmse
                    best_fh = fh

            self.best_fh = best_fh
            print(f'best_fh : {best_fh}')

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear' and  'lgbm'"
            )

    def objective(self, trial):
        try:
            processor = DataProcessor(self.data)
            X, y = processor.preprocess(self.best_s, self.fh, self.use_neighbours)
            X_train, X_test, y_train, y_test = processor.train_test_split(X, y)

            if self.baseline_type == "simple-linear":
                alpha = trial.suggest_float("alpha", 0.1, self.max_alpha, log=True)
                regressor_type = trial.suggest_categorical(
                    "regressor_type", self.regressors
                )
                linearreg = SimpleLinearRegressor(
                    X.shape,
                    self.fh,
                    self.feature_list,
                    regressor_type=regressor_type,
                    alpha=alpha,
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
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
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
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
                regressor = LightGBMRegressor(
                    X.shape, self.fh, self.feature_list, **params
                )
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
                mean_rmse = np.mean(rmse_values)
            else:
                raise InvalidBaselineException

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear' and  'lgbm'"
            )

        return mean_rmse

    def fh_objective(self, trial):
        try:
            max_fh = self.best_s + 1
            fh = trial.suggest_int("fh", 1, max_fh)

            processor = DataProcessor(self.data)
            X, y = processor.preprocess(self.best_s, fh, self.use_neighbours)
            X_train, X_test, y_train, y_test = processor.train_test_split(X, y)

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
                regressor = LightGBMRegressor(
                    X.shape, fh, self.feature_list, **self.params
                )
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
                mean_rmse = np.mean(rmse_values)
            else:
                raise InvalidBaselineException

            self.fh_plot_x.append(fh)
            self.fh_plot_y.append(mean_rmse)

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear' and  'lgbm'"
            )

        return mean_rmse

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
        study.optimize(self.fh_objective, n_trials=self.fh_n_trials)
        if self.verbosity == False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.best_fh = study.best_params["fh"]
        print("Forcasting horizon study finished.")

    def run_best_params(self):
        try:
            processor = DataProcessor(self.data)
            X, y = processor.preprocess(self.best_s, self.best_fh, self.use_neighbours)
            X_train, X_test, y_train, y_test = processor.train_test_split(X, y)

            if self.baseline_type == "simple-linear":
                linearreg = SimpleLinearRegressor(
                    X.shape, self.best_fh, self.feature_list, **self.params
                )
                linearreg.train(X_train, y_train, normalize=False)
                _ = linearreg.predict_and_evaluate(X_test, y_test, max_samples=1)
            elif self.baseline_type == "linear":
                linearreg = LinearRegressor(
                    X.shape, self.best_fh, self.feature_list, **self.params
                )
                linearreg.train(X_train, y_train, normalize=False)
                _ = linearreg.predict_and_evaluate(X_test, y_test, max_samples=1)
            elif self.baseline_type == "lgbm":
                regressor = LightGBMRegressor(
                    X.shape, self.best_fh, self.feature_list, **self.params
                )
                regressor.train(X_train, y_train, normalize=False)
                _ = regressor.predict_and_evaluate(X_test, y_test, max_samples=1)
            else:
                raise InvalidBaselineException

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear' and  'lgbm'"
            )

    def run_full_study(self):
        if(self.engine == 'optuna'):
            self.run_sequence_study()
            # self.best_s = 3
            self.run_study()
            self.write_params_to_json()
            self.run_fh_study()
        elif(self.engine == 'greed'):
            self.determine_best_s()
            self.run_study()
            self.write_params_to_json()
            self.determine_best_fh()
        else:
            print('Incorrect engine')
            


    def report(self):
        self.plot_sequence()
        print(f"Best s => {self.best_s}")
        self.print_parameters()
        self.plot_fh()
        print(f"Best fh=> {self.best_fh}")
        self.run_best_params()

    def run_and_report(self):
        self.run_full_study()
        self.report()

    def plot_sequence(self):
        tmp = {self.sequence_plot_x[i] : self.sequence_plot_y[i] for i in range(len(self.sequence_plot_x))}
        tmp = dict(sorted(tmp.items()))
        plt.plot(tmp.keys(), tmp.values())
        plt.title("Sequence length")
        plt.xlabel("s")
        plt.ylabel("mean_rmse")
        plt.show()

    def plot_fh(self):
        tmp = {self.fh_plot_x[i] : self.fh_plot_y[i] for i in range(len(self.fh_plot_x))}
        tmp = dict(sorted(tmp.items()))
        plt.plot(tmp.keys(), tmp.values())
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
