import xarray as xr
import cfgrib
import numpy as np
import optuna
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from sklearn.metrics import mean_squared_error

import json
import time
import sys

sys.path.append("..")

from models.data_processor import DataProcessor
from models.linear_reg.linear_regressor import LinearRegressor
from models.linear_reg.simple_linear_regressor import SimpleLinearRegressor
from models.grad_boost.grad_booster import GradBooster
from models.gnn.trainer import Trainer
from models.cnn.trainer import Trainer as CNNTrainer
from models.config import config as cfg

from utils.progress_bar import printProgressBar


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
        # max_sequence_length = 15,
        sequence_length=1,
        sequence_n_trials=15,
        sequence_alpha=5,
        sequence_regressor="ridge",
        fh_n_trials=15,
        max_alpha=10,
        num_epochs=3,
    ):
        

        self.baseline_type = baseline_type
        self.n_trials = n_trials
        self.use_neighbours = use_neighbours
        # self.max_sequence_length = max_sequence_length
        self.sequence_n_trials = sequence_n_trials
        self.sequence_alpha = sequence_alpha
        self.sequence_regressor = sequence_regressor
        self.fh_n_trials = fh_n_trials
        self.processor = DataProcessor()
        self.data, self.feature_list = self.processor.data, self.processor.feature_list
        self.best_s = sequence_length
        self.fh = 1
        self.best_fh = self.fh
        self.regressors = ["lasso", "ridge", "elastic_net"]

        self.subset = 1
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

        self.metrics_for_scalers = {}


        self.not_normalized_plot_sequence = {}
        self.not_normalized_plot_fh = {}


        self.month_error = {}

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

            printProgressBar(0, self.sequence_n_trials + 1, prefix = ' Sequence Progress:', suffix = 'Complete', length = 50)


            if self.baseline_type == 'gnn':
                trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3, subset=self.subset)
            elif self.baseline_type == 'cnn':
                trainer = CNNTrainer(subset=self.subset)

            for s in range(1, self.sequence_n_trials + 1):
                # processor = DataProcessor(self.data)
                self.processor.upload_data(self.data)
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

                    rmse_not_normalized = linearreg.get_rmse(y_hat, y_test, normalize=False)
                    print(rmse_not_normalized)
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
     
                    rmse_not_normalized = linearreg.get_rmse(y_hat, y_test, normalize=False)
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "lgbm":
                    regressor = GradBooster(X.shape, self.fh, self.feature_list)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
 
                    rmse_not_normalized = regressor.get_rmse(y_hat, y_test, normalize=False)
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "gnn":
                    
                    cfg.FH  = self.fh
                    cfg.INPUT_SIZE = s
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    rmse_not_normalized, _ = trainer.evaluate("test", verbose=False)
                    rmse_not_normalized = rmse_not_normalized[0]

                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "cnn":
                    
                    cfg.FH  = self.fh
                    cfg.INPUT_SIZE = s
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    rmse_not_normalized, _ = trainer.evaluate("test", verbose=False)
                    rmse_not_normalized = rmse_not_normalized[0]
                    mean_rmse = np.mean(rmse_values)
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

                printProgressBar(s, self.sequence_n_trials + 1, prefix = 'Sequence Progress:', suffix = 'Complete', length = 50)

            self.best_s = best_s

        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )   

    def objective(self, trial):
        try:
            # processor = DataProcessor(self.data)
            self.processor.upload_data(self.data)
            X, y = self.processor.preprocess(input_size=self.best_s, fh=self.fh, use_neighbours=self.use_neighbours)
            X_train, X_val, X_test, y_train, y_val, y_test = self.processor.train_val_test_split(X, y, split_type=0)

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
            printProgressBar(0, self.sequence_n_trials + 1, prefix = ' Forcasting Horizon Progress:', suffix = 'Complete', length = 50)

            if self.baseline_type == 'gnn':
                trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3, subset=self.subset)
            elif self.baseline_type == 'cnn':
                trainer = CNNTrainer(subset=self.subset)

            for fh in range(1, self.fh_n_trials + 1):
                # processor = DataProcessor(self.data)
                self.processor.upload_data(self.data)
                X, y = self.processor.preprocess(self.best_s,fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)
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

                    rmse_not_normalized = linearreg.get_rmse(y_hat, y_test, normalize=False)
                    mean_rmse = np.mean(rmse_values)

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
                    rmse_not_normalized = linearreg.get_rmse(y_hat, y_test, normalize=False)
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "lgbm":
                    regressor = GradBooster(X.shape, fh, self.feature_list)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
                    rmse_not_normalized = regressor.get_rmse(y_hat, y_test, normalize=False)
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "gnn":
                    cfg.FH  = fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    # rmse_values, _ = trainer.autoreg_evaluate("test", fh=fh, verbose=False)                    
                    rmse_not_normalized, _ = trainer.evaluate("test", verbose=False)
                    rmse_not_normalized = rmse_not_normalized[0]
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "cnn":
                    # trainer = CNNTrainer(subset=self.subset)
                    cfg.FH  = fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    rmse_not_normalized, _ = trainer.evaluate("test", verbose=False)
                    rmse_not_normalized = rmse_not_normalized[0]
                    mean_rmse = np.mean(rmse_values)
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

                printProgressBar(fh, self.fh_n_trials + 1, prefix = 'Forcasting Horizon Progress:', suffix = 'Complete', length = 50)

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
        if self.baseline_type not in ['gnn', 'cnn']:
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


    def collect_metrics(self):
        try:
            self.processor.upload_data(self.data)
            X, y = self.processor.preprocess(input_size=self.best_s, fh=self.best_fh, use_neighbours=self.use_neighbours)
            X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y, split_type=2)
            if self.baseline_type == "simple-linear":
                linearreg = SimpleLinearRegressor(
                    X.shape,
                    self.best_fh,
                    self.feature_list,
                    **self.params
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
            elif self.baseline_type == "linear":
                linearreg = LinearRegressor(
                    X.shape,
                    self.fh,
                    self.feature_list,
                    **self.params
                )
                linearreg.train(X_train, y_train, normalize=True)
                y_hat = linearreg.predict_(X_test, y_test)
                rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
            elif self.baseline_type == "lgbm":
                regressor = GradBooster(X.shape, self.best_fh, self.feature_list, **self.params)
                regressor.train(X_train, y_train, normalize=True)
                y_hat = regressor.predict_(X_test, y_test)
                rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
            elif self.baseline_type == "gnn":
                    trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3, subset=self.subset)
                    cfg.FH  = self.fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]
    
            elif self.baseline_type == "cnn":
                    trainer = CNNTrainer(subset=self.subset)
                    cfg.FH  = self.fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]

            else:
                raise InvalidBaselineException

            self.metrics = rmse_values

            print("Metrics collected.", rmse_values)


        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )        



    
    def write_plots_to_json(self):
        file_name = "modelsplots.json"
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
            "month_error": self.month_error 
        }

        # Write data to file
        with open(file_name, "w") as outfile:
            json.dump(data, outfile)


    def test_scalers(self):
        try:
           

            for scaler in self.scalers:
                self.processor.upload_data(self.data)
                X, y = self.processor.preprocess(self.best_s,self.best_fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y)
                start_time = time.time()
                if self.baseline_type == "simple-linear":
                    linearreg = SimpleLinearRegressor(
                        X.shape,
                        self.best_fh,
                        self.feature_list,
                        regressor_type=self.sequence_regressor,
                        alpha=self.sequence_alpha,
                        scaler_type=scaler
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
                        scaler_type=scaler
                    )
                    linearreg.train(X_train, y_train, normalize=True)
                    y_hat = linearreg.predict_(X_test, y_test)
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)

                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "lgbm":
                    regressor = GradBooster(X.shape, self.best_fhfh, self.feature_list, scaler_type=scaler)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True)
    
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "gnn":
                    trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3, subset=self.subset)
                    cfg.FH  = self.best_fh
                    cfg.INPUT_SIZE = self.best_s
                    cfg.SCALER_TYPE = scaler
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                elif self.baseline_type == "cnn":
                    trainer = CNNTrainer(subset=self.subset)
                    cfg.FH  = self.best_fh
                    cfg.INPUT_SIZE = self.best_s
                    cfg.SCALER_TYPE = scaler
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False)
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                else:
                    raise InvalidBaselineException
                
                end_time = time.time()

                
                execution_time = end_time - start_time

                self.metrics_for_scalers[scaler] = {
                    "rmse": mean_rmse,
                    "execution_time": execution_time
                }
                
           
        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )   

    def monthly_error(self):
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


            if self.baseline_type == 'gnn':
                    trainer = Trainer(architecture='trans', hidden_dim=32, lr=1e-3, subset=self.subset, test_shuffle=False)
            elif self.baseline_type == 'cnn':
                trainer = CNNTrainer(subset=self.subset, test_shuffle=False)


            for month in range(1, 13):
                self.processor.upload_data(self.data)
                X, y = self.processor.preprocess(self.best_s,self.best_fh, self.use_neighbours)
                X_train, X_test, y_train, y_test = self.processor.train_val_test_split(X, y, split_type=2, test_shuffle=False)
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
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True, begin=months_days[month][0], end=months_days[month][1]) 

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
                    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True, begin=months_days[month][0], end=months_days[month][1])

                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse
                elif self.baseline_type == "lgbm":
                    regressor = GradBooster(X.shape, self.best_fh, self.feature_list)
                    regressor.train(X_train, y_train, normalize=True)
                    y_hat = regressor.predict_(X_test, y_test)
                    rmse_values = regressor.get_rmse(y_hat, y_test, normalize=True, begin=months_days[month][0], end=months_days[month][1])

                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse
                elif self.baseline_type == "gnn":
                    cfg.FH  = self.best_fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(num_epochs=self.num_epochs)
                    print(months_days[month][0], months_days[month][1])
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False, begin=months_days[month][0], end=months_days[month][1])   
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse
                elif self.baseline_type == "cnn":
                    cfg.FH  = self.best_fh
                    cfg.INPUT_SIZE = self.best_s
                    trainer.update_config(cfg)
                    trainer.train(self.num_epochs)
                    rmse_values, _ = trainer.evaluate("test", verbose=False, inverse_norm=False, begin=months_days[month][0], end=months_days[month][1])
                    rmse_values = rmse_values[0]
                    mean_rmse = np.mean(rmse_values)
                    self.month_error[months_names[month]] = mean_rmse
                else:
                        raise InvalidBaselineException
        except InvalidBaselineException:
            print(
                "Exception occurred: Invalid Baseline, choose between 'linear' , 'simple-linear', 'lgbm', 'gnn' and 'cnn'"
            )   
