import xarray as xr
import cfgrib
import numpy as np
import optuna
import sys
from datetime import datetime

from functools import partial
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("..")

from baselines.data_processor import DataProcessor
from baselines.linear_reg.linear_regressor import LinearRegressor
from baselines.linear_reg.simple_linear_regressor import SimpleLinearRegressor


class InvalidBaselineException(Exception):
    "Raised when baseline type in invalid"
    pass


def objective(trial, baseline_type, data, feature_list, use_neighbours=False, max_sequence_length=8 , max_alpha=2.1, regressors=['lasso', 'ridge', 'elastic_net']):
    try:
        s = trial.suggest_int('s', 3, max_sequence_length)  
        fh = 1
        alpha = trial.suggest_float('alpha', 0.1, max_alpha, log=True)
        regressor_type = trial.suggest_categorical('regressor_type', regressors)

        processor = DataProcessor(data)
        X, y = processor.preprocess(s, fh, use_neighbours=use_neighbours)
        X_train, X_test, y_train, y_test = processor.train_test_split(X, y)
        
        if(baseline_type == 'simple-linear'):
            linearreg = SimpleLinearRegressor(X.shape, fh, feature_list, regressor_type=regressor_type, alpha=alpha)
            linearreg.train(X_train, y_train, normalize=True)
            y_hat = linearreg.predict_(X_test, y_test)
            rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
            mean_rmse = np.mean(rmse_values)
        elif(baseline_type == 'linear'):
            linearreg = LinearRegressor(X.shape, fh, feature_list, regressor_type=regressor_type, alpha=alpha)
            linearreg.train(X_train, y_train, normalize=True)
            y_hat = linearreg.predict_(X_test, y_test)
            rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
            mean_rmse = np.mean(rmse_values)
        else:
            raise InvalidBaselineException


    except InvalidBaselineException:
        print('Exception occurred: Invalid Baseline, choose between \'linear\' and \'simple-linear\'')

    return mean_rmse




def run_study(baseline_type, n_trials, data, feature_list, objective_function=objective, verbosity=True):
    study = optuna.create_study(direction='minimize')
    objective_func = partial(objective_function, baseline_type=baseline_type, data=data, feature_list=feature_list)
    study.optimize(objective_func, n_trials=n_trials)
    if (verbosity == False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    best_s = study.best_params['s']
    # best_fh = study.best_params['fh']
    best_regressor_type = study.best_params['regressor_type']
    best_alpha = study.best_params['alpha']
    # best_neighbour = study.best_params['use_neighbours']


    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])



    print('Best hyperparameters:')
    print(f"Best sequence_length: {best_s}")
    print(f"Best regressor type: {best_regressor_type}")
    print(f"Best regularization constant: {best_alpha}")
    # print(f"Best use_neighbours value: {best_neighbour}")
    print(f"Params: {trial_with_highest_accuracy.params}")


    return best_s, best_regressor_type, best_alpha




def write_stats_to_file(file_name, baseline_type, n_trials, use_neighbours, best_s, best_regressor_type, best_alpha):
    with open(file_name, 'a') as file:
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:S")
        file.write(current_time)
        file.write("\n")
        file.write(f'Used baseline: {baseline_type}')
        file.write(f'Number of trials: {n_trials}\n')
        if(use_neighbours):
            file.write("Neighbours used: Yes\n")
        else:
            file.write("Neighbours used: No\n")
        file.write('Best hyperparameters:\n')
        file.write(f"Best input window: {best_s}\n")
        file.write(f"Best regressor type: {best_regressor_type}\n")
        file.write(f"Best regularization constant: {best_alpha}\n")
        file.write("_____________________________________________\n")

def run_hpo(data_path, baseline_type, n_trials, use_neighbours, file_name):
    data, feature_list = DataProcessor.load_data(data_path)

    best_s, best_regressor_type, best_alpha = run_study(baseline_type, n_trials, data, feature_list)

    write_stats_to_file(file_name, baseline_type, n_trials, use_neighbours, best_s, best_regressor_type, best_alpha)


if __name__ == "__main__":
    data_path = '../../data2022_full.grib'

    baseline_type = sys.argv[1]
    n_trials = int(sys.argv[2])
    use_neighbours = eval(sys.argv[3])

    file_name = 'hpo_linear_regression_results.txt'

    run_hpo(data_path, baseline_type, n_trials, use_neighbours, file_name)
    
