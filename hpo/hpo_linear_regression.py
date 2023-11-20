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




def load_data(data_path):

    grib_data = cfgrib.open_datasets(data_path)
    surface = grib_data[0] 
    hybrid = grib_data[1] 


    
    t2m = surface.t2m.to_numpy() - 273.15  # -> C
    sp = surface.sp.to_numpy() / 100       # -> hPa
    tcc = surface.tcc.to_numpy()
    u10 = surface.u10.to_numpy()
    v10 = surface.v10.to_numpy()
    tp = hybrid.tp.to_numpy().reshape((-1,) + hybrid.tp.shape[2:])
    data = np.stack((t2m, sp, tcc, u10, v10, tp), axis=-1)

    return data



def objective(trial, data, feature_list, use_neighbours=False):
    s = trial.suggest_int('s', 3, 8)  
    fh = 1
    alpha = trial.suggest_float('alpha', 0.1, 2.1, step=0.2)
    regressor_type = trial.suggest_categorical('regressor_type', ['lasso', 'ridge', 'elastic_net'])

    processor = DataProcessor(data)
    X, y = processor.preprocess(s, fh, use_neighbours=use_neighbours)
    X_train, X_test, y_train, y_test = processor.train_test_split(X, y)
    
    
    linearreg = LinearRegressor(X.shape, fh, feature_list, regressor_type=regressor_type, alpha=alpha)
    linearreg.train(X_train, y_train, normalize=True)
    y_hat = linearreg.predict_(X_test, y_test)
    
    rmse_values = linearreg.get_rmse(y_hat, y_test, normalize=True)
    mean_rmse = np.mean(rmse_values)

    return mean_rmse




def run_study(n_trials, data, feature_list, objective_function=objective, verbosity=True):
    study = optuna.create_study(direction='minimize')
    objective_func = partial(objective_function, data=data, feature_list=feature_list)
    study.optimize(objective_func, n_trials=n_trials)
    if (verbosity == False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    best_s = study.best_params['s']
    # best_fh = study.best_params['fh']
    best_regressor_type = study.best_params['regressor_type']
    best_alpha = study.best_params['alpha']
    # best_neighbour = study.best_params['use_neighbours']


    print('Best hyperparameters:')
    print(f"Best input window: {best_s}")
    print(f"Best regressor type: {best_regressor_type}")
    print(f"Best regularization constant: {best_alpha}")
    # print(f"Best use_neighbours value: {best_neighbour}")

    return best_s, best_regressor_type, best_alpha




def write_stats_to_file(file_name, n_trials, use_neighbours, best_s, best_regressor_type, best_alpha):
    with open(file_name, 'a') as file:
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:S")
        file.write(current_time)
        file.write("\n")
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


if __name__ == "__main__":
    data_path = '../data2022.grib'

    n_trials = int(sys.argv[1])
    use_neighbours = eval(sys.argv[2])

    file_name = 'hpo_linear_regression_results.txt'

    feature_list = ['t2m', 'sp', 'tcc', 'u10', 'v10', 'tp']
    data = load_data(data_path)

    best_s, best_regressor_type, best_alpha = run_study(n_trials, data, feature_list)

    write_stats_to_file(file_name, n_trials, use_neighbours, best_s, best_regressor_type, best_alpha)
