import cfgrib
import numpy as np
import sys

sys.path.append("../..")
from models.baseline_regressor import BaselineRegressor


def step_split(feature, n_steps=3):
    step_split = np.split(feature, n_steps, axis=1)
    step_split = [np.squeeze(arr, axis=1) for arr in step_split]

    return np.array(step_split)


def load_tigge_0_to_12_by_6(grib_file):
    grib_data = cfgrib.open_datasets(grib_file)

    # grib_data.shape()

    tcc_tigge = grib_data[0].tcc.to_numpy()
    tcc_step_0, tcc_step_6, tcc_step_12 = step_split(tcc_tigge) / 100

    u10_tigge = grib_data[1].u10.to_numpy()
    u10_step_0, u10_step_6, u10_step_12 = step_split(u10_tigge)

    v10_tigge = grib_data[1].v10.to_numpy()
    v10_step_0, v10_step_6, v10_step_12 = step_split(v10_tigge)

    t2m_tigge = grib_data[2].t2m.to_numpy()
    t2m_step_0, t2m_step_6, t2m_step_12 = step_split(t2m_tigge) - 273.15

    sp_tigge = grib_data[3].sp.to_numpy()
    sp_step_0, sp_step_6, sp_step_12 = step_split(sp_tigge) / 100

    tp_tigge = grib_data[3].tp.to_numpy()
    tp_step_0, tp_step_6, tp_step_12 = step_split(tp_tigge)

    data_step_0 = np.stack(
        (t2m_step_0, sp_step_0, tcc_step_0, u10_step_0, v10_step_0, tp_step_0), axis=-1
    )
    data_step_6 = np.stack(
        (t2m_step_6, sp_step_6, tcc_step_6, u10_step_6, v10_step_6, tp_step_6), axis=-1
    )
    data_step_12 = np.stack(
        (
            t2m_step_12,
            sp_step_12,
            tcc_step_12,
            u10_step_12,
            v10_step_12,
            tp_step_12,
            sp_step_12,
        ),
        axis=-1,
    )

    return data_step_0, data_step_6, data_step_12


def evaluate_and_compare(data1, data2, max_samples=1):
    feature_list = ["t2m", "sp", "tcc", "u10", "v10", "tp"]

    X, Y = data1[..., np.newaxis, :], data2[..., np.newaxis, :]
    data1_regressor = BaselineRegressor(X.shape, 1, feature_list)
    data1_regressor.plot_predictions(X, Y, max_samples=max_samples)
    rmse_scores, mae_scores = data1_regressor.evaluate(X, Y)

    for i in range(len(feature_list)):
        print(f"{feature_list[i]} => RMSE: {rmse_scores[i]};  MAE: {mae_scores[i]};")
