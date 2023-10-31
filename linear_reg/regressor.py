from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

FEATURES_LIST = ["Temperature", "Pressure"]


class Regressor:
    def __init__(
        self, data_shape: tuple, feature_list: list, regressor_type="linear_reg"
    ):
        if regressor_type == "linear_reg":
            self.model = LinearRegression()
        else:
            print("Not implemented")

        _, self.input_state, self.latitude, self.longitude, self.features = data_shape
        self.feature_list = feature_list

    def train(self, X_train, y_train):
        X = X_train.reshape(
            -1, self.input_state, self.longitude * self.latitude * self.features
        )
        X = X.transpose((0, 2, 1))
        X = X.reshape(-1, self.input_state)
        y = y_train.reshape(-1, 1)
        self.model.fit(X, y)

    def predict_and_evaluate(self, X_test, y_test, limit=5, verbose=True):
        X = X_test.reshape(
            -1, self.input_state, self.longitude * self.latitude * self.features
        )
        X = X.transpose((0, 2, 1))

        y_hat = self.model.predict(X.reshape(-1, self.input_state))
        y_hat = y_hat.reshape((-1, self.latitude, self.longitude, self.features))
        y_test = y_test.reshape(y_hat.shape)

        for i in range(limit):
            y_test_sample = y_test[i].reshape(-1, self.features)
            y_hat_sample = y_hat[i].reshape(-1, self.features)
            fig, ax = plt.subplots(self.features, 2, figsize=(8, 5))

            for j in range(self.features):
                y_test_sample_feature_j = y_test_sample[:, j]
                y_hat_sample_feature_j = y_hat_sample[:, j]
                mse = mean_squared_error(
                    y_test_sample_feature_j, y_hat_sample_feature_j
                )
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_sample_feature_j, y_hat_sample_feature_j)

                if verbose:
                    ax[j, 0].imshow(y_hat[i, :, :, j], cmap=plt.cm.coolwarm)
                    ax[j, 0].set_title(f"Predicted [{self.feature_list[j]}]")
                    ax[j, 0].axis("off")
                    ax[j, 1].imshow(y_test[i, :, :, j], cmap=plt.cm.coolwarm)
                    ax[j, 1].set_title(f"Actual [{self.feature_list[j]}]")
                    ax[j, 1].axis("off")

                rmse, r2 = round(rmse, 3), round(r2, 3)
                print(
                    f"RMSE {self.feature_list[j]}: {rmse}; R2 {self.feature_list[j]}: {r2}"
                )

            plt.show()

        return y_hat