from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


class Regressor:
    def __init__(
        self, X_shape: tuple, fh: int, feature_list: list, regressor_type="linear"
    ):
        if regressor_type == "linear":
            self.model = LinearRegression()
        elif regressor_type == "ridge":
            self.model = Ridge()
        elif regressor_type == "lasso":
            self.model = Lasso()
        elif regressor_type == "elastic_net":
            self.model = ElasticNet()
        else:
            print("Not implemented")

        _, self.input_state, self.latitude, self.longitude, self.features = X_shape
        self.fh = fh
        self.feature_list = feature_list
        self.models = [self.model for _ in range(self.features)]

    def train(self, X_train, y_train):
        flatten_features = self.longitude * self.latitude * self.features
        X = X_train.reshape(-1, self.input_state, flatten_features)
        X = X.transpose((0, 2, 1)).reshape(-1, self.input_state * self.features)
        for i in range(self.features):
            yi = y_train[:, 0, :, :, i].reshape(-1, 1)
            self.models[i].fit(X, yi)

    def evaluate(self, y_hat, y_test):
        rmse_features = []
        for i in range(self.features):
            y_hat_i = y_hat[:, :, :, :, i].flatten()
            y_test_i = y_test[:, :, :, :, i].flatten()
            err = round(np.sqrt(mean_squared_error(y_hat_i, y_test_i)), 3)
            rmse_features.append(err)
        return rmse_features

    def predict_and_evaluate(self, X_test, y_test, max_samples=5):
        flatten_features = self.longitude * self.latitude * self.features
        X = X_test.reshape(-1, self.input_state, flatten_features).transpose((0, 2, 1))
        if self.fh == 1:
            y_hat = []
            for i in range(self.features):
                y_hat_i = (
                    self.models[i]
                    .predict(X.reshape(-1, self.input_state * self.features))
                    .reshape(-1, self.fh, self.latitude, self.longitude)
                )
                y_hat.append(y_hat_i)
            y_hat = np.array(y_hat).transpose((1, 2, 3, 4, 0))
        else:
            y_hat = self.predict_autoreg(X_test, y_test)

        rmse_features = self.evaluate(y_hat, y_test)

        for i in range(max_samples):
            y_test_sample, y_hat_sample = y_test[i], y_hat[i]
            fig, ax = plt.subplots(
                self.features, 2 * self.fh, figsize=(8 * self.fh, 3 * self.features)
            )

            for j in range(self.features):
                cur_feature = self.feature_list[j]
                y_test_sample_feature_j = y_test_sample[:, :, :, j].flatten()
                y_hat_sample_feature_j = y_hat_sample[:, :, :, j].flatten()
                mse = mean_squared_error(
                    y_test_sample_feature_j, y_hat_sample_feature_j
                )
                rmse = np.sqrt(mse)
                print(f"RMSE {cur_feature}: {round(rmse,3)}")

                for k in range(2 * self.fh):
                    ts = k // 2
                    if k % 2 == 0:
                        title = rf"$X_{{{cur_feature},t+{ts+1}}}$"
                        value = y_test[i, ts, :, :, j]
                    else:
                        title = rf"$\hat{{X}}_{{{cur_feature},t+{ts+1}}}$"
                        value = y_hat[i, ts, :, :, j]
                    pl = ax[j, k].imshow(value, cmap=plt.cm.coolwarm)
                    ax[j, k].set_title(title)
                    ax[j, k].axis("off")
                    _ = fig.colorbar(pl, ax=ax[j, k], fraction=0.15)
            plt.show()

        print("=======================================")
        print("Evaluation metrics for entire test set:")
        print("=======================================")
        for i in range(self.features):
            print(f"RMSE {self.feature_list[i]}: {rmse_features[i]}")

        return y_hat

    def predict_autoreg(self, X_test, y_test):
        """
        Prediction in an autoregressive manner:
        Depends on forecasting horizon parameter, each model prediction
        becomes a part of an input for next prediction.

        self.input_size -> n
        self.fh -> k

        (Xi,Xi+1, ..., Xi+n) -> Yi+n+1
        (Xi+1, Xi+2, ..., Xi+n, Yi+n+1) -> Yi+n+2
        ...
        (Xi+k-1, ..., Yi+n+k-2 ,Yi+n+k-1) -> Yi+n+k

        """
        y_hat = np.zeros(y_test.shape)
        num_samples = X_test.shape[0]
        for i in range(num_samples):
            for j in range(self.features):
                y_hat_ij = np.zeros(y_test[i].shape[:-1])
                Xij = X_test[i, :, :, :, j].reshape(
                    1, self.input_state, self.latitude * self.longitude
                )
                Xij = Xij.transpose((0, 2, 1)).reshape(-1, self.input_state)
                y_hat_ij[0] = (
                    self.models[j]
                    .predict(Xij)
                    .reshape(1, self.latitude, self.longitude)
                )
                for k in range(self.fh - 1):
                    Xij = np.concatenate(
                        (X_test[i, k + 1 :, :, :, j], y_hat_ij[: k + 1]), axis=0
                    )
                    Xij = Xij.reshape(
                        (1, self.input_state, self.latitude * self.longitude)
                    )
                    Xij = Xij.transpose((0, 2, 1)).reshape(-1, self.input_state)
                    y_hat_ij[k + 1] = (
                        self.models[j]
                        .predict(Xij)
                        .reshape(1, self.latitude, self.longitude)
                    )
                y_hat[i, :, :, :, j] = y_hat_ij
        return y_hat
