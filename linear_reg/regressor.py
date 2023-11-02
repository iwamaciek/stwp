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
        # self.models = [self.model for _ in range(self.features)]

    def train(self, X_train, y_train):
        flatten_features = self.longitude * self.latitude * self.features
        X = X_train.reshape(-1, self.input_state, flatten_features)
        X = X.transpose((0, 2, 1)).reshape(-1, self.input_state)
        y = y_train[:, 0, :, :, :].reshape(-1, 1)
        self.model.fit(X, y)

    def evaluate(self, y_hat, y_test):
        y_hat = y_hat.reshape(-1, self.features)
        y_test = y_test.reshape(-1, self.features)
        rmse_features, r2_features = [], []
        for i in range(self.features):
            rmse_features.append(np.sqrt(mean_squared_error(y_hat[:, i], y_test[:, i])))
            r2_features.append(r2_score(y_hat[:, i], y_test[:, i]))
        return rmse_features, r2_features

    def predict_and_evaluate(self, X_test, y_test, limit=5, verbose=True):
        flatten_features = self.longitude * self.latitude * self.features
        X = X_test.reshape(-1, self.input_state, flatten_features).transpose((0, 2, 1))

        if self.fh == 1:
            y_hat = self.model.predict(X.reshape(-1, self.input_state))
        else:
            y_hat = self.predict_autoreg(X_test, y_test)

        rmse_features, r2_features = self.evaluate(y_hat, y_test)
        y_hat = y_hat.reshape(
            (-1, self.fh, self.latitude, self.longitude, self.features)
        )
        y_test = y_test.reshape(y_hat.shape)

        for i in range(limit):
            y_test_sample = y_test[i].reshape(-1, self.features)
            y_hat_sample = y_hat[i].reshape(-1, self.features)
            fig, ax = plt.subplots(
                self.features, 2 * self.fh, figsize=(8 * self.fh, 3 * self.features)
            )

            for j in range(self.features):
                cur_feature = self.feature_list[j]
                y_test_sample_feature_j = y_test_sample[:, j]
                y_hat_sample_feature_j = y_hat_sample[:, j]
                mse = mean_squared_error(
                    y_test_sample_feature_j, y_hat_sample_feature_j
                )
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_sample_feature_j, y_hat_sample_feature_j)
                rmse, r2 = round(rmse, 3), round(r2, 3)
                print(f"RMSE {cur_feature}: {rmse}; R2 {cur_feature}: {r2}")

                if verbose:
                    for k in range(2 * self.fh):
                        ts = k // 2
                        if k % 2 == 0:
                            title = rf"$X_{{{cur_feature},t+{ts+1}}}$"
                            value = y_test[i, ts, :, :, j]
                        else:
                            title = rf"$\hat{{X}}_{{{cur_feature},t+{ts+1}}}$"
                            value = y_hat[i, ts, :, :, j]
                        _ = ax[j, k].imshow(value, cmap=plt.cm.coolwarm)
                        ax[j, k].set_title(title)
                        ax[j, k].axis("off")

                        # _ = fig.colorbar(predicted, ax=ax[j, 1], fraction=0.15)
            plt.show()

        if verbose:
            print("=======================================")
            print("Evaluation metrics for entire test set:")
            print("=======================================")
            for i in range(self.features):
                print(
                    f"RMSE {self.feature_list[i]}: {rmse_features[i]}; R2 {self.feature_list[i]}: {r2_features[i]}"
                )

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
        flatten_features = self.longitude * self.latitude * self.features
        y_hat = np.zeros(y_test.shape)
        for i in range(X_test.shape[0]):
            y_hat_i = np.zeros(y_test[i].shape)
            X = X_test[i].reshape((1,) + X_test[i].shape)
            X = X.reshape((-1, self.input_state, flatten_features)).transpose((0, 2, 1))
            X = X.reshape((-1, self.input_state))
            y_hat_i[0] = self.model.predict(X).reshape(y_test[i][0].shape)
            for j in range(self.fh - 1):
                X = np.concatenate(
                    (X_test[i, j + 1 :, :, :, :], y_hat_i[: j + 1]), axis=0
                )
                X = X.reshape((-1, self.input_state, flatten_features)).transpose(
                    (0, 2, 1)
                )
                X = X.reshape(-1, self.input_state)
                y_hat_i[j + 1] = self.model.predict(X).reshape(X_test[i, j].shape)
            y_hat[i] = y_hat_i
        return y_hat
