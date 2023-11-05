from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.stats import t
import matplotlib.pyplot as plt
import numpy as np
import copy


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

        if len(X_shape) > 5:
            (
                _,
                self.latitude,
                self.longitude,
                self.neighbours,
                self.input_state,
                self.features,
            ) = X_shape
        else:
            (
                _,
                self.latitude,
                self.longitude,
                self.input_state,
                self.features,
            ) = X_shape
            self.neighbours = 1

        self.fh = fh
        self.feature_list = feature_list
        self.models = [copy.deepcopy(self.model) for _ in range(self.features)]

    def train(self, X_train, y_train):
        X = X_train.reshape(-1, self.neighbours * self.input_state * self.features)
        for i in range(self.features):
            yi = y_train[..., 0, i].reshape(-1, 1)
            self.models[i].fit(X, yi)

    def get_rmse(self, y_hat, y_test):
        rmse_features = []
        for i in range(self.features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            err = round(np.sqrt(mean_squared_error(y_hat_i, y_test_i)), 3)
            rmse_features.append(err)
        return rmse_features

    def get_pred_intervals(self, X, y_hat, y_test):
        # TODO
        """
        https://stats.stackexchange.com/questions/16493/difference-between-confidence-intervals-and-prediction-intervals/16496#16496
        """
        for i in range(self.features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            n = y_hat_i.shape[0]
            df = 10
            critical_value = t.ppf((1 + 0.95) / 2, df)
        #     residual_variance =
        # y_test, y_hat, y_forecast = np.array(y_test), np.array(y_hat), np.array(y_forecast)
        # X_test, X_forecast = X_test_forecast[:-FORECASTING_HORIZON], X_test_forecast[-FORECASTING_HORIZON:]
        # residuals = y_test - y_hat
        # rmse = np.sqrt(np.mean(residuals ** 2))
        # n, p = len(X_test), X_test.shape[1]
        # if n - 1 < p:
        #     degrees_of_freedom = n - 1
        # else:
        #     degrees_of_freedom = n - p - 1
        # critical_value = t.ppf((1 + CONFIDENCE_LVL) / 2, degrees_of_freedom)
        # intervals = []
        #
        # for x0, y0 in zip(np.array(X_forecast), y_forecast):
        #     uncertainty_factor = rmse * np.sqrt(
        #         1 + np.dot(x0, np.dot(np.linalg.inv(np.dot(X_test.T, X_test)), x0))
        #     )
        #
        #     lower_bound = max(0, y0 - critical_value * uncertainty_factor)
        #     upper_bound = y0 + critical_value * uncertainty_factor
        #     intervals.append((lower_bound, upper_bound))

        # return zip(*intervals)

    def evaluate(self, y_hat, y_test):
        return self.get_rmse(y_hat, y_test)

    def plot_predictions(self, y_hat, y_test, max_samples):
        for i in range(max_samples):
            y_test_sample, y_hat_sample = y_test[i], y_hat[i]
            fig, ax = plt.subplots(
                self.features, 3 * self.fh, figsize=(10 * self.fh, 3 * self.features)
            )

            for j in range(self.features):
                cur_feature = self.feature_list[j]
                y_test_sample_feature_j = y_test_sample[..., j].reshape(-1, 1)
                y_hat_sample_feature_j = y_hat_sample[..., j].reshape(-1, 1)
                mse = mean_squared_error(
                    y_test_sample_feature_j, y_hat_sample_feature_j
                )
                rmse = np.sqrt(mse)
                print(f"RMSE {cur_feature}: {round(rmse,5)}")

                for k in range(3 * self.fh):
                    ts = k // 3
                    if k % 3 == 0:
                        title = rf"$X_{{{cur_feature},t+{ts+1}}}$"
                        value = y_test[i, :, :, ts, j]
                        cmap = plt.cm.coolwarm
                    elif k % 3 == 1:
                        title = rf"$\hat{{X}}_{{{cur_feature},t+{ts+1}}}$"
                        value = y_hat[i, :, :, ts, j]
                        cmap = plt.cm.coolwarm
                    else:
                        title = rf"$|X - \hat{{X}}|_{{{cur_feature},t+{ts+1}}}$"
                        value = np.abs(y_test[i, ..., ts, j] - y_hat[i, ..., ts, j])
                        cmap = "binary"
                    pl = ax[j, k].imshow(value, cmap=cmap)
                    ax[j, k].set_title(title)
                    ax[j, k].axis("off")
                    _ = fig.colorbar(pl, ax=ax[j, k], fraction=0.15)
            plt.show()

    def predict_and_evaluate(self, X_test, y_test, max_samples=5):
        X = X_test.reshape(-1, self.neighbours * self.input_state * self.features)
        if self.fh == 1:
            y_hat = []
            for i in range(self.features):
                y_hat_i = (
                    self.models[i]
                    .predict(X)
                    .reshape(-1, self.latitude, self.longitude, self.fh)
                )
                y_hat.append(y_hat_i)
            y_hat = np.array(y_hat).transpose((1, 2, 3, 4, 0))
        else:
            y_hat = self.predict_autoreg(X_test, y_test)

        self.plot_predictions(y_hat, y_test, max_samples=max_samples)
        eval_scores = self.evaluate(y_hat, y_test)
        print("=======================================")
        print("Evaluation metrics for entire test set:")
        print("=======================================")
        for i in range(self.features):
            print(f"RMSE {self.feature_list[i]}: {eval_scores[i]}")

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
        y_hat = np.empty(y_test.shape)
        num_samples = X_test.shape[0]

        for i in range(num_samples):
            Xi = X_test[i]
            if self.neighbours > 1:
                Yik = np.empty(Xi[..., 0, 0, :].shape)
            else:
                Yik = np.empty(Xi[..., 0, :].shape)

            for j in range(self.features):
                y_hat_j = (
                    self.models[j]
                    .predict(
                        Xi.reshape(
                            -1, self.neighbours * self.input_state * self.features
                        )
                    )
                    .reshape(1, self.latitude, self.longitude)
                )
                Yik[..., j] = y_hat_j

            y_hat[i, ..., 0, :] = Yik

            for k in range(self.fh - 1):
                Yik = np.empty(Yik.shape)
                if self.neighbours > 1:
                    # TODO: Xik above, miss self.neighbours dimension
                    Xik = np.concatenate(
                        (Xi[..., 0, k + 1 :, :], y_hat[i, ..., : k + 1, :]), axis=2
                    )
                else:
                    Xik = np.concatenate(
                        (Xi[..., k + 1 :, :], y_hat[i, ..., : k + 1, :]), axis=2
                    )
                for j in range(self.features):
                    y_hat_j = (
                        self.models[j]
                        .predict(
                            Xik.reshape(
                                -1, self.neighbours * self.input_state * self.features
                            )
                        )
                        .reshape(1, self.latitude, self.longitude)
                    )
                    Yik[..., j] = y_hat_j

                y_hat[i, ..., k + 1, :] = Yik

        return y_hat
