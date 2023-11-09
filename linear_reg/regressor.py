from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# from skleran.
from scipy.stats import t
from data_processor import DataProcessor
import matplotlib.pyplot as plt
import numpy as np
import copy


class Regressor:
    def __init__(
        self, X_shape: tuple, fh: int, feature_list: list, regressor_type="linear", alpha = 1
    ):
        if regressor_type == "linear":
            self.model = LinearRegression()
        elif regressor_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif regressor_type == "lasso":
            self.model = Lasso(alpha=alpha)
        elif regressor_type == "elastic_net":
            self.model = ElasticNet(alpha=alpha)
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
        # self.feature_scaler = MinMaxScaler()
        self.scaler_list = []

    def train(self, X_train, y_train):
        X = X_train.reshape(-1, self.neighbours * self.input_state * self.features)
        for i in range(self.features):
            yi = y_train[..., 0, i].reshape(-1, 1)
            self.models[i].fit(X, yi)
            
    
    def train_and_scale(self, X_train, y_train):
        
        X = X_train.reshape(-1, self.neighbours * self.input_state * self.features)

        tmp = y_train[..., 0, 0].reshape(-1, 1)
        # self.feature_scaler.fit(tmp)

        for i in range(self.features):
            scaler = MinMaxScaler()
            
            yi = y_train[..., 0, i].reshape(-1, 1)
            scaler.fit(yi)
            self.scaler_list.append(scaler)
            # yi = self.feature_scaler.transform(yi)
            self.models[i].fit(X, yi)

    def get_rmse(self, y_hat, y_test, normalize=False):
        rmse_features = []
        for i in range(self.features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            if normalize:
                y_test_i = self.scaler_list[i].transform(y_test_i)
                y_hat_i = self.scaler_list[i].transform(y_hat_i)
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
        #     residual_variance = (y_test_u - y_hat_i)**2
        # residuals = y_test - y_hat
        # rmse = np.sqrt(np.mean(residuals ** 2))
        # n, p = len(X), X.shape[1] (idk those supposted to be no. of predictors)
        # if n - 1 < p:
        #     degrees_of_freedom = n - 1
        # else:
        #     degrees_of_freedom = n - p - 1
        # critical_value = t.ppf((1 + CONFIDENCE_LVL) / 2, degrees_of_freedom)
        #
        # for x0, y0 in zip(np.array(X), y_hat):
        #     uncertainty_factor = rmse * np.sqrt(
        #         1 + np.dot(x0, np.dot(np.linalg.inv(np.dot(X_test.T, X_test)), x0))
        #     )
        #     y0 +- critical_value * uncertainty_factor

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
                std = np.std(y_test_sample_feature_j)
                sqrt_n = np.sqrt(y_test_sample_feature_j.shape[0])
                print(f"{cur_feature} => RMSE:  {round(rmse,5)}; SE: {std / sqrt_n}")

                for k in range(3 * self.fh):
                    ts = k // 3
                    if k % 3 == 0:
                        title = rf"$X_{{{cur_feature},t+{ts+1}}}$"
                        value = y_test[i, ..., ts, j]
                        cmap = plt.cm.coolwarm
                    elif k % 3 == 1:
                        title = rf"$\hat{{X}}_{{{cur_feature},t+{ts+1}}}$"
                        value = y_hat[i, ..., ts, j]
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

    def predict_(self, X_test, y_test):
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
        return y_hat

    def predict_and_evaluate(self, X_test, y_test, max_samples=5):
        y_hat = self.predict_(X_test, y_test)
        self.plot_predictions(y_hat, y_test, max_samples=max_samples)
        eval_scores = self.evaluate(y_hat, y_test)
        # TODO eval_score for t2m look kinda suspicious
        print("=======================================")
        print("Evaluation metrics for entire test set:")
        print("=======================================")

        sqrt_n = np.sqrt(y_test.shape[0] * self.latitude * self.longitude * self.fh)
        for i in range(self.features):
            print(
                f"{self.feature_list[i]} => RMSE: {eval_scores[i]}; SE: {np.std(y_test[...,i]) / sqrt_n}"
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
        y_hat = np.empty(y_test.shape)
        num_samples = X_test.shape[0]
        for i in range(num_samples):
            Xi = X_test[i]
            Yik = np.empty(
                (self.latitude, self.longitude, self.neighbours, self.fh, self.features)
            )
            for k in range(-1, self.fh - 1):
                Xik = Xi
                if k > -1:
                    if self.neighbours > 1:
                        Yik[..., k, :] = self.extend(y_hat[i, ..., k : k + 1, :])
                        Xik = np.concatenate(
                            (Xi[..., k + 1 :, :], Yik[..., : k + 1, :]), axis=3
                        )
                    else:
                        Xik = np.concatenate(
                            (Xi[..., k + 1 :, :], y_hat[i, ..., : k + 1, :]), axis=2
                        )
                for j in range(self.features):
                    y_hat[i, ..., k + 1, j] = (
                        self.models[j]
                        .predict(
                            Xik.reshape(
                                -1, self.neighbours * self.input_state * self.features
                            )
                        )
                        .reshape(1, self.latitude, self.longitude)
                    )
        return y_hat

    def extend(self, Y):
        """
        Extend data sample such that it will use neighbours
        shape: (latitude, longitude, neighbours, features)
        It might be in data_processor
        """
        # TODO function that maps no. of neighbours -> radius
        if self.neighbours <= 5:
            radius = 1
        elif self.neighbours <= 13:
            radius = 2
        # ...
        else:
            radius = 3

        _, indices = DataProcessor.count_neighbours(radius=radius)
        Y_out = np.empty(
            (self.latitude, self.longitude, self.neighbours, self.features)
        )
        for n in range(self.neighbours):
            i, j = indices[n - 1]
            for lo in range(self.longitude):
                for la in range(self.latitude):
                    if 0 < la + i < self.latitude and 0 < lo + j < self.longitude:
                        Y_out[la, lo, n] = Y[la + i, lo + j]
                    else:
                        Y_out[la, lo, n] = Y[la, lo]
        return Y_out
