from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

class SmoothingPredictor:
    def __init__(
        self, X_shape: tuple, fh: int, feature_list: list, smoothing_type="simple"
    ):
        if smoothing_type in ("simple", "holt"):
            self.type = smoothing_type
        else:
            print("Not implemented")
            raise ValueError
        
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
        self.models = [_ for _ in range(self.features)]
        self.params = [_ for _ in range(self.features)]

    def train(self, X_train, y_train):
        X = X_train.reshape(-1, self.features)
        # print(X.shape)
        for i in range(self.features):
            if self.type == "simple":
                self.models[i] = SimpleExpSmoothing(X[:, i], initialization_method='estimated').fit()
                self.params[i] = self.models[i].params_formatted["param"]
            elif self.type == "holt":
                self.models[i] = Holt(X[:, i], initialization_method='estimated').fit()
                self.params[i] = self.models[i].params_formatted["param"]
        print(self.params[0])
        print()

    def get_rmse(self, y_hat, y_test):
        rmse_features = []
        for i in range(self.features):
            y_hat_i = y_hat[..., i].reshape(-1, 1)
            y_test_i = y_test[..., i].reshape(-1, 1)
            err = round(np.sqrt(mean_squared_error(y_hat_i, y_test_i)), 3)
            rmse_features.append(err)
        return rmse_features
    
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

    def predict_and_evaluate(self, X_test, y_test, max_samples=5):
        X = X_test.reshape(-1, self.latitude, self.longitude, self.input_state, self.features)
        y_hat = []
        for i in range(X.shape[0]):
            y_hat_i = []
            for j in range(self.features):
                ylat = []
                for lat in range(X.shape[1]):
                    ylon = []
                    for lon in range(X.shape[2]):
                        if self.type == "simple":
                            forecast = SimpleExpSmoothing(X[i, lat, lon, :, j], initialization_method="known", initial_level=self.params[j]["initial_level"]).fit(smoothing_level=self.params[j]["smoothing_level"]).forecast(self.fh)
                        elif self.type == "holt":
                            forecast = Holt(X[i, lat, lon, :, j], initialization_method = "known", initial_level=self.params[j]["initial_level"], initial_trend=self.params[j]["initial_trend"]).fit(smoothing_level=self.params[j]["smoothing_level"], smoothing_trend=self.params[j]["smoothing_trend"]).forecast(self.longitude*self.latitude*self.fh)
                        else: 
                            raise ValueError
                        ylon.append(forecast)
                    ylat.append(ylon)
                y_hat_i.append(ylat)
            y_hat.append(y_hat_i)
            if i%50 == 0:
                print(i, "/", X.shape[0])
        y_hat = np.array(y_hat).reshape(X.shape[0], self.features, self.latitude, self.longitude, self.fh).transpose((0, 2, 3, 4, 1))

        self.plot_predictions(y_hat, y_test, max_samples=max_samples)
        eval_scores = self.evaluate(y_hat, y_test)
        print("=======================================")
        print("Evaluation metrics for entire test set:")
        print("=======================================")
        for i in range(self.features):
            print(f"RMSE {self.feature_list[i]}: {eval_scores[i]}")

        return y_hat