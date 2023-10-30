from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

FEATURES_LIST = ["Temperature", "Pressure"]


class Regressor:
    def __init__(self, data_shape: tuple, regressor_type="linear_reg"):
        if regressor_type == "linear_reg":
            self.model = LinearRegression()
        else:
            print("Not implemented")

        _, self.input_state, self.latitude, self.longitude, self.features = data_shape

    def train(self, X_train, y_train):
        X = X_train.reshape(-1, self.input_state)
        y = y_train.reshape(-1, 1)
        self.model.fit(X, y)
        # return X, y
        # X = X_train.reshape(-1, self.input_state, self.features)
        # y = y_train.reshape(-1, self.features)
        #
        # for i in range(self.features):
        #     self.model.fit(X[:, :, i], y[:, i])

    def predict_and_evaluate(self, X_test, y_test, limit=5, verbose=True):

        # X = X_test.reshape(-1, self.input_state, self.features)
        # for i in range(self.features):
        # y_hat = self.model.predict(X[:, :, i])
        # return y_hat

        # y_hat = self.model.predict(X_test.reshape(-1, self.input_state))
        # return y_hat

        y_hat = self.model.predict(X_test.reshape(-1, self.input_state))
        y_hat = y_hat.reshape((-1, self.latitude, self.longitude, self.features))
        y_test = y_test.reshape(y_hat.shape)

        # return y_hat, y_test

        for i in range(limit):
            # y_test_sample = y_test[i].reshape(self.longitude * self.latitude, self.features)
            # y_hat_sample = y_hat[i].reshape(self.longitude * self.latitude, self.features)
            for j in range(self.features):
                # y_t = y_test_sample[:, j]
                # y_h = y_hat_sample[:, j]
                # mse = mean_squared_error(y_t, y_h)
                # r2 = r2_score(y_t, y_h)

                if verbose:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    plt.suptitle(FEATURES_LIST[j])
                    ax[0].imshow(y_hat[i, :, :, j], cmap=plt.cm.coolwarm)
                    ax[0].set_title("Predicted")
                    ax[0].axis("off")
                    ax[1].imshow(y_test[i, :, :, j], cmap=plt.cm.coolwarm)
                    ax[1].set_title("Actual")
                    ax[1].axis("off")
                    plt.show()

                # print(f"MSE: {mse}; R2: {r2}")
