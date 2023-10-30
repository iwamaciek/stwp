from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, resolution, sequence_length, regressor_type="linear_reg"):
        if regressor_type == "linear_reg":
            self.model = LinearRegression()
        else:
            print("Not implemented")
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.latitude, self.longitude = resolution

    def train(self, X_train, y_train):
        X = X_train.reshape(-1, self.sequence_length - 1)
        y = y_train.reshape(-1, 1)
        self.model.fit(X, y)

    def predict_and_evaluate(self, X_test, y_test, limit=10, verbose=True):
        y_hat = self.model.predict(X_test.reshape(-1, self.sequence_length - 1))
        y_hat = y_hat.reshape(-1, self.longitude * self.latitude)

        for i in range(limit):
            y_test_sample = y_test[i].reshape(self.resolution)
            y_hat_sample = y_hat[i].reshape(self.resolution)

            mse = mean_squared_error(y_test_sample.reshape(-1, 1), y_hat[i])
            r2 = r2_score(y_test_sample.reshape(-1, 1), y_hat[i])

            if verbose:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].imshow(y_hat_sample, cmap=plt.cm.coolwarm)
                ax[0].set_title("Predicted")
                ax[0].axis("off")
                ax[1].imshow(y_test_sample, cmap=plt.cm.coolwarm)
                ax[1].set_title("Actual")
                ax[1].axis("off")
                plt.show()

            print(f"MSE: {mse}; R2: {r2}")
