from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, latitude, longitude, regressor_type="linear_reg"):
        if regressor_type == "linear_reg":
            self.model = LinearRegression()
        else:
            print("Not implemented")
        self.latitude = latitude
        self.longitude = longitude

    def train(self, X_train, y_train):
        for X, y in zip(X_train, y_train):
            self.model.fit(X, y)

    def predict_and_evaluate(self, X_test, y_test, limit=10, verbose=True):
        for i in range(limit):
            y_hat = self.model.predict(X_test[i])
            mse = mean_squared_error(y_test[i], y_hat)
            r2 = r2_score(y_test[i], y_hat)
            y_hat = y_hat.reshape(self.latitude, self.longitude)
            y_test_sample = y_test[i].reshape(self.latitude, self.longitude)

            if verbose:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].imshow(y_hat, cmap=plt.cm.coolwarm)
                ax[0].set_title("Predicted")
                ax[0].axis("off")
                ax[1].imshow(y_test_sample, cmap=plt.cm.coolwarm)
                ax[1].set_title("Actual")
                ax[1].axis("off")
                plt.show()

            print(f"MSE: {mse}; R2: {r2}")
