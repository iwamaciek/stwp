from baselines.config import (
    FH,
    INPUT_SIZE,
)
# from baselines.gnn.processor import NNDataProcessor

# sys.path.append("..")
from baselines.gnn.processor import NNDataProcessor

class CNNDataProcessor(NNDataProcessor):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, subset=None):
        X_train, X_test, y_train, y_test = self.train_test_split()
        X, y = self.get_scalers(X_train, X_test, y_train, y_test)
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))
        X = X.reshape(-1, self.num_latitudes, self.num_longitudes, INPUT_SIZE, self.num_features)
        y = y.reshape(-1, self.num_latitudes, self.num_longitudes, FH, self.num_features)
        self.train_loader, self.test_loader = self.get_loaders(X, y, subset)