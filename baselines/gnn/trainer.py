import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from torch.optim.lr_scheduler import StepLR
from baselines.gnn.processor import NNDataProcessor
from baselines.gnn.config import DEVICE, FH, BATCH_SIZE, INPUT_SIZE
from baselines.gnn.callbacks import (
    LRAdjustCallback,
    CkptCallback,
    EarlyStoppingCallback,
)
from baselines.gnn.temporal_gnn import TemporalGNN
from baselines.gnn.crystal_gcn import CrystalGNN
from baselines.gnn.basic_gcn import BasicGCN


class Trainer:
    def __init__(
        self, architecture="a3tgcn", hidden_dim=2048, lr=0.001, gamma=0.5, subset=None
    ):

        # Full data preprocessing for nn input run in NNDataProcessor constructor
        # If subset param is given train_data and test_data will have len=subset
        self.nn_proc = NNDataProcessor()
        self.nn_proc.preprocess(subset=subset)
        self.train_loader = self.nn_proc.train_loader
        self.test_loader = self.nn_proc.test_loader
        (
            _,
            self.latitude,
            self.longitude,
            self.features,
        ) = self.nn_proc.get_shapes()
        self.edge_index = self.nn_proc.edge_index
        self.edge_weights = self.nn_proc.edge_weights
        self.scalers = self.nn_proc.scalers
        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)
        if subset is None:
            self.subset = self.train_size
        else:
            self.subset = subset

        # Architecture details
        if architecture == "a3tgcn":
            self.model = TemporalGNN(self.features, hidden_dim, FH).to(DEVICE)
        elif architecture == "cgcn":
            self.model = CrystalGNN(self.features * INPUT_SIZE, 1, hidden_dim).to(
                DEVICE
            )
        elif architecture == "gcn":
            self.model = BasicGCN(self.features * INPUT_SIZE, hidden_dim).to(DEVICE)
        else:
            # TODO handling
            self.model = None

        self.criterion = lambda output, target: (output - target).pow(2).sum()
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)

        # Callbacks
        self.lr_callback = LRAdjustCallback(self.optimizer, self.scheduler)
        self.ckpt_callback = CkptCallback(self.model)
        self.early_stop_callback = EarlyStoppingCallback()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, num_epochs=50):
        gradient_clip = 1.0
        start = time.time()

        val_loss_list = []
        train_loss_list = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                y_hat = self.model(batch.x, batch.edge_index, batch.edge_attr)

                loss = self.criterion(y_hat, batch.y)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / (self.subset * BATCH_SIZE)
            train_loss_list.append(avg_loss)
            last_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, lr: {last_lr}"
            )

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.test_loader:
                    y_hat = self.model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = self.criterion(y_hat, batch.y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (min(self.subset, self.test_size) * BATCH_SIZE)
            val_loss_list.append(avg_val_loss)

            print(f"Val Loss: {avg_val_loss:.4f}\n---------")

            self.lr_callback.step(avg_loss)
            self.ckpt_callback.step(avg_val_loss)
            self.early_stop_callback.step(avg_loss)
            if self.early_stop_callback.early_stop:
                break

        end = time.time()
        print(f"{end - start} [s]")
        self.plot_loss(val_loss_list, train_loss_list)

    @staticmethod
    def plot_loss(val_loss_list, train_loss_list):
        x = np.arange(1, len(train_loss_list) + 1)
        plt.figure(figsize=(20, 7))
        plt.plot(x, train_loss_list, label="train loss")
        plt.plot(x, val_loss_list, label="val loss")
        plt.title("Loss plot")
        plt.legend()
        plt.show()

    def inverse_normalization_predict(self, X, y):
        y = y.reshape((-1, self.latitude, self.longitude, self.features, FH))
        y = y.cpu().detach().numpy()

        y_hat = self.model(X, self.edge_index, self.edge_weights)
        y_hat = y_hat.reshape((-1, self.latitude, self.longitude, self.features, FH))
        y_hat = y_hat.cpu().detach().numpy()

        yshape = (self.latitude, self.longitude, FH)

        for i in range(self.features):
            for j in range(y_hat.shape[0]):
                yi = y[j, ..., i, :].copy().reshape(-1, 1)
                yhat_i = y_hat[j, ..., i, :].copy().reshape(-1, 1)

                y[j, ..., i, :] = self.scalers[i].inverse_transform(yi).reshape(yshape)
                y_hat[j, ..., i, :] = (
                    self.scalers[i].inverse_transform(yhat_i).reshape(yshape)
                )

        return y, y_hat

    def plot_predictions(self, data_type="test"):
        # TODO
        # for now it is just hard-coded as first train or test sample
        if data_type == "train":
            sample = next(iter(self.train_loader))
        elif data_type == "test":
            sample = next(iter(self.test_loader))
        else:
            print("Invalid type: (train, test)")
            raise ValueError

        X, y = sample.x, sample.y
        y, y_hat = self.inverse_normalization_predict(X, y)

        for i in range(BATCH_SIZE):
            fig, ax = plt.subplots(
                self.features, 3 * FH, figsize=(10 * FH, 3 * self.features)
            )
            for j in range(self.features):
                cur_feature = f"f{j}"
                for k in range(3 * FH):
                    ts = k // 3
                    if k % 3 == 0:
                        title = rf"$X_{{{cur_feature},t+{ts + 1}}}$"
                        value = y[i, ..., j, ts]
                        cmap = plt.cm.coolwarm
                    elif k % 3 == 1:
                        title = rf"$\hat{{X}}_{{{cur_feature},t+{ts + 1}}}$"
                        value = y_hat[i, ..., j, ts]
                        cmap = plt.cm.coolwarm
                    else:
                        title = rf"$|X - \hat{{X}}|_{{{cur_feature},t+{ts + 1}}}$"
                        value = np.abs(y[i, ..., j, ts] - y_hat[i, ..., j, ts])
                        cmap = "binary"

                    pl = ax[j, k].imshow(
                        value.reshape(self.latitude, self.longitude), cmap=cmap
                    )
                    ax[j, k].set_title(title)
                    ax[j, k].axis("off")
                    _ = fig.colorbar(pl, ax=ax[j, k], fraction=0.15)

        for j in range(self.features):
            cur_feature = f"f{j}"
            loss = (
                np.mean(
                    (y_hat[..., j, :].reshape(-1, 1) - y[..., j, :].reshape(-1, 1)) ** 2
                )
            ) ** 0.5
            print(f"RMSE for {cur_feature}: {loss}")

    def evaluate(self, data_type="test"):
        if data_type == "train":
            loader = self.train_loader
        elif data_type == "test":
            loader = self.test_loader
        else:
            print("Invalid type: (train, test)")
            raise ValueError

        y = np.empty((0, self.latitude, self.longitude, self.features, FH))
        y_hat = np.empty((0, self.latitude, self.longitude, self.features, FH))
        for batch in loader:
            y_i, y_hat_i = self.inverse_normalization_predict(batch.x, batch.y)
            y = np.concatenate((y, y_i), axis=0)
            y_hat = np.concatenate((y_hat, y_hat_i), axis=0)

        for i in range(self.features):
            y_fi = y[..., i, :].reshape(-1, 1)
            y_hat_fi = y_hat[..., i, :].reshape(-1, 1)
            rmse_fi = np.mean((y_fi - y_hat_fi) ** 2) ** (1 / 2)
            print(f"RMSE for f{i}: {rmse_fi}")

    def get_model(self):
        return self.model
