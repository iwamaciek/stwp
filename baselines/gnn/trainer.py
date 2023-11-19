import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from torch.optim.lr_scheduler import StepLR
from baselines.gnn.processor import preprocess
from baselines.gnn.config import DEVICE, FH, TRAIN_RATIO
from baselines.gnn.callbacks import LRAdjustCallback, CkptCallback
from baselines.gnn.temporal_gnn import TemporalGNN


class Trainer:
    def __init__(self, hidden_dim=2048, lr=0.001, gamma=0.5):
        self.dataset, self.scalers, self.shapes = preprocess()
        self.samples, self.latitude, self.longitude, self.features = self.shapes
        self.model = TemporalGNN(self.features, hidden_dim, FH).to(DEVICE)
        self.edge_index = self.dataset[0].edge_index
        self.edge_weights = self.dataset[0].edge_attr
        self.train_size = int(self.samples * TRAIN_RATIO)
        self.test_size = self.samples - self.train_size
        self.train_data, self.test_data = (
            self.dataset[: self.train_size],
            self.dataset[self.train_size :],
        )
        self.criterion = lambda output, target: (output - target).pow(2).sum()
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)
        self.lr_callback = LRAdjustCallback(self.optimizer, self.scheduler)
        self.ckp_callback = CkptCallback(self.model)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, num_epochs=50, subset=None):
        gradient_clip = 1.0
        start = time.time()

        if subset is None:
            subset = self.train_size

        val_loss_list = []
        train_loss_list = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_data[:subset]:
                y_hat = self.model(batch.x, batch.edge_index, batch.edge_attr)

                loss = self.criterion(y_hat, batch.y)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_data[:subset])
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}, Train Loss: {avg_loss:.4f}"
            )
            train_loss_list.append(avg_loss)
            self.lr_callback.step(avg_loss)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.test_data[:subset]:
                    y_hat = self.model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = self.criterion(y_hat, batch.y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.test_data[:subset])
            print(f"Val Loss: {avg_val_loss:.4f}\n---------")
            val_loss_list.append(avg_val_loss)

            self.ckp_callback.step(avg_val_loss)

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

    def plot_predictions(self, type="test"):
        if type == "train":
            sample = self.train_data[0]
        elif type == "test":
            sample = self.test_data[0]
        else:
            print("Invalid type: (train, test)")
            raise ValueError

        X = sample.x
        y = sample.y
        y = y.reshape((self.latitude, self.longitude, self.features, FH))
        y = y.cpu().detach().numpy()

        y_hat = self.model(X, self.edge_index, self.edge_weights)
        y_hat = y_hat.reshape((self.latitude, self.longitude, self.features, FH))
        y_hat = y_hat.cpu().detach().numpy()

        yshape = (self.latitude, self.longitude, FH)

        for i in range(self.features):
            yi = y[..., i, :].copy().reshape(-1, 1)
            yhat_i = y_hat[..., i, :].copy().reshape(-1, 1)

            y[..., i, :] = self.scalers[i].inverse_transform(yi).reshape(yshape)
            y_hat[..., i, :] = self.scalers[i].inverse_transform(yhat_i).reshape(yshape)

        for i in range(self.features):
            loss = np.mean(
                np.abs(y_hat[..., i, :].reshape(-1, 1) - y[..., i, :].reshape(-1, 1))
            )
            print(f"MAE for {i + 1} feature: {loss}")

        fig, ax = plt.subplots(
            self.features, 2 * FH, figsize=(10 * FH, 3 * self.features)
        )

        for j in range(self.features):
            cur_feature = f"f{j}"
            ts = 0
            for k in range(2 * FH):
                if k % 2 == 0:
                    title = rf"$X_{{{cur_feature},t+{ts + 1}}}$"
                    value = y[..., j, :]
                    cmap = plt.cm.coolwarm
                else:
                    title = rf"$\hat{{X}}_{{{cur_feature},t+{ts + 1}}}$"
                    value = y_hat[..., j, :]
                    cmap = plt.cm.coolwarm

                pl = ax[j, k].imshow(
                    value.reshape(self.latitude, self.longitude), cmap=cmap
                )
                ax[j, k].set_title(title)
                ax[j, k].axis("off")
                _ = fig.colorbar(pl, ax=ax[j, k], fraction=0.15)

    def evaluate(self):
        pass

    def get_model(self):
        return self.model
