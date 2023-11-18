import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from torch_geometric_temporal.nn.recurrent import A3TGCN
from baselines.gnn.processor import preprocess
from baselines.gnn.config import DEVICE, FH, TRAIN_RATIO


class TemporalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tgnn = A3TGCN(
            in_channels=input_dim, out_channels=hidden_dim, periods=output_dim
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.linear = torch.nn.Linear(hidden_dim, output_dim * input_dim)

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.batch_norm(h)
        h = self.dropout(h)
        h = self.linear(h)
        return h.view(-1, x.size(1), self.output_dim)


class Trainer:
    def __init__(self, hidden_dim=2048):
        self.dataset, self.scalers, self.shapes = preprocess()
        self.samples, self.latitude, self.longitude, self.features = self.shapes
        self.model = TemporalGNN(self.features, hidden_dim, FH).to(DEVICE)
        self.edge_index = self.dataset[0].edge_index
        self.train_size = int(self.samples * TRAIN_RATIO)
        self.test_size = self.samples - self.train_size
        self.train_data, self.test_data = (
            self.dataset[: self.train_size],
            self.dataset[self.train_size :],
        )

    def train(self, num_epochs=50, subset=None, path="model_state.pt"):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # criterion = nn.MSELoss()
        # criterion = criterion.to(DEVICE)

        start = time.time()

        if subset is None:
            subset = self.train_size

        val_loss_list = []
        train_loss_list = []
        min_val_loss = np.inf

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_data[:subset]:
                optimizer.zero_grad()
                y_hat = self.model(batch.x, batch.edge_index)
                # loss = criterion(y_hat, batch.y)
                loss = torch.sum((y_hat - batch.y) ** 2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / self.train_size
            print(f"Epoch {epoch + 1}/{num_epochs}\nTrain Loss: {avg_loss:.4f}")
            train_loss_list.append(avg_loss)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.test_data[:subset]:
                    y_hat = self.model(batch.x, batch.edge_index)
                    # loss = criterion(y_hat, batch.y)
                    loss = torch.sum((y_hat - batch.y) ** 2)
                    val_loss += loss.item()

            avg_val_loss = val_loss / self.test_size
            print(f"Val Loss: {avg_val_loss:.4f}\n---------")
            val_loss_list.append(avg_val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), path)

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

        y_hat = self.model(X, self.edge_index)
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
