from baselines.cnn.processor import NNDataProcessor
from baselines.cnn.cnn import UNet
from baselines.cnn.config import DEVICE, FH, BATCH_SIZE, INPUT_SIZE
from baselines.cnn.callbacks import (
    LRAdjustCallback,
    CkptCallback,
    EarlyStoppingCallback,
)
from torch.optim.lr_scheduler import StepLR
import torch
import time
import numpy as np
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, base_units=16, lr=0.01, gamma=0.5, subset=None) -> None:
        self.data_processor = NNDataProcessor()
        self.data_processor.preprocess(subset=subset)
        self.train_loader = self.data_processor.train_loader
        self.test_loader = self.data_processor.test_loader

        (
            _,
            self.latitude,
            self.longitude,
            self.features,
        ) = self.data_processor.get_shapes()

        self.scalers = self.data_processor.scalers
        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)

        if subset is None:
            self.subset = self.train_size
        else:
            self.subset = subset

        self.model = UNet(features=self.features, s=INPUT_SIZE, fh=FH, base_units=base_units).to(DEVICE)

        self.criterion = torch.nn.MSELoss()
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters())#, lr=self.lr
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)

        # Callbacks
        self.lr_callback = LRAdjustCallback(self.optimizer, self.scheduler)
        self.ckpt_callback = CkptCallback(self.model)
        self.early_stop_callback = EarlyStoppingCallback(patience=30)

    def train(self, num_epochs=100):
        train_loss_list = []
        val_loss_list = []

        start = time.time()

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                inputs = batch[:, :INPUT_SIZE, ...].reshape(batch.shape[0], INPUT_SIZE*self.features, batch.shape[3], batch.shape[4]).to(DEVICE)
                labels = batch[:, -FH:, ...].reshape(batch.shape[0], FH*self.features, batch.shape[3], batch.shape[4]).to(DEVICE)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / (self.subset * BATCH_SIZE)
            last_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{num_epochs}:\nTrain Loss: {avg_loss:.4f}, Last LR: {last_lr:.4f}")
            train_loss_list.append(avg_loss)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.test_loader:
                    inputs = batch[:, :INPUT_SIZE, ...].reshape(batch.shape[0], INPUT_SIZE*self.features, batch.shape[3], batch.shape[4]).to(DEVICE)
                    labels = batch[:, -FH:, ...].reshape(batch.shape[0], FH*self.features, batch.shape[3], batch.shape[4]).to(DEVICE)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (min(self.subset, self.test_size) * BATCH_SIZE)
            print(f"Val Loss: {avg_val_loss:.4f}\n---------")
            val_loss_list.append(avg_val_loss)

            # self.lr_callback.step(avg_loss)
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
        y = y.cpu().detach().numpy()
        y = y.transpose((0, 2, 3, 1))
        y = y.reshape(y.shape[:3]+(FH, self.features))

        y_hat = self.model(X)
        y_hat = y_hat.cpu().detach().numpy()
        y_hat = y_hat.transpose((0, 2, 3, 1))
        y_hat = y_hat.reshape(y_hat.shape[:3]+(FH, self.features))

        for i in range(self.features):
            og_shape = y_hat[..., i].shape
            y_hat[..., i] = self.scalers[i].inverse_transform(y_hat[..., i].reshape(-1, 1)).reshape(og_shape)
            og_shape = y[..., i].shape
            y[..., i] = self.scalers[i].inverse_transform(y[..., i].reshape(-1, 1)).reshape(og_shape)

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

        X = sample[:, :INPUT_SIZE, ...].reshape(sample.shape[0], INPUT_SIZE*self.features, sample.shape[3], sample.shape[4]).to(DEVICE)
        y = sample[:, -FH:, ...].reshape(sample.shape[0], FH*self.features, sample.shape[3], sample.shape[4])
        y, y_hat = self.inverse_normalization_predict(X, y)

        y = y.transpose((0, 1, 2, 4, 3))
        y_hat = y_hat.transpose((0, 1, 2, 4, 3))

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
            y_i = y_i.transpose((0, 1, 2, 4, 3))
            y_hat_i = y_hat_i.transpose((0, 1, 2, 4, 3))
            y = np.concatenate((y, y_i), axis=0)
            y_hat = np.concatenate((y_hat, y_hat_i), axis=0)

        for i in range(self.features):
            y_fi = y[..., i, :].reshape(-1, 1)
            y_hat_fi = y_hat[..., i, :].reshape(-1, 1)
            rmse_fi = np.mean((y_fi - y_hat_fi) ** 2) ** (1 / 2)
            print(f"RMSE for f{i}: {rmse_fi}")

    def get_model(self):
        return self.model
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))