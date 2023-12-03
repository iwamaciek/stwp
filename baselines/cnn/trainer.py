from baselines.cnn.processor import CNNDataProcessor
from baselines.cnn.cnn import UNet
from baselines.config import DEVICE, FH, BATCH_SIZE, INPUT_SIZE
from baselines.gnn.callbacks import (
    LRAdjustCallback,
    CkptCallback,
    EarlyStoppingCallback,
)
from baselines.gnn.trainer import Trainer as GNNTrainer
from torch.optim.lr_scheduler import StepLR
import torch
import time

class Trainer(GNNTrainer):
    def __init__(self, base_units=16, lr=0.001, gamma=0.001, subset=None) -> None:
        self.data_processor = CNNDataProcessor()
        self.data_processor.preprocess(subset=subset)
        self.train_loader = self.data_processor.train_loader
        self.test_loader = self.data_processor.test_loader
        self.feature_list = self.data_processor.feature_list

        (
            _,
            self.latitude,
            self.longitude,
            self.features,
        ) = self.data_processor.get_shapes()

        self.scalers = self.data_processor.scalers
        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)
        self.edge_index = None
        self.edge_weights = None

        if subset is None:
            self.subset = self.train_size
        else:
            self.subset = subset

        self.model = UNet(features=self.features, s=INPUT_SIZE, fh=FH, base_units=base_units).to(DEVICE)

        self.criterion = torch.nn.MSELoss()
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=self.gamma)

        # Callbacks
        self.lr_callback = LRAdjustCallback(self.optimizer, self.scheduler, epsilon=0.01)
        self.ckpt_callback = CkptCallback(self.model)
        self.early_stop_callback = EarlyStoppingCallback(patience=100)
        
    def train(self, num_epochs=100):
        train_loss_list = []
        val_loss_list = []

        start = time.time()

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                inputs = batch.x.reshape(-1, self.latitude, self.longitude, INPUT_SIZE*self.features).permute((0, 3, 1, 2)).to(DEVICE)
                labels = batch.y.reshape(-1, self.latitude, self.longitude, FH*self.features).permute((0, 3, 1, 2)).to(DEVICE)
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
                    inputs = batch.x.reshape(-1, self.latitude, self.longitude, INPUT_SIZE*self.features).permute((0, 3, 1, 2)).to(DEVICE)
                    labels = batch.y.reshape(-1, self.latitude, self.longitude, FH*self.features).permute((0, 3, 1, 2)).to(DEVICE)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (min(self.subset, self.test_size) * BATCH_SIZE)
            print(f"Val Loss: {avg_val_loss:.4f}\n---------")
            val_loss_list.append(avg_val_loss)

            self.lr_callback.step(val_loss)
            self.ckpt_callback.step(avg_val_loss)
            self.early_stop_callback.step(val_loss)
            if self.early_stop_callback.early_stop:
                break

        end = time.time()
        print(f"{end - start} [s]")
        self.plot_loss(val_loss_list, train_loss_list)

    def inverse_normalization_predict(self, X, y, *args):
        X = X.reshape(-1, self.latitude, self.longitude, INPUT_SIZE*self.features).permute((0, 3, 1, 2)).to(DEVICE)
        y_hat = self.model(X)
        y_hat = y_hat.permute((0, 2, 3, 1)).reshape(-1, self.latitude, self.longitude, FH, self.features).permute((0, 1, 2, 4, 3))
        y_hat = y_hat.cpu().detach().numpy()

        y = y.reshape(-1, self.latitude, self.longitude, FH, self.features).permute((0, 1, 2, 4, 3))
        y = y.cpu().detach().numpy()

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