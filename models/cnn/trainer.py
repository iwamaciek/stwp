from models.cnn.processor import CNNDataProcessor
from models.cnn.cnn import UNet
from models.config import config as cfg
from models.gnn.callbacks import (
    LRAdjustCallback,
    CkptCallback,
    EarlyStoppingCallback,
)
from models.gnn.trainer import Trainer as GNNTrainer
import torch
import time
import numpy as np
from datetime import datetime


class Trainer(GNNTrainer):
    def __init__(
        self, base_units=16, lr=0.001, gamma=0.5, subset=None, spatial_mapping=True
    ) -> None:
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.feature_list = None
        self.features = None
        self.constants = None
        self.edge_index = None
        self.edge_weights = None
        self.edge_attr = None
        self.scalers = None
        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.spatial_mapping = spatial_mapping
        self.subset = subset

        self.cfg = cfg
        self.nn_proc = CNNDataProcessor(additional_encodings=True)
        self.init_data_process()

        self.model = None
        self.base_units = base_units
        self.init_architecture()

        self.lr = lr
        self.gamma = gamma
        self.criterion = torch.nn.L1Loss()
        self.optimizer = None
        self.lr_callback = None
        self.ckpt_callback = None
        self.early_stop_callback = EarlyStoppingCallback()
        self.init_train_details()

    def init_architecture(self):
        self.model = UNet(
            features=self.features,
            spatial_features=self.nn_proc.num_spatial_constants,
            temporal_features=self.nn_proc.num_temporal_constants,
            out_features=self.features,
            s=self.cfg.INPUT_SIZE,
            fh=self.cfg.FH,
            base_units=self.base_units,
        ).to(self.cfg.DEVICE)

    def train(self, num_epochs=100):
        train_loss_list = []
        val_loss_list = []

        start = time.time()

        for epoch in range(num_epochs):
            gradient_clip = 32
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                # if batch.x.shape[0] < self.cfg.BATCH_SIZE:
                #     continue
                inputs = (
                    batch.x.reshape(
                        -1,
                        self.latitude,
                        self.longitude,
                        self.cfg.INPUT_SIZE * self.features,
                    )
                    .permute((0, 3, 1, 2))
                    .to(self.cfg.DEVICE)
                )
                labels = (
                    batch.y.reshape(
                        -1, self.latitude, self.longitude, self.cfg.FH * self.features
                    )
                    .permute((0, 3, 1, 2))
                    .to(self.cfg.DEVICE)
                )
                t = batch.time.to(self.cfg.DEVICE)
                s = batch.pos.to(self.cfg.DEVICE)
                self.optimizer.zero_grad()

                outputs = self.model(inputs, t, s)

                if self.spatial_mapping:
                    labels = self.nn_proc.map_latitude_longitude_span(labels)
                    outputs = self.nn_proc.map_latitude_longitude_span(outputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / (self.subset * self.cfg.BATCH_SIZE)
            last_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.5f}, lr: {last_lr}"
            )
            train_loss_list.append(avg_loss)

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.val_loader:
                    # if batch.x.shape[0] < self.cfg.BATCH_SIZE:
                    #     continue
                    inputs = (
                        batch.x.reshape(
                            -1,
                            self.latitude,
                            self.longitude,
                            self.cfg.INPUT_SIZE * self.features,
                        )
                        .permute((0, 3, 1, 2))
                        .to(self.cfg.DEVICE)
                    )
                    labels = (
                        batch.y.reshape(
                            -1,
                            self.latitude,
                            self.longitude,
                            self.cfg.FH * self.features,
                        )
                        .permute((0, 3, 1, 2))
                        .to(self.cfg.DEVICE)
                    )
                    t = batch.time.to(self.cfg.DEVICE)
                    s = batch.pos.to(self.cfg.DEVICE)

                    outputs = self.model(inputs, t, s)

                    if self.spatial_mapping:
                        labels = self.nn_proc.map_latitude_longitude_span(labels)
                        outputs = self.nn_proc.map_latitude_longitude_span(outputs)

                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (
                min(self.subset, self.val_size) * self.cfg.BATCH_SIZE
            )
            print(f"Val Loss: {avg_val_loss:.5f}\n---------")
            val_loss_list.append(avg_val_loss)

            self.lr_callback.step(val_loss)
            self.ckpt_callback.step(avg_val_loss)
            self.early_stop_callback.step(val_loss)
            if self.early_stop_callback.early_stop:
                break

        end = time.time()
        print(f"{end - start} [s]")
        self.plot_loss(val_loss_list, train_loss_list)

    def predict(self, X, y, edge_index, edge_attr, pos, time, inverse_norm=True):
        X = (
            X.reshape(
                -1, self.latitude, self.longitude, self.cfg.INPUT_SIZE * self.features
            )
            .permute((0, 3, 1, 2))
            .to(self.cfg.DEVICE)
        )
        y_hat = self.model(X, time, pos)
        y_hat = (
            y_hat.permute((0, 2, 3, 1))
            .reshape(-1, self.latitude, self.longitude, self.cfg.FH, self.features)
            .permute((0, 1, 2, 4, 3))
        )
        y_hat = y_hat.cpu().detach().numpy()

        y = y.reshape(
            -1, self.latitude, self.longitude, self.cfg.FH, self.features
        ).permute((0, 1, 2, 4, 3))
        y = y.cpu().detach().numpy()

        yshape = (self.latitude, self.longitude, self.cfg.FH)

        for i in range(self.features):
            for j in range(y_hat.shape[0]):
                yi = y[j, ..., i, :].copy().reshape(-1, 1)
                yhat_i = y_hat[j, ..., i, :].copy().reshape(-1, 1)

                y[j, ..., i, :] = self.scalers[i].inverse_transform(yi).reshape(yshape)
                y_hat[j, ..., i, :] = (
                    self.scalers[i].inverse_transform(yhat_i).reshape(yshape)
                )
        if inverse_norm:
            y_hat = self.clip_total_cloud_cover(y_hat)
        return y, y_hat

    def save_prediction_tensor(self, y_hat, path=None):
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().detach().numpy()
        elif not isinstance(y_hat, np.ndarray):
            raise ValueError(
                "Input y_hat should be either a PyTorch Tensor or a NumPy array."
            )
        if path is None:
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            path = f"../data/pred/unet_{t}.npy"
        np.save(path, y_hat)
