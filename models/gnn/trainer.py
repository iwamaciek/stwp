import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import cartopy.crs as ccrs

from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.gnn.processor import NNDataProcessor
from models.data_processor import DataProcessor
from models.config import config as cfg
from models.gnn.callbacks import (
    LRAdjustCallback,
    CkptCallback,
    EarlyStoppingCallback,
)
from models.gnn.gnn_module import GNNModule
from utils.draw_functions import draw_poland
from datetime import datetime

from utils.trig_encode import trig_decode


class Trainer:
    def __init__(
        self,
        architecture="trans",
        hidden_dim=32,
        lr=1e-3,
        gamma=0.5,
        subset=None,
        spatial_mapping=True,
        additional_encodings=True,
        test_shuffle=True,
    ):
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
        self.nn_proc = NNDataProcessor(additional_encodings=additional_encodings, test_shuffle=test_shuffle)
        self.init_data_process()

        self.model = None
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.init_architecture()

        self.lr = lr
        self.gamma = gamma
        self.criterion = torch.nn.L1Loss()
        # self.criterion = torch.mean(torch.log(torch.cosh((y-y_hat) + 1e-12))) # LogCosh
        # self.criterion = torch.nn.HuberLoss()
        self.optimizer = None
        self.lr_callback = None
        self.ckpt_callback = None
        self.early_stop_callback = None
        self.init_train_details()

    def update_config(self, c):
        self.cfg = c
        self.init_architecture()
        self.update_data_process()
        self.init_train_details()

    def init_data_process(self):
        self.nn_proc.preprocess(subset=self.subset)
        self.train_loader = self.nn_proc.train_loader
        self.val_loader = self.nn_proc.val_loader
        self.test_loader = self.nn_proc.test_loader
        self.feature_list = self.nn_proc.feature_list
        self.features = len(self.feature_list)
        (_, self.latitude, self.longitude, self.features) = self.nn_proc.get_shapes()
        self.constants = (
            self.nn_proc.num_spatial_constants + self.nn_proc.num_temporal_constants
        )
        self.edge_index = self.nn_proc.edge_index
        self.edge_weights = self.nn_proc.edge_weights
        self.edge_attr = self.nn_proc.edge_attr
        self.scalers = self.nn_proc.scalers
        self.train_size = len(self.train_loader)
        self.val_size = len(self.val_loader)
        self.test_size = len(self.test_loader)
        self.spatial_mapping = self.spatial_mapping
        if self.subset is None:
            self.subset = self.train_size

    def update_data_process(self):
        self.nn_proc.update(self.cfg)
        self.init_data_process()

    def init_architecture(self):
        init_dict = {
            "arch": self.architecture,
            "input_features": self.features,
            "output_features": self.features,
            "edge_dim": self.edge_attr.size(-1),
            "hidden_dim": self.hidden_dim,
            "input_t_dim": self.nn_proc.num_temporal_constants,
            "input_s_dim": self.nn_proc.num_spatial_constants,
            "input_size": self.cfg.INPUT_SIZE,
            "fh": self.cfg.FH,
            "num_graph_cells": self.cfg.GRAPH_CELLS,
        }
        self.model = GNNModule(**init_dict).to(self.cfg.DEVICE)
        # self.model = torch.compile(self.model)

    def init_train_details(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_callback = LRAdjustCallback(self.optimizer, gamma=self.gamma)
        self.ckpt_callback = CkptCallback(self.model)
        self.early_stop_callback = EarlyStoppingCallback()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, num_epochs=50, verbose=False):
        # gradient_clip = 32
        start = time.time()

        val_loss_list = []
        train_loss_list = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                y_hat = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.time, batch.pos
                )
                batch_y = batch.y

                if self.spatial_mapping:
                    y_hat = self.nn_proc.map_latitude_longitude_span(y_hat)
                    batch_y = self.nn_proc.map_latitude_longitude_span(batch.y)

                loss = self.criterion(y_hat, batch_y)
                loss.backward()

                # nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / (self.subset * self.cfg.BATCH_SIZE)
            train_loss_list.append(avg_loss)
            last_lr = self.optimizer.param_groups[0]["lr"]
            
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.5f}, lr: {last_lr}"
                )

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.val_loader:
                    y_hat = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.time,
                        batch.pos,
                    )
                    batch_y = batch.y

                    if self.spatial_mapping:
                        y_hat = self.nn_proc.map_latitude_longitude_span(y_hat)
                        batch_y = self.nn_proc.map_latitude_longitude_span(batch.y)

                    loss = self.criterion(y_hat, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (
                min(self.subset, self.val_size) * self.cfg.BATCH_SIZE
            )
            val_loss_list.append(avg_val_loss)

            if verbose:
                print(f"Val Loss: {avg_val_loss:.5f}\n---------")

            self.lr_callback.step(avg_val_loss)
            self.ckpt_callback.step(avg_val_loss)
            self.early_stop_callback.step(avg_val_loss)
            if self.early_stop_callback.early_stop:
                break

        end = time.time()
        if verbose:
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

    def predict(self, X, y, edge_index, edge_attr, s, t, inverse_norm=True):
        y = y.reshape((-1, self.latitude, self.longitude, self.features, self.cfg.FH))
        y_hat = self.model(X, edge_index, edge_attr, t, s).reshape(
            (-1, self.latitude, self.longitude, self.features, self.cfg.FH)
        )

        y = y.cpu().detach().numpy()
        y_hat = y_hat.cpu().detach().numpy()

        if inverse_norm:
            y_shape = (self.latitude, self.longitude, self.cfg.FH)
            for i in range(self.features):
                for j in range(y_hat.shape[0]):
                    yi = y[j, ..., i, :].copy().reshape(-1, 1)
                    yhat_i = y_hat[j, ..., i, :].copy().reshape(-1, 1)

                    y[j, ..., i, :] = (
                        self.scalers[i].inverse_transform(yi).reshape(y_shape)
                    )
                    y_hat[j, ..., i, :] = (
                        self.scalers[i].inverse_transform(yhat_i).reshape(y_shape)
                    )
        if inverse_norm:
            y_hat = self.clip_total_cloud_cover(y_hat)
        return y, y_hat

    def plot_predictions(self, data_type="test", pretty=False):
        if data_type == "train":
            sample = next(iter(self.train_loader))
        elif data_type == "test":
            sample = next(iter(self.test_loader))
        elif data_type == "val":
            sample = next(iter(self.val_loader))
        else:
            print("Invalid type: (train, test, val)")
            raise ValueError

        if pretty:
            lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
            spatial = {
                "lat_span": lat_span,
                "lon_span": lon_span,
                "spatial_limits": spatial_limits,
            }

        X, y = sample.x, sample.y
        y, y_hat = self.predict(
            X, y, sample.edge_index, sample.edge_attr, sample.pos, sample.time
        )
        latitude, longitude = self.latitude, self.longitude

        if self.spatial_mapping:
            y_hat = self.nn_proc.map_latitude_longitude_span(y_hat, flat=False)
            y = self.nn_proc.map_latitude_longitude_span(y, flat=False)
            latitude, longitude = y_hat.shape[1:3]

        for i in range(self.cfg.BATCH_SIZE):
            if pretty:
                fig, axs = plt.subplots(
                    self.features,
                    3 * self.cfg.FH,
                    figsize=(10 * self.cfg.FH, 3 * self.features),
                    subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
                )
            else:
                fig, ax = plt.subplots(
                    self.features,
                    3 * self.cfg.FH,
                    figsize=(10 * self.cfg.FH, 3 * self.features),
                )

            for j, feature_name in enumerate(self.feature_list):
                for k in range(3 * self.cfg.FH):
                    ts = k // 3
                    if pretty:
                        ax = axs[j, k]
                    if k % 3 == 0:
                        title = rf"$X_{{{feature_name},t+{ts + 1}}}$"
                        value = y[i, ..., j, ts]
                        cmap = plt.cm.coolwarm
                    elif k % 3 == 1:
                        title = rf"$\hat{{X}}_{{{feature_name},t+{ts + 1}}}$"
                        value = y_hat[i, ..., j, ts]
                        cmap = plt.cm.coolwarm
                    else:
                        title = rf"$|X - \hat{{X}}|_{{{feature_name},t+{ts + 1}}}$"
                        value = np.abs(y[i, ..., j, ts] - y_hat[i, ..., j, ts])
                        cmap = "binary"

                    if pretty:
                        draw_poland(ax, value, title, cmap, **spatial)
                    else:
                        pl = ax[j, k].imshow(
                            value.reshape(latitude, longitude), cmap=cmap
                        )
                        ax[j, k].set_title(title)
                        ax[j, k].axis("off")
                        _ = fig.colorbar(pl, ax=ax[j, k], fraction=0.15)

        self.calculate_metrics(y_hat, y)

    def evaluate(self, data_type="test", verbose=True, inverse_norm=True, begin=None, end=None):
        if data_type == "train":
            loader = self.train_loader
        elif data_type == "test":
            loader = self.test_loader
        elif data_type == "val":
            loader = self.val_loader
        else:
            print("Invalid type: (train, test, val)")
            raise ValueError

        y = np.empty((0, self.latitude, self.longitude, self.features, self.cfg.FH))
        y_hat = np.empty((0, self.latitude, self.longitude, self.features, self.cfg.FH))
        for i, batch in enumerate(loader):
            print(i)
            if begin is not None and end is not None:
                v_sin = batch.time[0].item()
                v_cos = batch.time[1].item()
                # ts = np.arctan2(v_sin, v_cos) / (2 * np.pi) * 365
                ts = trig_decode(v_sin, v_cos, 366)
                print(f"ts: {ts}, begin: {begin}, end: {end}")
                if begin > ts:
                    print("continue")
                    continue
                elif end < ts:
                    print("break")
                    break
            y_i, y_hat_i = self.predict(
                batch.x,
                batch.y,
                batch.edge_index,
                batch.edge_attr,
                batch.pos,
                batch.time,
                inverse_norm=inverse_norm,
            )
            y = np.concatenate((y, y_i), axis=0)
            y_hat = np.concatenate((y_hat, y_hat_i), axis=0)


            print(f'y_hat: {y_hat.shape}, y_hat_i: {y_hat_i.shape}, y_i: {y_i.shape}, batch.x: {batch.x.shape}, y: {y.shape}')

        if self.spatial_mapping:
            y_hat = self.nn_proc.map_latitude_longitude_span(y_hat, flat=False)
            y = self.nn_proc.map_latitude_longitude_span(y, flat=False)
        
        # print(y_hat.shape)
        # print(y_hat)
        try:
            return self.calculate_metrics(y_hat, y, verbose=verbose), y_hat
        
        except Exception as e:
            print(e)
            return None, None

    def autoreg_evaluate(self, data_type="test", fh=2, verbose=True, inverse_norm=True):
        # Only works for fh=1 for now
        self.cfg.BATCH_SIZE = 1
        self.cfg.FH = fh
        self.update_data_process()
        self.cfg.FH = 1

        if data_type == "train":
            loader = self.train_loader
        elif data_type == "test":
            loader = self.test_loader
        elif data_type == "val":
            loader = self.val_loader
        else:
            print("Invalid type: (train, test, val)")
            raise ValueError

        y = torch.empty((0, self.latitude, self.longitude, self.features, fh)).to(
            self.cfg.DEVICE
        )
        y_hat = torch.empty((0, self.latitude, self.longitude, self.features, fh)).to(
            self.cfg.DEVICE
        )
        y_shape = (self.latitude * self.longitude, self.features, 1)
        for batch in loader:
            y_hat_autoreg_i = torch.zeros_like(batch.y)
            y_i = torch.zeros_like(batch.y)
            for t in range(fh):
                input_batch = batch.clone()
                input_batch.y = input_batch.y[..., t : t + 1]
                if t == 0:
                    y_it, y_hat_it = self.predict(
                        input_batch.x,
                        input_batch.y,
                        input_batch.edge_index,
                        input_batch.edge_attr,
                        input_batch.pos,
                        input_batch.time,
                        inverse_norm=inverse_norm,
                    )
                else:
                    input_batch.x = torch.cat(
                        (input_batch.x[..., :-t], y_hat_autoreg_i[..., :t]), dim=-1
                    )
                    y_it, y_hat_it = self.predict(
                        input_batch.x,
                        input_batch.y,
                        input_batch.edge_index,
                        input_batch.edge_attr,
                        input_batch.pos,
                        input_batch.time,
                        inverse_norm=inverse_norm,
                    )
                y_hat_i = torch.from_numpy(y_hat_it).to(self.cfg.DEVICE)
                y_it = torch.from_numpy(y_it).to(self.cfg.DEVICE)
                y_hat_autoreg_i[..., t : t + 1] = y_hat_i.reshape(y_shape)
                y_i[..., t : t + 1] = y_it.reshape(y_shape)

            y = torch.cat(
                (y, y_i.reshape(1, self.latitude, self.longitude, self.features, fh)),
                dim=0,
            )
            y_hat = torch.cat(
                (
                    y_hat,
                    y_hat_autoreg_i.reshape(
                        1, self.latitude, self.longitude, self.features, fh
                    ),
                ),
                dim=0,
            )

        y_hat = y_hat.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

        if self.spatial_mapping:
            y_hat = self.nn_proc.map_latitude_longitude_span(y_hat, flat=False)
            y = self.nn_proc.map_latitude_longitude_span(y, flat=False)

        self.cfg.FH = 1
        return self.calculate_metrics(y_hat, y, verbose=verbose), y_hat

    def calculate_metrics(self, y_hat, y, verbose=False):
        rmse_features = []
        mae_features = []
        for i, feature_name in enumerate(self.feature_list):
            y_fi = y[..., i, :].reshape(-1, 1)
            y_hat_fi = y_hat[..., i, :].reshape(-1, 1)
            rmse = np.sqrt(mean_squared_error(y_hat_fi, y_fi))
            mae = mean_absolute_error(y_hat_fi, y_fi)
            if verbose:
                print(
                    f"RMSE for {feature_name}: {rmse}; MAE for {feature_name}: {mae};"
                )
            rmse_features.append(rmse)
            mae_features.append(mae)
        return rmse_features, mae_features

    def save_prediction_tensor(self, y_hat, path=None):
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().detach().numpy()
        elif not isinstance(y_hat, np.ndarray):
            raise ValueError(
                "Input y_hat should be either a PyTorch Tensor or a NumPy array."
            )
        if path is None:
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            path = f"../data/pred/{self.architecture}_{t}.npy"
        np.save(path, y_hat)

    @staticmethod
    def clip_total_cloud_cover(y_hat, idx=2):
        y_hat[..., idx, :] = np.clip(y_hat[..., idx, :], 0, 1)
        return y_hat
