import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import json
import cartopy.crs as ccrs

from sklearn.metrics import mean_squared_error, mean_absolute_error
from baselines.gnn.processor import NNDataProcessor
from baselines.data_processor import DataProcessor
from baselines.config import DEVICE, FH, BATCH_SIZE
from baselines.gnn.callbacks import (
    LRAdjustCallback,
    CkptCallback,
    EarlyStoppingCallback,
)
from baselines.gnn.cgc_conv import CrystalGNN
from baselines.gnn.transformer_conv import TransformerGNN
from baselines.gnn.gat_conv import GATConvNN
from baselines.gnn.gen_conv import GENConvNN
from baselines.gnn.pdn_conv import PDNConvNN
from utils.draw_functions import draw_poland


class Trainer:
    def __init__(
        self,
        architecture="cgcn",
        hidden_dim=64,
        lr=0.01,
        gamma=0.5,
        subset=None,
        spatial_mapping=True,
        additional_encodings=True,
    ):

        # Full data preprocessing for nn input run in NNDataProcessor constructor
        # If subset param is given train_data and test_data will have len=subset
        self.nn_proc = NNDataProcessor(additional_encodings=additional_encodings)
        self.nn_proc.preprocess(subset=subset)
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
        self.spatial_mapping = spatial_mapping
        if subset is None:
            self.subset = self.train_size
        else:
            self.subset = subset

        # Architecture details
        init_dict = {
            "input_features": self.features,
            "output_features": self.features,
            "edge_dim": self.edge_attr.size(-1),
            "hidden_dim": hidden_dim,
            "input_t_dim": self.nn_proc.num_temporal_constants,
            "input_s_dim": self.nn_proc.num_spatial_constants,
        }

        if architecture == "cgcn":
            self.model = CrystalGNN(**init_dict).to(DEVICE)
        elif architecture == "trans":
            self.model = TransformerGNN(**init_dict).to(DEVICE)
        elif architecture == "gat":
            self.model = GATConvNN(**init_dict).to(DEVICE)
        elif architecture == "gen":
            self.model = GENConvNN(**init_dict).to(DEVICE)
        elif architecture == "pdn":
            self.model = PDNConvNN(**init_dict).to(DEVICE)
        else:
            # TODO handling
            self.model = None

        # Does not improve performance
        # feature_means = torch.zeros(self.features).to(DEVICE)
        # for b in self.train_loader:
        #     feature_means += b.y.mean(dim=0).squeeze()
        # self.model.initialize_last_layer_bias(feature_means / len(self.train_loader))

        # Training details
        self.criterion = lambda output, target: (output - target).pow(2).sum()
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(), betas=(0.9, 0.95), weight_decay=0.1, lr=self.lr
        # )

        # Callbacks
        self.lr_callback = LRAdjustCallback(self.optimizer, gamma=self.gamma)
        self.ckpt_callback = CkptCallback(self.model)
        self.early_stop_callback = EarlyStoppingCallback()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, num_epochs=50):
        # gradient_clip = 32
        start = time.time()

        val_loss_list = []
        train_loss_list = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                batch = batch.to(DEVICE)
                y_hat = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.time, batch.pos
                )
                batch_y = batch.y

                if self.spatial_mapping:
                    y_hat = self.nn_proc.map_latitude_longitude_span(y_hat)
                    batch_y = self.nn_proc.map_latitude_longitude_span(batch.y)

                loss = self.criterion(y_hat, batch_y) / BATCH_SIZE
                loss.backward()

                # nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / self.subset
            train_loss_list.append(avg_loss)
            last_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, lr: {last_lr}"
            )

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in self.val_loader:
                    batch = batch.to(DEVICE)
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

                    loss = self.criterion(y_hat, batch_y) / BATCH_SIZE
                    val_loss += loss.item()

            avg_val_loss = val_loss / min(self.subset, self.val_size)
            val_loss_list.append(avg_val_loss)

            print(f"Val Loss: {avg_val_loss:.4f}\n---------")

            self.lr_callback.step(avg_val_loss)
            self.ckpt_callback.step(avg_val_loss)
            self.early_stop_callback.step(avg_val_loss)
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

    def inverse_normalization_predict(self, X, y, edge_index, edge_attr, s, t):
        y = y.reshape((-1, self.latitude, self.longitude, self.features, FH))
        y = y.cpu().detach().numpy()

        y_hat = self.model(X, edge_index, edge_attr, t, s)
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
        y, y_hat = self.inverse_normalization_predict(
            X, y, sample.edge_index, sample.edge_attr, sample.pos, sample.time
        )
        latitude, longitude = self.latitude, self.longitude

        if self.spatial_mapping:
            y_hat = self.nn_proc.map_latitude_longitude_span(y_hat, flat=False)
            y = self.nn_proc.map_latitude_longitude_span(y, flat=False)
            latitude, longitude = y_hat.shape[1:3]

        for i in range(BATCH_SIZE):
            if pretty:
                fig, axs = plt.subplots(
                    self.features,
                    3 * FH,
                    figsize=(10 * FH, 3 * self.features),
                    subplot_kw={"projection": ccrs.Mercator(central_longitude=40)},
                )
            else:
                fig, ax = plt.subplots(
                    self.features, 3 * FH, figsize=(10 * FH, 3 * self.features)
                )

            for j, feature_name in enumerate(self.feature_list):
                for k in range(3 * FH):
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

        self.calculate_matrics(y_hat, y)

    def predict_sample(self):
        sample = self.test_loader

    def evaluate(self, data_type="test"):
        if data_type == "train":
            loader = self.train_loader
        elif data_type == "test":
            loader = self.test_loader
        elif data_type == "val":
            loader = self.val_loader
        else:
            print("Invalid type: (train, test, val)")
            raise ValueError

        y = np.empty((0, self.latitude, self.longitude, self.features, FH))
        y_hat = np.empty((0, self.latitude, self.longitude, self.features, FH))
        for batch in loader:
            y_i, y_hat_i = self.inverse_normalization_predict(
                batch.x,
                batch.y,
                batch.edge_index,
                batch.edge_attr,
                batch.pos,
                batch.time,
            )
            y = np.concatenate((y, y_i), axis=0)
            y_hat = np.concatenate((y_hat, y_hat_i), axis=0)

        if self.spatial_mapping:
            y_hat = self.nn_proc.map_latitude_longitude_span(y_hat, flat=False)
            y = self.nn_proc.map_latitude_longitude_span(y, flat=False)
        self.calculate_matrics(y_hat, y)

    def calculate_matrics(self, y_hat, y):
        for i, feature_name in enumerate(self.feature_list):
            y_fi = y[..., i, :].reshape(-1, 1)
            y_hat_fi = y_hat[..., i, :].reshape(-1, 1)
            rmse = np.sqrt(mean_squared_error(y_hat_fi, y_fi))
            mae = mean_absolute_error(y_hat_fi, y_fi)
            print(f"RMSE for {feature_name}: {rmse}; MAE for {feature_name}: {mae};")

    def get_model(self):
        return self.model

    def predict_to_json(self, X=None, path="../data/data.json"):
        if X is None:
            X = next(iter(self.test_loader))  # batch size should be set to 1 !
        _, y_hat = self.inverse_normalization_predict(
            X.x, X.y, X.edge_index, X.edge_attr
        )
        y_hat = y_hat.reshape((self.latitude, self.longitude, self.features, -1))
        lat_span, lon_span, spatial_limits = DataProcessor.get_spatial_info()
        lat_span = list(lat_span[:, 0])
        lon_span = list(lon_span[0, :])

        json_data = {}

        for i, lat in enumerate(lat_span):
            json_data[lat] = {}
            for j, lon in enumerate(lon_span):
                json_data[lat][lon] = {}
                for k, feature in enumerate(self.feature_list):
                    json_data[lat][lon][feature] = {}
                    for ts in range(y_hat.shape[-1]):
                        json_data[lat][lon][feature][ts] = float(y_hat[i, j, k, ts])

        with open(path, "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
