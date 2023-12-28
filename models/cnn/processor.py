from models.config import config as cfg

# from models.gnn.processor import NNDataProcessor

# sys.path.append("..")
from models.gnn.processor import NNDataProcessor


class CNNDataProcessor(NNDataProcessor):
    def __init__(
        self,
        spatial_encoding=False,
        temporal_encoding=False,
        additional_encodings=False,
    ) -> None:
        super().__init__(
            spatial_encoding=spatial_encoding,
            temporal_encoding=temporal_encoding,
            additional_encodings=additional_encodings,
        )

    def preprocess(self, subset=None):
        X_train, X_test, y_train, y_test = self.train_val_test_split()
        X, y = self.fit_transform_scalers(
            X_train, X_test, y_train, y_test, scaler_type=self.cfg.SCALER_TYPE
        )
        X = X.transpose((0, 1, 3, 2))
        y = y.transpose((0, 1, 3, 2))
        X = X.reshape(
            -1,
            self.num_latitudes,
            self.num_longitudes,
            self.cfg.INPUT_SIZE,
            self.num_features,
        )
        y = y.reshape(
            -1, self.num_latitudes, self.num_longitudes, cfg.FH, self.num_features
        )
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(
            X, y, subset
        )

    def map_latitude_longitude_span(
        self, input_tensor, old_span=(32, 48), new_span=(25, 45), flat=False
    ):
        """
        Maps latitude-longitude span e.g. (32,48) -> (25,45)
        """
        if flat:
            input_tensor = input_tensor.reshape(
                (
                    self.cfg.BATCH_SIZE,
                    self.num_features,
                    self.num_latitudes,
                    self.num_longitudes,
                    1,
                )
            )

        old_lat, old_lon = old_span
        new_lat, new_lon = new_span

        lat_diff = old_lat - new_lat
        left_lat = lat_diff // 2
        right_lat = new_lat + lat_diff - left_lat - 1

        lon_diff = old_lon - new_lon
        up_lon = lon_diff // 2
        down_lon = new_lon + lon_diff - up_lon - 1

        if len(input_tensor.shape) == 4:
            mapped_tensor = input_tensor[:, :, left_lat:right_lat, up_lon:down_lon]
        else:
            mapped_tensor = input_tensor[:, left_lat:right_lat, up_lon:down_lon, ...]

        return mapped_tensor
