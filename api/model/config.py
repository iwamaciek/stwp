import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_RATIO = 0
    BATCH_SIZE = 1
    FH = 5
    INPUT_SIZE = 5
    R = 2
    SCALER_TYPE = "standard"
    DATA_PATH = "../model/data/data.grib"
    RANDOM_STATE = 42
    INPUT_DIMS = (32, 48)
    OUTPUT_DIMS = (25, 45)


config = Config()
