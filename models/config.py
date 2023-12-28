import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_RATIO = 1 / 3
    BATCH_SIZE = 1
    FH = 1
    INPUT_SIZE = 5
    R = 2
    DATA_PATH = "../data/data2019-2021.grib"
    RANDOM_STATE = 42
    INPUT_DIMS = (32, 48)
    OUTPUT_DIMS = (25, 45)


config = Config()