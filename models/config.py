import torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_RATIO = 1 / 3
    BATCH_SIZE = 8
    FH = 1
    # AUTOREG_FH = None
    INPUT_SIZE = 6
    R = 2
    GRAPH_CELLS = 5
    SCALER_TYPE = "standard"
    DATA_PATH = "../../data2019-2021_BIG.grib"
    RANDOM_STATE = 42
    INPUT_DIMS = (32, 48)
    OUTPUT_DIMS = (25, 45)




config = Config()
