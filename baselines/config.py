import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 1 / 3
BATCH_SIZE = 4
FH = 1
INPUT_SIZE = 5
R = 2
DATA_PATH = "../data2020-2022.grib"
RANDOM_STATE = 42
