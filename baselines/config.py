import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.5
BATCH_SIZE = 4
FH = 1
INPUT_SIZE = 3
DATA_PATH = "../data2021-2022.grib"
