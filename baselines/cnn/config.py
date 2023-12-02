import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.7
BATCH_SIZE = 1
FH = 2
INPUT_SIZE = 3
DATA_PATH = "../data2022-2div.grib"