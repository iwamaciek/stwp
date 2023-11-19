import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.7
BATCH_SIZE = 2
FH = 1
INPUT_SIZE = 3
DATA_PATH = "../data2022.grib"
