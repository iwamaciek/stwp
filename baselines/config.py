import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.7
BATCH_SIZE = 4
FH = 2
INPUT_SIZE = 3
DATA_PATH = "../data/data2021_2022.grib"