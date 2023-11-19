import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.7
FH = 1
INPUT_SIZE = 3
DATAPATH = "../data2022.grib"
