import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.7
FH = 1
INPUT_SIZE = 6
DATAPATH = "../data2022.grib"
NUM_FEATURES = 2
