import os

WEIGHTS_PATH = "model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt"
DATASET_PATH = "Data/ReCOVery"
DATA_CSV = os.path.join(DATASET_PATH, "recovery.csv")
OUTPUT_DIR = "results"
