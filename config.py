import os

ROOT = os.path.dirname(__file__)
WEIGHTS_PATH = "model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt"
MODEL_IMPORT_PATH = "themis_model"
MODEL_CLASS_NAME = "Themis"
DATASET_PATH = "Data/ReCOVery"
DATA_CSV = os.path.join(DATASET_PATH, "recovery.csv")
OUTPUT_DIR = "results"
NUM_WORKERS = 0