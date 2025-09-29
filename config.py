# Config semplice: modifica i path e i parametri
import os

import sys
sys.path.append(os.path.abspath("../StatoDellArte/Projects/It-Is-Fake-Or-Not"))  # per importare il modello

ROOT = os.path.dirname(__file__)
WEIGHTS_PATH = "model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt"   # TODO: path ai pesi .pth
MODEL_IMPORT_PATH = "themis_model"  # TODO: modulo python che contiene la classe del modello
MODEL_CLASS_NAME = "Themis"  # TODO: nome della classe da importare
DATASET_PATH = "Data/ReCOVery"  # path alla cartella del dataset
DATA_CSV = os.path.join(DATASET_PATH, "recovery-news-data.csv")  # CSV con colonne: image_path,text,label
OUTPUT_DIR = "results"
BATCH_SIZE = 8
NUM_WORKERS = 4