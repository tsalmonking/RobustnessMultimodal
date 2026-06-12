import os
import json
import pandas as pd

# Custom imports
from utils import create_late_fusion, create_late_fusion_parameters

if __name__ == "__main__":
    BASE = "data/Recovery/classification_results/perturbed"

    text_csv = os.path.join(BASE, "text", "perturbed_results.csv")
    image_csv = os.path.join(BASE, "image", "perturbed_results.csv")

    text_params = os.path.join(BASE, "text", "parameters.json")
    image_params = os.path.join(BASE, "image", "parameters.json")

    for fusion in ["mean", "min", "max"]:
        out_dir = os.path.join(BASE, "late-fusion", fusion)
        create_late_fusion(text_csv, image_csv, out_dir, fusion)
        create_late_fusion_parameters(text_params, image_params, out_dir, fusion)