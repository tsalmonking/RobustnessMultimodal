import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Custom imports
from utils import compute_metrics, plot_confusion_matrix, build_curve_name, update_roc_cache, regenerate_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["clean", "perturbed"])
    parser.add_argument("--modality", type=str, choices=["feature-fusion", "intermediate-fusion", "late-fusion", "text", "image"])
    parser.add_argument("--mode",  type=str, choices=["mean", "min", "max", "text-perturbed", "image-perturbed"])
    parser.add_argument("--roc-set", type=str, help="Name of ROC comparison group")
    args = parser.parse_args()

    base = f"data/Recovery/classification_results/{args.type}/{args.modality}"
    if args.modality == "late-fusion":
        base = f"data/Recovery/classification_results/{args.type}/{args.modality}/{args.mode}"
    elif args.modality == "feature-fusion" or args.modality == "intermediate-fusion":
        if args.mode is not None:
            base = f"data/Recovery/classification_results/{args.type}/{args.modality}/{args.mode}"

    if args.type == "clean":
        df = pd.read_csv(f"{base}/results.csv")
    else:
        df = pd.read_csv(f"{base}/perturbed_results.csv")

    y_true = df["label"]
    y_pred = df["pred"]
    scores = df["score"]

    y_true = 1-np.asarray(y_true)
    y_pred = 1-np.asarray(y_pred)
    scores = 1-np.asarray(scores)

    if os.path.exists(os.path.join(base, "metrics.json")):
        fpr, tpr, thr = roc_curve(y_true, scores)
        auc_score = auc(fpr, tpr)
    else:
        metrics, fpr, tpr, cm = compute_metrics(y_true, y_pred, scores)
        auc_score = metrics["auc"]
        plot_confusion_matrix(cm, range(cm.shape[0]), os.path.join(base, "confusion_matrix.png"))
        with open(os.path.join(base, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
    
    if args.roc_set is not None:
        curve_name = build_curve_name(args)
        roc_cache = update_roc_cache(args.roc_set, curve_name, auc_score, fpr, tpr)
        regenerate_plot(roc_cache, args.roc_set)