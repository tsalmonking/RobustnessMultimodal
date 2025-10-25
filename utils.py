import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Custom modules
from themis_model import get_Themis


# Utilities for logging
def info(msg):
    print(f"\033[32m{msg}\033[0m")


def warning(msg):
    print(f"\033[33m{msg}\033[0m")


def error(msg):
    print(f"\033[31m{msg}\033[0m")


# -----------------------
# Data handling
# -----------------------
def multimodal_collate(batch):
    """
    batch: list of tuples (text: str, image: PIL.Image, label: int)
    Returns:
      texts: list[str]
      images: list[PIL.Image]
      labels: list[int]
    """
    id = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    images = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    return id, texts, images, labels


# -----------------------
# Model loading
# -----------------------
def load_model(
    device,
    weights_path="model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt",
    name_llm="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    name_img_embed="openai/clip-vit-base-patch32",
):
    p = weights_path.split("\\")[-1].split("_")
    lora_alpha = int(p[2])
    lora_r = int(p[3])
    lora_dropout = float(p[4])
    use_lora = True if "True" in p[5] else False
    
    model, tokenizer, processor = get_Themis(
        name_llm=name_llm,
        name_img_embed=name_img_embed,
        use_lora=use_lora,
        lora_alpha=lora_alpha,
        lora_r=lora_r,
        lora_dropout=lora_dropout,
    )

    if os.path.exists(weights_path):
        try:
            # when a serious GPU will be available change map_location to device
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception:
            error(
                "Error loading weights, it will be used random weights. The results will be meaningless."
            )
    else:
        warning(
            "Warning: weights file not found, using random weights. The results will be meaningless."
        )

    model.to(device).eval()
    return model, tokenizer, processor


# -----------------------
# Metrics & reporting
# -----------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    conf_matr = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "macro_f1": round(macro_f1, 3)
    }, conf_matr

def compute_robustness_metrics(y_true, y_clean, y_corr):
    y_true = np.array(y_true)
    y_clean = np.array(y_clean)
    y_corr = np.array(y_corr)
    
    # Accuracy on corrupted input
    accuracy_on_corrupted = accuracy_score(y_true, y_corr)
    
    # Attack Success Rate (ASR) - proportion of originally correct classifications that were flipped to an incorrect label by the attack.
    is_correct_clean = (y_clean == y_true)
    total_correct_clean = np.sum(is_correct_clean)
    if total_correct_clean == 0:
        asr = 0.0
    else:
        is_successful_attack = is_correct_clean & (y_corr != y_true)
        successful_attacks = np.sum(is_successful_attack)
        asr = successful_attacks / total_correct_clean

    # Delta Accuracy - the difference between accuracy on clean and corrupted input
    accuracy_on_clean = accuracy_score(y_true, y_clean)
    delta_acc = accuracy_on_clean - accuracy_on_corrupted
    
    # Flip Rate - percentage of inputs were correctly classified even after perturbation
    flip_rate = np.sum(y_clean != y_corr) / len(y_clean)
    
    return {
        "accuracy_on_clean": round(accuracy_on_clean, 3),
        "accuracy_on_corrupted": round(accuracy_on_corrupted, 3),
        "delta_accuracy": round(delta_acc, 3),
        "flip_rate": round(flip_rate, 3),
        "attack_success_rate": round(asr, 3) 
    }


def save_results_csv(rows, out_path):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def plot_confusion_matrix(cm, labels, out_file):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
