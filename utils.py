import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import torch
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Image corruptions (lecite)
# -----------------------
def add_gaussian_noise(pil_img, std=0.05):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def random_occlusion(pil_img, occ_size_ratio=0.2, color=(127,127,127)):
    img = pil_img.copy()
    w,h = img.size
    occ_w = int(w * occ_size_ratio)
    occ_h = int(h * occ_size_ratio)
    x0 = random.randint(0, max(0,w-occ_w))
    y0 = random.randint(0, max(0,h-occ_h))
    draw = Image.new('RGB', (occ_w, occ_h), color)
    img.paste(draw, (x0,y0))
    return img

def change_brightness(pil_img, factor=0.7):
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)

def jpeg_compress(pil_img, quality=30):
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def gaussian_blur(pil_img, radius=2):
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

def resize_scale(pil_img, scale=0.9):
    w,h = pil_img.size
    new_w, new_h = int(w*scale), int(h*scale)
    return pil_img.resize((new_w,new_h)).resize((w,h))

# mapping delle corruzioni per iterare
STANDARD_CORRUPTIONS = {
    "clean": lambda im: im,
    "gaussian_noise": lambda im: add_gaussian_noise(im, std=0.08),
    "random_occlusion": lambda im: random_occlusion(im, occ_size_ratio=0.25),
    "brightness_low": lambda im: change_brightness(im, 0.5),
    "jpeg30": lambda im: jpeg_compress(im, quality=30),
    "gaussian_blur": lambda im: gaussian_blur(im, radius=2),
    "resize_scale": lambda im: resize_scale(im, scale=0.85)
}

# -----------------------
# Text perturbations (lecite)
# -----------------------
def inject_typos(text, p=0.1):
    # semplice: scambia caratteri casuali con probabilità p per singola parola
    chars = list(text)
    for i in range(len(chars)-1):
        if random.random() < p:
            # swap i and i+1
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)

def negate_statement(text):
    # Semplice heuristica: aggiunge "not" o "non" — attenzione: modello-language-specific
    if text.strip()=="":
        return text
    return "not " + text

def append_confusing_phrase(text):
    return text + " . This statement may be false or misleading."

def synonym_swap(text, n_swaps=1):
    # placeholder very simple: ruota parola con successiva - utente può sostituire con NLP più sofisticato
    toks = text.split()
    if len(toks) < 2:
        return text
    for _ in range(min(n_swaps, len(toks)-1)):
        i = random.randint(0, len(toks)-2)
        toks[i], toks[i+1] = toks[i+1], toks[i]
    return " ".join(toks)

TEXT_PERTURBATIONS = {
    "clean": lambda s: s,
    "typos": lambda s: inject_typos(s, p=0.12),
    "negation": lambda s: negate_statement(s),
    "append_confuse": lambda s: append_confusing_phrase(s),
    "syn_swap": lambda s: synonym_swap(s, n_swaps=2)
}

# -----------------------
# Metrics & reporting
# -----------------------
def compute_classic_metrics(y_true, y_pred, pos_label=1):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary' if len(set(y_true))==2 else 'macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }

def save_results_csv(rows, out_path):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

def plot_confusion_matrix(cm, labels, out_file):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()