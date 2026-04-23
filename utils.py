import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchattacks
import torchvision.transforms as T
import ollama
import json

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure
from sentence_transformers import SentenceTransformer, util


# Custom modules
from themis_model import get_Themis
from prompt import LLM_CORRUPTER_PROMPT

# Utilities for logging
def info(msg):
    print(f"\033[32m{msg}\033[0m")
def warning(msg):
    print(f"\033[33m{msg}\033[0m")
def error(msg):
    print(f"\033[31m{msg}\033[0m")

# The wrapped model for PGD attack, to get only one modality perturbed
class WrappedModel(torch.nn.Module):
    def __init__(self, model, fixed_txt, processor):
        super().__init__()
        self.model = model
        self.fixed_txt = fixed_txt
        self.processor = processor
    def forward(self, x):
        fixed_txt_repeated = {}
        for key, tensor in self.fixed_txt.items():
            fixed_txt_repeated[key] = tensor.repeat(x.size(0), 1)
        mean = torch.tensor(self.processor.image_mean, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.processor.image_std, device=x.device).view(1, -1, 1, 1)
        logit_class_1 = self.model({"pixel_values": ((x - mean) / std).unsqueeze(1)}, fixed_txt_repeated)
        logit_class_0 = torch.zeros_like(logit_class_1)

        return torch.cat((logit_class_0, logit_class_1), dim=1)

def use_model(model, tokenizer, processor, args, news, thr, modality=None):
    # Text tokenization
    token_txt = tokenizer(
        news["txt"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        max_length=args.n_tokens,
    )
    # Image processing
    if isinstance(news["img"], Image.Image):
        process_img = processor(images=news["img"], return_tensors="pt")
        process_img["pixel_values"] = process_img["pixel_values"].unsqueeze(1)
    else:
        process_img = {"pixel_values": news["img"]}
        if process_img["pixel_values"].dim() == 4:
            process_img["pixel_values"] = process_img["pixel_values"].unsqueeze(1)

    # Using the model
    with torch.no_grad():
        if modality == "text":
            output = model(images=None, texts=token_txt)
        elif modality == "image":
            output = model(images=process_img, texts=None)
        else:
            output = model(process_img, token_txt)
        preds = [1 if i > thr else 0 for i in output.cpu().detach().numpy()]

    return preds, output

def save_img(img, save_path):
    to_pil = T.ToPILImage()
    to_pil(img.squeeze(0).cpu().float()).save(save_path)

def img_corruption(model, tokenizer, processor, args, news, label):
    # Text tokenization for fixed text
    token_txt = tokenizer(
        news["txt"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        max_length=args.n_tokens,
    )
    # Image processing for clean image
    process_img = processor(images=news["img"], return_tensors="pt", do_normalize=False)

    # PGD Attack
    wrapped_model = WrappedModel(model, token_txt, processor)
    alpha = args.epsilon / (args.pgd_iters * args.alpha_factor)
    attack = torchattacks.PGD(wrapped_model, eps=args.epsilon, alpha=alpha, steps=args.pgd_iters, random_start=True)
    corr_img = attack(process_img["pixel_values"], label)

    # Construct corrupted news
    corr_news = {"txt": news["txt"], "img": corr_img}

    # Compute SSIM
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_val = ssim(preds=corr_img.float(), target=process_img["pixel_values"].float())

    return corr_news, ssim_val, process_img["pixel_values"]

model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
def txt_corruption(news):
    # LLM-based text corruption
    client = ollama.Client(host="http://127.0.0.1:11435")
    user_content = f"News article:\n{news['txt']}"
    response = client.chat(
        model="qwen2.5:14b-instruct",
        options={"temperature": 0.5, "max_tokens": 2048},
        messages=[
            {"role": "system", "content": LLM_CORRUPTER_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    corr_txt = response["message"]["content"].strip()
    corr_news = {"txt": corr_txt, "img": news["img"]}
    
    # Compute text similarity
    txt_similarity = util.cos_sim(model_sbert.encode(news["txt"], convert_to_tensor=True), model_sbert.encode(corr_txt, convert_to_tensor=True)).item()

    return corr_news, txt_similarity


def save_results(
    output_dir,
    idx,
    img_path,
    news,
    clean_img,
    img_corr_news_pgd,
    multimodal_corr_news,
    preds,
    ssim_pgd,
    txt_similarity,
):
    # Create result directory
    result_dir = os.path.join(output_dir, str(idx))
    os.makedirs(result_dir, exist_ok=True)
    # Save images, clean and corrupted
    save_img(clean_img, os.path.join(result_dir, "clean_img.png"))
    save_img(img_corr_news_pgd["img"], os.path.join(result_dir, "corr_img.png"))

    # Save result data
    result_data = {
        "news": int(idx),
        "img": img_path,
        "label": int(preds["label"]),
        "output_clean": float(preds["output_clean"]),
        "threshold": float(preds["thr_multimodal_cross_clean"]),
        "pred_clean": int(preds["pred_clean"]),
        "output_txt_corr": float(preds["output_txt_corr"]),
        "pred_txt_corr": int(preds["pred_txt_corr"]),
        "output_img_corr": float(preds["output_img_corr"]),
        "pred_img_corr": int(preds["pred_img_corr"]),
        "output_multimodal_corr": float(preds["output_multimodal_corr"]),
        "pred_multimodal_corr": int(preds["pred_multimodal_corr"]),
        "SSIM": round(float(ssim_pgd.item() if hasattr(ssim_pgd, "item") else ssim_pgd), 3),
        "txt_similarity": round(float(txt_similarity.item() if hasattr(txt_similarity, "item") else txt_similarity), 3),
        "original_txt": news["txt"],
        "corr_txt": multimodal_corr_news["txt"],
    }
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=4)


# -----------------------
# Model loading
# -----------------------

def load_model(device, args, modality=None):
    name_llm = args.name_llm
    name_img_embed = args.name_img_embed
    merge_tokens = args.merge_tokens
    if merge_tokens == 0:
        merge_tokens = None
    lora_alpha = args.lora_alpha
    lora_r = args.lora_r
    lora_dropout = args.lora_dropout
    use_lora = args.use_lora
    model_path = args.model_path
    set_params = args.set_params
    if set_params:
        p = model_path.split("\\")[-1].split("_")
        lora_alpha = int(p[2])
        lora_r = int(p[3])
        lora_dropout = float(p[4])
        use_lora = True if "True" in p[5] else False

    model, tokenizer, processor = get_Themis(
        name_llm=name_llm,
        name_img_embed=name_img_embed,
        use_lora=use_lora,
        is_pythia=True if "pythia" in name_llm else False,
        lora_alpha=lora_alpha,
        lora_r=lora_r,
        lora_dropout=lora_dropout,
        merge_tokens=merge_tokens,
    )

    if modality == "text":
        model_path = model_path.replace(".pt", "_txt_only.pt")
    elif modality == "image":
        model_path = model_path.replace(".pt", "_img_only.pt")

    if os.path.exists(model_path):
        try:
            # when a serious GPU will be available change map_location to device
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            error("Error loading weights, it will be used random weights. The results will be meaningless.")
    else:
        warning("Warning: weights file not found, using random weights. The results will be meaningless.")

    model.to(device).eval()

    return model, tokenizer, processor


# -----------------------
# Metrics & reporting
# -----------------------

def compute_metrics(y_true, y_pred, scores):
    acc = accuracy_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    conf_matr = confusion_matrix(y_true, y_pred)
    
    # AUC on on samples
    fpr, tpr, thr = roc_curve(y_true, scores)
    auc_score = auc(fpr, tpr)
    return {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "macro_f1": round(macro_f1, 3),
        "auc": round(auc_score, 3),
    }, conf_matr


def compute_robustness_metrics(y_true, y_clean, y_corr):
    y_true = np.array(y_true)
    y_clean = np.array(y_clean)
    y_corr = np.array(y_corr)

    # Accuracy on corrupted input
    accuracy_on_corrupted = accuracy_score(y_true, y_corr)

    # Attack Success Rate (ASR) - proportion of originally correct classifications that were flipped to an incorrect label by the attack.
    is_correct_clean = y_clean == y_true
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
        "attack_success_rate": round(asr, 3),
    }


def plot_confusion_matrix(cm, labels, out_file):
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_text_vs_image( y_true, logits_text, logits_image, out_file):
    logits_text = np.asarray(logits_text).reshape(-1)
    logits_image = np.asarray(logits_image).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    mask_fake = y_true == 0
    mask_real = y_true == 1

    plt.figure(figsize=(10, 10))

    # Fake = cerchio vuoto rosso
    plt.scatter(
        logits_text[mask_fake],
        logits_image[mask_fake],
        c='red',
        label='Fake (label=0)',
        alpha=0.8
    )

    # Real = cerchio pieno blu
    plt.scatter(
        logits_text[mask_real],
        logits_image[mask_real],
        c='blue',
        label='Real (label=1)',
        alpha=0.8
    )

    plt.axvline(0, linestyle='--')
    plt.axhline(0, linestyle='--')

    plt.xlabel("Logit testo")
    plt.ylabel("Logit immagine")
    plt.title("Scatter logit: testo vs immagine")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_shift_arrows(
    y_true,
    logit_text_clean,
    logit_image_clean,
    logit_text_corr,
    logit_image_corr,
    out_file
):
    x0 = np.asarray(logit_text_clean)
    y0 = np.asarray(logit_image_clean)
    x1 = np.asarray(logit_text_corr)
    y1 = np.asarray(logit_image_corr)
    y_true = np.asarray(y_true)

    dx = x1 - x0
    dy = y1 - y0

    plt.figure(figsize=(10, 10))

    mask_fake = y_true == 0
    mask_real = y_true == 1

    # punti iniziali
    plt.scatter(
        x0[mask_fake], y0[mask_fake],
        c ="red", label="Fake clean", alpha=0.8
    )
    plt.scatter(
        x0[mask_real], y0[mask_real],
        c="blue", label="Real clean", alpha=0.8
    )

    # frecce
    plt.quiver(
        x0[mask_fake], y0[mask_fake],
        dx[mask_fake], dy[mask_fake],
        angles='xy', scale_units='xy', scale=1,
        color='red', alpha=0.5
    )
    plt.quiver(
        x0[mask_real], y0[mask_real],
        dx[mask_real], dy[mask_real],
        angles='xy', scale_units='xy', scale=1,
        color='blue', alpha=0.5
    )

    plt.axvline(0, linestyle='--')
    plt.axhline(0, linestyle='--')

    plt.xlabel("Logit testo")
    plt.ylabel("Logit immagine")
    plt.title("Spostamento dei sample dopo corruzione")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def compute_threshold(logits, labels):
    fpr, tpr, thr = roc_curve(labels, logits)
    auc_score = auc(fpr, tpr)
    best_thr = thr[(tpr - fpr).argmax()]
    return best_thr, round(auc_score, 3)

def preds_fusion(first_modality_outputs, second_modality_outputs, labels, mode="mean"):
    if mode == "mean":
        fused_scores = (first_modality_outputs + second_modality_outputs) / 2
    elif mode == "max":
        fused_scores = torch.max(first_modality_outputs, second_modality_outputs)
    elif mode == "min":
        fused_scores = torch.min(first_modality_outputs, second_modality_outputs)
    else:
        raise ValueError(f"Invalid fusion mode: {mode}")

    th, auc_score = compute_threshold(fused_scores.detach().cpu().numpy(), labels.detach().cpu().numpy())
    fused_preds = [1 if s > th else 0 for s in fused_scores]

    return fused_preds, fused_scores.detach().cpu().numpy().reshape(-1)
