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

def use_model(model, tokenizer, processor, args, news):
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
        outputs = model(process_img, token_txt)
        preds = [1 if i > 0.5 else 0 for i in outputs.cpu().detach().numpy()]

    return preds

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
    user_content = f"News article:\n{news['txt']}"
    response = ollama.chat(
        model="phi3:instruct",
        options={"temperature": 0.3},
        messages=[
            {"role": "system", "content": LLM_CORRUPTER_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    corr_txt = response["message"]["content"].strip()
    corr_news = {"txt": corr_txt, "img": news["img"]}
    
    # Compute text similarity
    txt_similarity = util.cos_sim(model_sbert.encode(news["txt"], convert_to_tensor=True), model_sbert.encode(corr_txt, convert_to_tensor=True)).item()

    return corr_news, txt_similarity


def save_results(
    output_dir,
    idx,
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
        "index": int(idx),
        "true_label": int(preds["label_true"]),
        "pred_clean": int(preds["clean"]),
        "pred_img_corr": int(preds["img_corr"]),
        "pred_txt_corr": int(preds["txt_corr"]),
        "pred_multimodal_corr": int(preds["multimodal_corr"]),
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

def load_model(device, args):
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

    if os.path.exists(model_path):
        try:
            # when a serious GPU will be available change map_location to device
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except Exception:
            error("Error loading weights, it will be used random weights. The results will be meaningless.")
    else:
        warning("Warning: weights file not found, using random weights. The results will be meaningless.")

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
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    conf_matr = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "macro_f1": round(macro_f1, 3),
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
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
