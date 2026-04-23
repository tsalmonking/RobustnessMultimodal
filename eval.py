import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import json
import glob
import argparse
import torchvision.transforms as T
import numpy as np

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

# Custom imports
from utils import (
    load_model,
    compute_metrics,
    compute_robustness_metrics,
    plot_confusion_matrix,
    plot_text_vs_image,
    plot_shift_arrows,
    compute_threshold,
    preds_fusion,
    use_model,
    save_results,
    img_corruption,
    txt_corruption,
)
from config import NAME_LLM, NAME_IMG_EMBED, WEIGHTS_PATH
import datasets

# Default configuration parameters
BATCH_SIZE = 2
N_TOKENS = 1024
PGD_ITERS = 25
EPSILON = 3 / 255
MAX_TXT_CORRUPTION_ATTEMPTS = 5

# Get available datasets from Data directory
available_datasets = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
# Create mappings dynamically
dataset_classes = {}
load_functions = {}
for dataset in available_datasets:
    class_name = f"{dataset}_Dataset"
    load_name = f"{dataset.lower()}_load_annotations_file"
    try:
        dataset_classes[dataset] = getattr(my_datasets, class_name)
        load_functions[dataset] = getattr(my_datasets, load_name)
    except AttributeError:
        print(f"Warning: Class {class_name} or function {load_name} not found in my_datasets.py for dataset {dataset}")

# Main evaluation function
def main():
    # Here there are the evaluation parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_llm", type=str, default=NAME_LLM)
    parser.add_argument("--name_img_embed", type=str, default=NAME_IMG_EMBED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model_path", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--n_tokens", type=int, default=N_TOKENS)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--set_params", type=bool, default=True)
    parser.add_argument("--pgd_iters", type=int, default=PGD_ITERS)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--alpha_factor", type=float, default=2.0)
    parser.add_argument("--results_path", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="Fakeddit", choices=list(dataset_classes.keys()))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, processor = load_model(device, args)
    txt_model, tokenizer_txt, processor_txt = load_model(device, args, modality="text")
    img_model, tokenizer_img, processor_img = load_model(device, args, modality="image")

    # Select dataset class and load function dynamically
    dataset_class = dataset_classes[args.dataset]
    load_func = load_functions[args.dataset]
    output_dir = os.path.join(args.results_path, f"{args.dataset}")

    dataset_test = my_datasets.get_dataset(
        dataset_class,
        load_func,
        args.n_tokens,
        processor,
        tokenizer,
        glob.glob(f"data/{args.dataset}/test.*")[0],
        f"data/{args.dataset}/images",
    )
    
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        #generator=torch.Generator(device=device),
    )

    # True labels
    y_true = []

    # Predicted labels on clean samples, multimodal cross, unimodal (img, txt) and multimodal fusion (mean, min, max)
    y_preds = []
    y_unimodal_txt_preds = []
    y_unimodal_img_preds = []
    y_multimodal_fusion_mean_preds = []
    y_multimodal_fusion_min_preds = []
    y_multimodal_fusion_max_preds = []

    # Predicted labels on corrupted samples, multimdoal cross, txt only and img only
    y_multimodal_cross_corr_preds = []
    y_multimodal_cross_txt_corr_preds = []
    y_multimodal_cross_img_corr_preds = []

    # Predicted labels on corrupted samples, unimodal (img, txt) and fusion
    y_unimodal_txt_corr_preds = []
    y_unimodal_img_corr_preds = []
    y_multimodal_fusion_mean_corr_preds = []
    y_multimodal_fusion_min_corr_preds = []
    y_multimodal_fusion_max_corr_preds = []

    # Logits of multimodal cross
    multimodal_cross_outputs = []
    multimodal_cross_corr_outputs = []
    multimodal_cross_txt_corr_outputs = []
    multimodal_cross_img_corr_outputs = []

    # Logits of txt only and img only
    unimodal_txt_outputs = []
    unimodal_img_outputs = []
    unimodal_txt_corr_outputs = []
    unimodal_img_corr_outputs = []

    # Logits of multimodal fusion
    multimodal_fusion_mean_outputs = []
    multimodal_fusion_min_outputs = []
    multimodal_fusion_max_outputs = []
    multimodal_fusion_mean_corr_outputs = []
    multimodal_fusion_min_corr_outputs = []
    multimodal_fusion_max_corr_outputs = []
    count = 0
    for images, labels, text, _, indices in tqdm(dataloader_test, desc="Evaluating", total=len(dataloader_test)):
        if count == 1:
            break
        count += 1
        corr_txts_list = []
        corr_imgs_pil_list = []
        indices = indices.tolist()

        images_copy = {k: v.clone() if torch.is_tensor(v) else v for k, v in images.items()}
        # Clean predictions
        with torch.no_grad():
            outputs = model(images, text)
            thr_multimodal_cross_clean, auc_multimodal_cross_clean = compute_threshold(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            preds = [1 if i > thr_multimodal_cross_clean else 0 for i in outputs.cpu().detach().numpy()]
            y_preds.extend(preds)
            multimodal_cross_outputs.extend(outputs.cpu().detach().numpy())
            y_true.extend(labels.cpu().numpy())

            txt_outputs = txt_model(images=None, texts=text)
            thr_unimodal_txt_clean, auc_unimodal_txt_clean = compute_threshold(txt_outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            txt_preds = [1 if i > thr_unimodal_txt_clean else 0 for i in txt_outputs.cpu().detach().numpy()]
            y_unimodal_txt_preds.extend(txt_preds)
            unimodal_txt_outputs.extend(txt_outputs.cpu().detach().numpy())

            img_outputs = img_model(images=images_copy, texts=None)
            thr_unimodal_img_clean, auc_unimodal_img_clean = compute_threshold(img_outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            img_preds = [1 if i > thr_unimodal_img_clean else 0 for i in img_outputs.cpu().detach().numpy()]
            y_unimodal_img_preds.extend(img_preds)
            unimodal_img_outputs.extend(img_outputs.cpu().detach().numpy())

            y_multimodal_fusion_mean_preds.extend(preds_fusion(txt_outputs, img_outputs, labels, mode="mean")[0])
            y_multimodal_fusion_min_preds.extend(preds_fusion(txt_outputs, img_outputs, labels, mode="min")[0])
            y_multimodal_fusion_max_preds.extend(preds_fusion(txt_outputs, img_outputs, labels, mode="max")[0])

            multimodal_fusion_mean_outputs.extend(preds_fusion(txt_outputs, img_outputs, labels, mode="mean")[1])
            multimodal_fusion_min_outputs.extend(preds_fusion(txt_outputs, img_outputs, labels, mode="min")[1])
            multimodal_fusion_max_outputs.extend(preds_fusion(txt_outputs, img_outputs, labels, mode="max")[1])

        # Challenging the model
        for i, (pred, output, label) in tqdm(enumerate(zip(preds, outputs.cpu().detach().numpy(), labels.cpu().numpy().tolist())), desc="Challenging the model", total=len(labels), leave=False):
            # Clean news
            news = {
                "txt": dataset_test.texts[indices[i]],
                "img": Image.open(os.path.join(dataset_test.img_dir, dataset_test.imgs_path[indices[i]])).convert("RGB"),
            }
            # Only consider correctly classified samples
            if pred == label:
                img_corr_news, ssim_pgd, proccess_img = img_corruption(model, tokenizer, processor, args, news, torch.tensor([label], device=device))
                # Ensure text corruption is significant
                counter = 0
                txt_similarity = 0.0
                # Ensure that the text corruption is significant enough to potentially fool the model, but also try to keep it as high as possible to maintain semantic similarity
                while txt_similarity < 0.5 or txt_similarity > 0.93 and counter < MAX_TXT_CORRUPTION_ATTEMPTS:
                    new_txt_corr_news, new_txt_similarity = txt_corruption(news)
                    if new_txt_similarity > txt_similarity:
                        txt_corr_news = new_txt_corr_news
                        txt_similarity = new_txt_similarity
                    counter += 1

                # Create multimodal corrupted news
                multimodal_corr_news = {
                    "txt": txt_corr_news["txt"],
                    "img": img_corr_news["img"],
                }
                # Get predictions on corrupted images only
                img_corr_pred, img_corr_output = use_model(model, tokenizer, processor, args, img_corr_news, thr_multimodal_cross_clean)
                # Get predictions on corrupted text only
                txt_corr_pred, txt_corr_output = use_model(model, tokenizer, processor, args, txt_corr_news, thr_multimodal_cross_clean)
                # Get predictions on corrupted images and text
                multimodal_corr_pred, multimodal_corr_output  = use_model(model, tokenizer, processor, args, multimodal_corr_news, thr_multimodal_cross_clean)
                # Save results where only multimodality corruption fools the model
                if (int(img_corr_pred[0]) == label and int(txt_corr_pred[0]) == label and int(multimodal_corr_pred[0]) != label):
                    save_results(
                        output_dir,
                        indices[i],
                        dataset_test.imgs_path[indices[i]],
                        news,
                        proccess_img,
                        img_corr_news,
                        multimodal_corr_news,
                        preds={
                            "label": label,
                            "output_clean": output,
                            "thr_multimodal_cross_clean": thr_multimodal_cross_clean,
                            "pred_clean": pred,
                            "output_txt_corr": txt_corr_output[0].cpu().detach().numpy()[0],
                            "pred_txt_corr": int(txt_corr_pred[0]),
                            "output_img_corr": img_corr_output[0].cpu().detach().numpy()[0],
                            "pred_img_corr": int(img_corr_pred[0]),
                            "output_multimodal_corr": multimodal_corr_output[0].cpu().detach().numpy()[0],
                            "pred_multimodal_corr": int(multimodal_corr_pred[0]),
                        },
                        ssim_pgd=ssim_pgd,
                        txt_similarity=txt_similarity,
                    )
                # Prepare corrupted samples for batch evaluation
                to_pil = T.ToPILImage()
                img_corr = to_pil(img_corr_news["img"].squeeze(0).cpu())
            else:
                img_corr = news["img"]
                txt_corr_news = news
                txt_similarity = 1.0
                ssim_pgd = 1.0
            
            corr_txts_list.append(txt_corr_news["txt"])
            corr_imgs_pil_list.append(img_corr)

        # Tokenization of corrupted samples
        corr_txts = tokenizer(corr_txts_list, return_tensors="pt", padding="max_length", truncation=True, return_attention_mask=False, max_length=args.n_tokens)
        corr_txts = {"input_ids": corr_txts.input_ids.unsqueeze(1)}

        # Processing of corrupted images
        corr_imgs = processor(images=corr_imgs_pil_list, return_tensors="pt", do_normalize=False)
        mean = torch.tensor(processor.image_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(processor.image_std, device=device).view(1, -1, 1, 1)
        corr_imgs = {"pixel_values": ((corr_imgs["pixel_values"] - mean) / std)}
        corr_imgs["pixel_values"] = corr_imgs["pixel_values"].unsqueeze(1)
        images["pixel_values"] = images["pixel_values"].unsqueeze(1)

        corr_imgs_copy = {k: v.to(device) for k, v in corr_imgs.items()}

        # Get predictions on corrupted samples in batch
        with torch.no_grad():
            batch_multimodal_cross_txt_corr_outputs = model(images, corr_txts)
            thr_multimodal_cross_txt_corr, auc_multimodal_cross_txt_corr = compute_threshold(batch_multimodal_cross_txt_corr_outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            txt_corr_preds = [1 if i > thr_multimodal_cross_txt_corr else 0 for i in batch_multimodal_cross_txt_corr_outputs.cpu().detach().numpy()]
            y_multimodal_cross_txt_corr_preds.extend(txt_corr_preds)
            multimodal_cross_txt_corr_outputs.extend(batch_multimodal_cross_txt_corr_outputs.cpu().detach().numpy())

            batch_multimodal_cross_img_corr_outputs = model(corr_imgs, text)
            thr_multimodal_cross_img_corr, auc_multimodal_cross_img_corr = compute_threshold(batch_multimodal_cross_img_corr_outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            img_corr_preds = [1 if i > thr_multimodal_cross_img_corr else 0 for i in batch_multimodal_cross_img_corr_outputs.cpu().detach().numpy()]
            y_multimodal_cross_img_corr_preds.extend(img_corr_preds)
            multimodal_cross_img_corr_outputs.extend(batch_multimodal_cross_img_corr_outputs.cpu().detach().numpy())

            corr_imgs["pixel_values"] = corr_imgs["pixel_values"].unsqueeze(1)
            batch_multimodal_cross_corr_outputs = model(corr_imgs, corr_txts)
            thr_multimodal_cross_corr, auc_multimodal_cross_corr = compute_threshold(batch_multimodal_cross_corr_outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            multimodal_cross_corr_preds = [1 if i > thr_multimodal_cross_corr else 0 for i in batch_multimodal_cross_corr_outputs.cpu().detach().numpy()]
            y_multimodal_cross_corr_preds.extend(multimodal_cross_corr_preds)
            multimodal_cross_corr_outputs.extend(batch_multimodal_cross_corr_outputs.cpu().detach().numpy())

            txt_corr_outputs = txt_model(images=None, texts=corr_txts)
            thr_unimodal_txt_corr, auc_unimodal_txt_corr = compute_threshold(txt_corr_outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            unimodal_txt_corr_preds = [1 if i > thr_unimodal_txt_corr else 0 for i in txt_corr_outputs.cpu().detach().numpy()]
            y_unimodal_txt_corr_preds.extend(unimodal_txt_corr_preds)
            unimodal_txt_corr_outputs.extend(txt_corr_outputs.cpu().detach().numpy())

            img_corr_outputs = img_model(images=corr_imgs_copy, texts=None)
            thr_unimodal_img_corr, auc_unimodal_img_corr = compute_threshold(img_corr_outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            unimodal_img_corr_preds = [1 if i > thr_unimodal_img_corr else 0 for i in img_corr_outputs.cpu().detach().numpy()]
            y_unimodal_img_corr_preds.extend(unimodal_img_corr_preds)
            unimodal_img_corr_outputs.extend(img_corr_outputs.cpu().detach().numpy())

            y_multimodal_fusion_mean_corr_preds.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, labels, mode="mean")[0])
            y_multimodal_fusion_min_corr_preds.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, labels, mode="min")[0])
            y_multimodal_fusion_max_corr_preds.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, labels, mode="max")[0])

            multimodal_fusion_mean_corr_outputs.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, labels, mode="mean")[1])
            multimodal_fusion_min_corr_outputs.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, labels, mode="min")[1])
            multimodal_fusion_max_corr_outputs.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, labels, mode="max")[1])

    # ----- RESULTS -----
    # Classic metrics on clean samples
    metrics_multimodal_cross_clean, cm_multimodal_cross_clean = compute_metrics(y_true, y_preds, multimodal_cross_outputs)
    metrics_unimodal_txt_clean, cm_unimodal_txt_clean = compute_metrics(y_true, y_unimodal_txt_preds, unimodal_txt_outputs)
    metrics_unimodal_img_clean, cm_unimodal_img_clean = compute_metrics(y_true, y_unimodal_img_preds, unimodal_img_outputs)
    metrics_multimodal_fusion_mean, cm_multimodal_fusion_mean = compute_metrics(y_true, y_multimodal_fusion_mean_preds, multimodal_fusion_mean_outputs)
    metrics_multimodal_fusion_min, cm_multimodal_fusion_min = compute_metrics(y_true, y_multimodal_fusion_min_preds, multimodal_fusion_min_outputs)
    metrics_multimodal_fusion_max, cm_multimodal_fusion_max = compute_metrics(y_true, y_multimodal_fusion_max_preds, multimodal_fusion_max_outputs)

    # Classic metrics on corrupted samples
    metrics_multimodal_cross_corr, cm_multimodal_cross_corr = compute_metrics(y_true, y_multimodal_cross_corr_preds, multimodal_cross_corr_outputs)
    metrics_unimodal_txt_corr, cm_unimodal_txt_corr = compute_metrics(y_true, y_unimodal_txt_corr_preds, unimodal_txt_corr_outputs)
    metrics_unimodal_img_corr, cm_unimodal_img_corr = compute_metrics(y_true, y_unimodal_img_corr_preds, unimodal_img_corr_outputs)
    metrics_multimodal_fusion_mean_corr, cm_multimodal_fusion_mean_corr = compute_metrics(y_true, y_multimodal_fusion_mean_corr_preds, multimodal_fusion_mean_corr_outputs)
    metrics_multimodal_fusion_min_corr, cm_multimodal_fusion_min_corr = compute_metrics(y_true, y_multimodal_fusion_min_corr_preds, multimodal_fusion_min_corr_outputs)
    metrics_multimodal_fusion_max_corr, cm_multimodal_fusion_max_corr = compute_metrics(y_true, y_multimodal_fusion_max_corr_preds, multimodal_fusion_max_corr_outputs)
    metrics_multimodal_txt_corr, cm_multimodal_txt_corr = compute_metrics(y_true, y_multimodal_cross_txt_corr_preds, multimodal_cross_txt_corr_outputs)
    metrics_multimodal_img_corr, cm_multimodal_img_corr = compute_metrics(y_true, y_multimodal_cross_img_corr_preds, multimodal_cross_img_corr_outputs)

    # Robustness metrics
    robustness_metrics_multimodal_cross = compute_robustness_metrics(y_true, y_preds, y_multimodal_cross_corr_preds)
    robustness_metrics_unimodal_txt = compute_robustness_metrics(y_true, y_unimodal_txt_preds, y_unimodal_txt_corr_preds)
    robustness_metrics_unimodal_img = compute_robustness_metrics(y_true, y_unimodal_img_preds, y_unimodal_img_corr_preds)
    robustness_metrics_multimodal_fusion_mean = compute_robustness_metrics(y_true, y_multimodal_fusion_mean_preds, y_multimodal_fusion_mean_corr_preds)
    robustness_metrics_multimodal_fusion_min = compute_robustness_metrics(y_true, y_multimodal_fusion_min_preds, y_multimodal_fusion_min_corr_preds)
    robustness_metrics_multimodal_fusion_max = compute_robustness_metrics(y_true, y_multimodal_fusion_max_preds, y_multimodal_fusion_max_corr_preds)
    robustness_metrics_multimodal_txt = compute_robustness_metrics(y_true, y_preds, y_multimodal_cross_txt_corr_preds)
    robustness_metrics_multimodal_img = compute_robustness_metrics(y_true, y_preds, y_multimodal_cross_img_corr_preds)

    # Save results
    clean_dir = os.path.join(output_dir, "clean")
    corr_dir = os.path.join(output_dir, "corr")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)

    metrics_clean = {
        "multimodal_cross_clean": {
            "metrics": metrics_multimodal_cross_clean,
        },
        "unimodal_txt_clean": {
            "metrics": metrics_unimodal_txt_clean,
        },
        "unimodal_img_clean": {
            "metrics": metrics_unimodal_img_clean,
        },
        "multimodal_fusion_mean_clean": {
            "metrics": metrics_multimodal_fusion_mean,
        },
        "multimodal_fusion_min_clean": {
            "metrics": metrics_multimodal_fusion_min,
        },
        "multimodal_fusion_max_clean": {
            "metrics": metrics_multimodal_fusion_max,
        },
    }

    with open(os.path.join(clean_dir, "metrics_clean.json"), "w") as f:
        json.dump(metrics_clean, f, indent=2)
    
    metrics_corr = {
        "multimodal_cross_corr": {
            "metrics": metrics_multimodal_cross_corr,
        },
        "unimodal_txt_corr": {
            "metrics": metrics_unimodal_txt_corr,
        },
        "unimodal_img_corr": {
            "metrics": metrics_unimodal_img_corr,
        },
        "multimodal_fusion_mean_corr": {
            "metrics": metrics_multimodal_fusion_mean_corr,
        },
        "multimodal_fusion_min_corr": {
            "metrics": metrics_multimodal_fusion_min_corr,
        },
        "multimodal_fusion_max_corr": {
            "metrics": metrics_multimodal_fusion_max_corr,
        },
        "multimodal_txt_corr": {
            "metrics": metrics_multimodal_txt_corr,
        },
        "multimodal_img_corr": {
            "metrics": metrics_multimodal_img_corr,
        },
    }
    with open(os.path.join(corr_dir, "metrics_corr.json"), "w") as f:
        json.dump(metrics_corr, f, indent=2)

    # Print results
    print("Multimodal Cross Clean Confusion Matrix")
    print(cm_multimodal_cross_clean)
    print("Unimodal Text Clean Confusion Matrix")
    print(cm_unimodal_txt_clean)
    print("Unimodal Image Clean Confusion Matrix")
    print(cm_unimodal_img_clean)
    print("Multimodal Fusion Mean Clean Confusion Matrix")
    print(cm_multimodal_fusion_mean)
    print("Multimodal Fusion Min Clean Confusion Matrix")
    print(cm_multimodal_fusion_min)
    print("Multimodal Fusion Max Clean Confusion Matrix")
    print(cm_multimodal_fusion_max)
    print("Multimodal Fusion Mean Corrupted Confusion Matrix")
    print(cm_multimodal_fusion_mean_corr)
    print("Multimodal Fusion Min Corrupted Confusion Matrix")
    print(cm_multimodal_fusion_min_corr)
    print("Multimodal Fusion Max Corrupted Confusion Matrix")
    print(cm_multimodal_fusion_max_corr)
    print("Multimodal Cross Corrupted Confusion Matrix")
    print(cm_multimodal_cross_corr)
    print("Multimodal Cross Text Corrupted Confusion Matrix")
    print(cm_multimodal_txt_corr)
    print("Multimodal Cross Image Corrupted Confusion Matrix")
    print(cm_multimodal_img_corr)

    # Print robustness metrics table
    headers = ["AUC CLEAN", "AUC TXT CORR", "AUC IMG CORR", "AUC BOTH CORR"]
    data = [
        [
            "Multimodal Cross",
            metrics_multimodal_cross_clean["auc"],
            metrics_multimodal_txt_corr["auc"],
            metrics_multimodal_img_corr["auc"],
            metrics_multimodal_cross_corr["auc"],
        ],
        [
            "Multimodal Fusion Mean",
            metrics_multimodal_fusion_mean["auc"],
            metrics_unimodal_txt_corr["auc"],
            metrics_unimodal_img_corr["auc"],
            metrics_multimodal_fusion_mean_corr["auc"],
        ],
        [
            "Multimodal Fusion Min",
            metrics_multimodal_fusion_min["auc"],
            metrics_unimodal_txt_corr["auc"],
            metrics_unimodal_img_corr["auc"],
            metrics_multimodal_fusion_min_corr["auc"],
        ],
        [
            "Multimodal Fusion Max",
            metrics_multimodal_fusion_max["auc"],
            metrics_unimodal_txt_corr["auc"],
            metrics_unimodal_img_corr["auc"],
            metrics_multimodal_fusion_max_corr["auc"],
        ],
    ]
    print("\n" + tabulate(data, headers, tablefmt="github") + "\n")

    plot_confusion_matrix(
        cm_multimodal_cross_clean,
        labels=range(cm_multimodal_cross_clean.shape[0]),
        out_file=os.path.join(clean_dir, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_unimodal_txt_clean,
        labels=range(cm_unimodal_txt_clean.shape[0]),
        out_file=os.path.join(corr_dir, "txt_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_unimodal_img_clean,
        labels=range(cm_unimodal_img_clean.shape[0]),
        out_file=os.path.join(corr_dir, "img_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_multimodal_fusion_mean,
        labels=range(cm_multimodal_fusion_mean.shape[0]),
        out_file=os.path.join(corr_dir, "multimodal_fusion_mean_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_multimodal_fusion_min,
        labels=range(cm_multimodal_fusion_min.shape[0]),
        out_file=os.path.join(corr_dir, "multimodal_fusion_min_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_multimodal_fusion_max,
        labels=range(cm_multimodal_fusion_max.shape[0]),
        out_file=os.path.join(corr_dir, "multimodal_fusion_max_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_multimodal_txt_corr,
        labels=range(cm_multimodal_txt_corr.shape[0]),
        out_file=os.path.join(corr_dir, "multimodal_txt_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_multimodal_img_corr,
        labels=range(cm_multimodal_img_corr.shape[0]),
        out_file=os.path.join(corr_dir, "multimodal_img_corr_confusion_matrix.png"),
    )

    # Save robustness metrics and confusion matrices of each modality
    robustness_dir = os.path.join(output_dir, "robustness")
    os.makedirs(robustness_dir, exist_ok=True)
    robustness_results = {
        "multimodal_cross": {
            "robustness_metrics": robustness_metrics_multimodal_cross,
        },
        "unimodal_txt": {
            "robustness_metrics": robustness_metrics_unimodal_txt,
        },
        "unimodal_img": {
            "robustness_metrics": robustness_metrics_unimodal_img,
        },
        "multimodal_fusion_mean": {
            "robustness_metrics": robustness_metrics_multimodal_fusion_mean,
        },
        "multimodal_fusion_min": {
            "robustness_metrics": robustness_metrics_multimodal_fusion_min,
        },
        "multimodal_fusion_max": {
            "robustness_metrics": robustness_metrics_multimodal_fusion_max,
        },
         "multimodal_txt": {
            "robustness_metrics": robustness_metrics_multimodal_txt,
        },
        "multimodal_img": {
            "robustness_metrics": robustness_metrics_multimodal_img,
        },
    }
    with open(os.path.join(robustness_dir, "robustness_results.json"), "w") as f:
        json.dump(robustness_results, f, indent=4)

    plot_text_vs_image(
        y_true,
        unimodal_txt_outputs,
        unimodal_img_outputs,
        os.path.join(f"results/{dataset}/text_vs_image_clean.png"),
    )

    plot_text_vs_image(
        y_true,
        unimodal_txt_corr_outputs,
        unimodal_img_corr_outputs,
        os.path.join(f"results/{dataset}/text_vs_image_corrupted.png"),
    )

    plot_shift_arrows(
        y_true,
        unimodal_txt_outputs,
        unimodal_img_outputs,
        unimodal_txt_corr_outputs,
        unimodal_img_corr_outputs,
        os.path.join(f"results/{dataset}/shift_arrows.png"),
    )

    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        f.write(f"Model: {args.name_llm}\n")
        f.write(f"Image Embedder: {args.name_img_embed}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Number of Tokens: {args.n_tokens}\n")
        f.write(f"Merge Tokens: {args.merge_tokens}\n")
        f.write(f"LoRA Alpha: {args.lora_alpha}\n")
        f.write(f"LoRA R: {args.lora_r}\n")
        f.write(f"LoRA Dropout: {args.lora_dropout}\n")
        f.write(f"Use LoRA: {args.use_lora}\n")
        f.write(f"Set Params from Filename: {args.set_params}\n")
        f.write(f"PGD Iterations: {args.pgd_iters}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write(f"Alpha Factor: {args.alpha_factor}\n")


if __name__ == "__main__":
    main()
