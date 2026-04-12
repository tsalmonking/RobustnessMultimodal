import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import json
import glob
import argparse
import torchvision.transforms as T

import datetime

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
    use_model,
    save_results,
    img_corruption,
    txt_corruption,
)
from config import NAME_LLM, NAME_IMG_EMBED, WEIGHTS_PATH, DEBUG
import my_datasets

# Configurations with debug options
if DEBUG:
    BATCH_SIZE = 16
    N_TOKENS = 128

    # PGD
    PGD_ITERS = 25
    EPSILON = 3 / 255
else:
    BATCH_SIZE = 64
    N_TOKENS = 1024

    # PGD
    PGD_ITERS = 25
    EPSILON = 3 / 255

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

    # Select dataset class and load function dynamically
    dataset_class = dataset_classes[args.dataset]
    load_func = load_functions[args.dataset]
    output_dir = os.path.join(args.results_path, f"{args.dataset}")

    dataset_test = my_datasets.get_dataset(
        dataset_class,
        load_func,
        N_TOKENS,
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

    y_true = []
    y_preds, y_txt_corr_preds, y_img_corr_preds, y_multimodal_corr_preds = ([], [], [], [])
    for images, labels, text, _, indices in tqdm(dataloader_test, desc="Evaluating", total=len(dataloader_test)):
        corr_txts_list = []
        corr_imgs_pil_list = []
        indices = indices.tolist()
        # Clean predictions
        with torch.no_grad():
            outputs = model(images, text)
            preds = [1 if i > 0.5 else 0 for i in outputs.cpu().detach().numpy()]
            y_preds.extend(preds)
            y_true.extend(labels.cpu().numpy())
        # Challenging the model
        for i, (pred, label) in tqdm(enumerate(zip(preds, labels.cpu().numpy().tolist())), desc="Challenging the model", total=len(labels), leave=False):
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
                # Print time and similarity for each attempt in a log file
                start = datetime.datetime.now().replace(microsecond=0)
                while txt_similarity < 0.5 and counter < 5:
                    new_txt_corr_news, new_txt_similarity = txt_corruption(news)
                    if new_txt_similarity > txt_similarity:
                        txt_corr_news = new_txt_corr_news
                        txt_similarity = new_txt_similarity
                    counter += 1
                finish = datetime.datetime.now().replace(microsecond=0)
                table = tabulate([[indices[i], start, finish, txt_similarity]], tablefmt="github")
                with open("log.txt", "a") as f:
                    f.write(table + "\n")

                # Create multimodal corrupted new
                multimodal_corr_news = {
                    "txt": txt_corr_news["txt"],
                    "img": img_corr_news["img"],
                }
                # Get predictions on corrupted images only
                img_corr_pred = use_model(model, tokenizer, processor, args, img_corr_news)[0]
                # Get predictions on corrupted text only
                txt_corr_pred = use_model(model, tokenizer, processor, args, txt_corr_news)[0]
                # Get predictions on corrupted images and text
                multimodal_corr_pred = use_model(model, tokenizer, processor, args, multimodal_corr_news)[0]
                # Save results where only multimodality corruption fools the model
                if (img_corr_pred == label and txt_corr_pred == label and multimodal_corr_pred != label):
                    save_results(
                        output_dir,
                        indices[i],
                        news,
                        proccess_img,
                        img_corr_news,
                        multimodal_corr_news,
                        preds={
                            "label_true": label,
                            "clean": pred,
                            "img_corr": img_corr_pred,
                            "txt_corr": txt_corr_pred,
                            "multimodal_corr": multimodal_corr_pred,
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

        # Get predictions on corrupted samples in batch
        with torch.no_grad():
            txt_corr_outputs = model(images, corr_txts)
            txt_corr_preds = [1 if i > 0.5 else 0 for i in txt_corr_outputs.cpu().detach().numpy()]
            y_txt_corr_preds.extend(txt_corr_preds)

            img_corr_outputs = model(corr_imgs, text)
            img_corr_preds = [1 if i > 0.5 else 0 for i in img_corr_outputs.cpu().detach().numpy()]
            y_img_corr_preds.extend(img_corr_preds)

            corr_imgs["pixel_values"] = corr_imgs["pixel_values"].unsqueeze(1)
            multimodal_corr_outputs = model(corr_imgs, corr_txts)
            multimodal_corr_preds = [1 if i > 0.5 else 0 for i in multimodal_corr_outputs.cpu().detach().numpy()]
            y_multimodal_corr_preds.extend(multimodal_corr_preds)

    # ----- RESULTS -----
    # Classic metrics
    metrics_clean, cm_clean = compute_metrics(y_true, y_preds)
    metrics_txt_corr, cm_txt_corr = compute_metrics(y_true, y_txt_corr_preds)
    metrics_img_corr, cm_img_corr = compute_metrics(y_true, y_img_corr_preds)
    metrics_multimodal_corr, cm_multimodal_corr = compute_metrics(y_true, y_multimodal_corr_preds)

    # Robustness metrics
    txt_robustness_metrics = compute_robustness_metrics(y_true, y_preds, y_txt_corr_preds)
    img_robustness_metrics = compute_robustness_metrics(y_true, y_preds, y_img_corr_preds)
    multimodal_robustness_metrics = compute_robustness_metrics(y_true, y_preds, y_multimodal_corr_preds)

    # Save results
    clean_dir = os.path.join(output_dir, "clean")
    corr_dir = os.path.join(output_dir, "corr")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)
    with open(os.path.join(clean_dir, "metrics_clean.json"), "w") as f:
        json.dump(metrics_clean, f, indent=2)
    with open(os.path.join(corr_dir, "metrics_txt_corr.json"), "w") as f:
        json.dump(metrics_txt_corr, f, indent=2)
    with open(os.path.join(corr_dir, "metrics_img_corr.json"), "w") as f:
        json.dump(metrics_img_corr, f, indent=2)
    with open(os.path.join(corr_dir, "metrics_multimodal_corr.json"), "w") as f:
        json.dump(metrics_multimodal_corr, f, indent=2)

    # Print results
    print("Clean Confusion Matrix")
    print(cm_clean)
    print("Text Corr Confusion Matrix")
    print(cm_txt_corr)
    print("Image Corr Confusion Matrix")
    print(cm_img_corr)
    print("Multimodal Corr Confusion Matrix")
    print(cm_multimodal_corr)
    print("\n--- Robustness Summary ---")
    print("- Adversarial Accuracy (ADV accuracy): percentage of inputs correctly classified after the adversarial perturbation.")
    print("- Delta Accuracy (DLT accuracy): difference between clean inputs and corrupted inputs")
    print("- Flip Rate (FR): percentage of inputs where the model's prediction changed after the perturbation.")
    print("- Attack Success Rate (ASR): proportion of originally correct classifications that were flipped to an incorrect label")

    # Print robustness metrics table
    headers = ["TEXT CORRUPTION", "IMAGE CORRUPTION", "MULTIMODAL CORRUPTION"]
    data = [
        [
            "ADV accuracy",
            txt_robustness_metrics["accuracy_on_corrupted"],
            img_robustness_metrics["accuracy_on_corrupted"],
            multimodal_robustness_metrics["accuracy_on_corrupted"],
        ],
        [
            "DLT accuracy",
            txt_robustness_metrics["delta_accuracy"],
            img_robustness_metrics["delta_accuracy"],
            multimodal_robustness_metrics["delta_accuracy"],
        ],
        [
            "FR",
            txt_robustness_metrics["flip_rate"],
            img_robustness_metrics["flip_rate"],
            multimodal_robustness_metrics["flip_rate"],
        ],
        [
            "ASR",
            txt_robustness_metrics["attack_success_rate"],
            img_robustness_metrics["attack_success_rate"],
            multimodal_robustness_metrics["attack_success_rate"],
        ],
    ]
    print(tabulate(data, headers, tablefmt="github"))

    # Save robustness metrics and confusion matrices of each modality
    robustness_dir = os.path.join(output_dir, "robustness")
    os.makedirs(robustness_dir, exist_ok=True)
    with open(os.path.join(robustness_dir, "txt_robustness.json"), "w") as f:
        json.dump(txt_robustness_metrics, f, indent=4)
    with open(os.path.join(robustness_dir, "img_robustness.json"), "w") as f:
        json.dump(img_robustness_metrics, f, indent=4)
    with open(os.path.join(robustness_dir, "multimodal_robustness.json"), "w") as f:
        json.dump(multimodal_robustness_metrics, f, indent=4)
    plot_confusion_matrix(
        cm_clean,
        labels=range(cm_clean.shape[0]),
        out_file=os.path.join(clean_dir, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_txt_corr,
        labels=range(cm_txt_corr.shape[0]),
        out_file=os.path.join(corr_dir, "txt_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_img_corr,
        labels=range(cm_img_corr.shape[0]),
        out_file=os.path.join(corr_dir, "img_corr_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        cm_multimodal_corr,
        labels=range(cm_multimodal_corr.shape[0]),
        out_file=os.path.join(corr_dir, "multimodal_corr_confusion_matrix.png"),
    )


if __name__ == "__main__":
    main()
