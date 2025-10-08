import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from dataset import RecoveryDataset
from utils import (
    compute_classic_metrics,
    save_results_csv,
    plot_confusion_matrix,
)
from corruptions import (
    STANDARD_CORRUPTIONS,
    TEXT_PERTURBATIONS,
)
from config import WEIGHTS_PATH, DATASET_PATH, DATA_CSV, OUTPUT_DIR
from themis_model import get_Themis
import numpy as np


# load model and weights
def load_model(device, weights_path=WEIGHTS_PATH):
    model, tokenizer, processor = get_Themis(
        name_llm="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name_img_embed="openai/clip-vit-base-patch32",
    )

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        try:
            model.load_state_dict(state)
        except Exception:
            new_state = {}
            for k, v in state.items():
                key = k.replace("module.", "") if k.startswith("module.") else k
                new_state[key] = v
            model.load_state_dict(new_state, strict=False)
    else:
        print("Attenzione: file pesi non trovato, uso modello non addestrato.")

    model.to(device).eval()
    return model, tokenizer, processor


# Main evaluation loop
def main():
    # Prepare output directory and device
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(device)

    # Dataset con immagini e testi già preprocessati
    dataset = RecoveryDataset(
        csv_file=DATA_CSV,
        image_dir=os.path.join(DATASET_PATH, "images"),
        tokenizer=tokenizer,
        processor=processor,
        max_length=64,
    )

    # DataLoader for batching
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # To store results
    rows = []

    # Iterate over all corruption x perturbation combinations
    combinations = [
        (img_name, img_fn, txt_name, txt_fn)
        for img_name, img_fn in STANDARD_CORRUPTIONS.items()
        for txt_name, txt_fn in TEXT_PERTURBATIONS.items()
    ]

    for img_name, img_fn, txt_name, txt_fn in tqdm(
        combinations, desc="Corruptions x Perturbations"
    ):
        y_true = []
        y_pred = []
        samples_changed = []

        pbar = tqdm(loader, desc=f"Eval {img_name} x {txt_name}", leave=False)
        for images, labels, texts in pbar:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            # Applying text perturbation
            decoded_texts = tokenizer.batch_decode(texts, skip_special_tokens=True)
            text_corr = [txt_fn(t) for t in decoded_texts]

            # Tokenize corrupted texts
            text_inputs = tokenizer(
                text_corr,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Applying image corruption
            pil_corr = img_fn(images)
            image_inputs = {"pixel_values": pil_corr}  # already tensor

            # Forward pass
            with torch.no_grad():
                outputs = model(image_inputs, text_inputs)

            # Get predictions
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            # Collect results
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())

            # Identify changed predictions
            for i, (yt, yp) in enumerate(zip(labels.cpu(), preds)):
                if yt != yp:
                    samples_changed.append(
                        {
                            "index": i,
                            "true": int(yt),
                            "pred": int(yp),
                            "image_variant": img_name,
                            "text_variant": txt_name,
                        }
                    )

        # Compute metrics and save results
        if len(y_true) > 0:
            metrics = compute_classic_metrics(y_true, y_pred)
            cm = metrics["confusion_matrix"]
            out_subdir = os.path.join(OUTPUT_DIR, f"{img_name}__{txt_name}")
            os.makedirs(out_subdir, exist_ok=True)

            plot_confusion_matrix(
                cm,
                labels=list(range(cm.shape[0])),
                out_file=os.path.join(out_subdir, "confusion_matrix.png"),
            )

            row = {
                "image_variant": img_name,
                "text_variant": txt_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "num_samples": len(y_true),
                "num_changed": len(samples_changed),
            }
            rows.append(row)

            # Saving the summary in a JSON file
            with open(os.path.join(out_subdir, "summary.json"), "w") as f:
                json.dump(
                    {"metrics": row, "examples_changed": samples_changed[:50]},
                    f,
                    indent=2,
                )

    save_results_csv(rows, os.path.join(OUTPUT_DIR, "robustness_report.csv"))
    print("DONE. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
