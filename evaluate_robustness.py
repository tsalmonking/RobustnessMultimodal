import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from dataset import RecoveryDataset
from utils import (
    STANDARD_CORRUPTIONS,
    TEXT_PERTURBATIONS,
    compute_classic_metrics,
    save_results_csv,
    plot_confusion_matrix,
)
from config import WEIGHTS_PATH, DATASET_PATH, DATA_CSV, OUTPUT_DIR, NUM_WORKERS
from transformers import AutoTokenizer
from themis_model import get_Themis
import numpy as np


# ---------- tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# ---------- load model ----------
def load_model(device, weights_path=WEIGHTS_PATH):
    model, tokenizer, processor = get_Themis(
        name_llm="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name_img_embed="openai/clip-vit-base-patch32",
    )

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            new_state = {}
            for k, v in state.items():
                key = k.replace("module.", "") if k.startswith("module.") else k
                new_state[key] = v
            model.load_state_dict(new_state)
    else:
        print("Attenzione: file pesi non trovato, uso modello con pesi random.")

    model.to(device).eval()
    return model, tokenizer, processor


def predict_multimodal(model, pil_img, text, device, tokenizer, processor):
    """
    Predizione multimodale con Themis (immagine + testo).
    Ritorna: pred (int), probs (np.ndarray)
    """
    # --- Preprocess immagine ---
    image_inputs = processor(images=pil_img, return_tensors="pt").to(device)
    image_inputs["pixel_values"] = image_inputs["pixel_values"].unsqueeze(1)

    breakpoint()

    # --- Preprocess testo ---
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    breakpoint()

    # --- Forward pass ---
    model.eval()
    with torch.no_grad():
        out = model(image_inputs, text_inputs)

    # --- Estrai logits ---
    logits = out[0] if isinstance(out, tuple) else out

    # --- Softmax e predizione ---
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))

    return pred, probs


# ---------- main evaluation ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(device)

    # Dataset già preprocessato (title+text e immagini salvate)
    dataset = RecoveryDataset(
        csv_file=DATA_CSV,
        image_dir=os.path.join(DATASET_PATH, "images"),
        tokenizer=tokenizer,
        processor=processor,
        max_length=64,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: x[0],  # prendo dict singolo
    )

    rows = []

    # iterate over all corruption x perturbation combinations
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
        for idx, batch in enumerate(pbar):
            pil = batch["image"]  # PIL.Image dal dataset
            text = batch["text"]  # stringa
            label = batch["label"]

            # Applica corruzioni/perturbazioni
            pil_corr = img_fn(pil)
            text_corr = txt_fn(text)

            breakpoint()

            #try:
            pred, probs = predict_multimodal(
                model,
                pil_corr,
                text_corr,
                device,
                tokenizer=tokenizer,
                processor=processor,
            )
            #except Exception as e:
                #print(f"ERROR executing predict_multimodal for idx={idx}: {e}")
                #continue

            y_true.append(label)
            y_pred.append(pred)

            if pred != label:
                samples_changed.append(
                    {
                        "index": idx,
                        "true": label,
                        "pred": pred,
                        "image_variant": img_name,
                        "text_variant": txt_name,
                    }
                )

        # compute metrics
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

            with open(os.path.join(out_subdir, "summary.json"), "w") as f:
                json.dump(
                    {"metrics": row, "examples_changed": samples_changed[:50]},
                    f,
                    indent=2,
                )

    # save overall csv
    save_results_csv(rows, os.path.join(OUTPUT_DIR, "robustness_report.csv"))
    print("DONE. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
 