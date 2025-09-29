import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from dataset import MultimodalDataset
from utils import STANDARD_CORRUPTIONS, TEXT_PERTURBATIONS, compute_classic_metrics, save_results_csv, plot_confusion_matrix
from config import WEIGHTS_PATH, DATASET_PATH, DATA_CSV, OUTPUT_DIR, NUM_WORKERS
from PIL import Image
from transformers import BertTokenizer
from themis_model import get_Themis
import numpy as np
from torchvision import transforms

# ---------- tokenizer ----------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------- load model ----------
def load_model(device, weights_path=WEIGHTS_PATH):
    model, tokenizer, processor = get_Themis(
        name_llm="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name_img_embed="openai/clip-vit-base-patch32"
    )

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            new_state = {}
            for k, v in state.items():
                key = k.replace('module.', '') if k.startswith('module.') else k
                new_state[key] = v
            model.load_state_dict(new_state)

    model.to(device).eval()
    return model, tokenizer, processor

def predict_multimodal(model, pil_img, text, device, tokenizer=None, preprocess=None):
    """
    Wrapper per predizione multimodale.
    INPUT:
        - model: modello Themis
        - pil_img: immagine PIL.Image
        - text: stringa
        - device: 'cuda' o 'cpu'
        - tokenizer: tokenizer testuale (opzionale)
        - preprocess: trasformazioni immagini torchvision (opzionale)
    RETURN:
        - pred: label predetta (int)
        - probs: probabilità softmax (np.array)
    """

    # --- Preprocess immagine ---
    if preprocess is None:
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Applica preprocess PIL -> tensor
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)  # batch_size=1

    # --- Preprocess testo ---
    if tokenizer is not None:
        # Tokenizza e restituisci tensori batch
        text_inputs = tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)
    else:
        # fallback: passa direttamente la stringa
        text_inputs = text

    # --- Forward pass ---
    model.eval()
    with torch.no_grad():
        try:
            if tokenizer is not None:
                # forward compatibile con input tokenizzati
                out = model(img_tensor, **text_inputs)
            else:
                # fallback semplice
                out = model(img_tensor, text_inputs)
        except Exception as e:
            print("Errore nella forward del modello:", e)
            return None, None

    # --- Estrai logits ---
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

    # --- Softmax e predizione ---
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))

    return pred, probs


# ---------- main evaluation ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(device)

    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = MultimodalDataset(csv_path=DATA_CSV, tokenizer=tokenizer, transform=default_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    rows = []

    # iterate over all corruption x perturbation combinations
    combinations = [(img_name, img_fn, txt_name, txt_fn)
                    for img_name, img_fn in STANDARD_CORRUPTIONS.items()
                    for txt_name, txt_fn in TEXT_PERTURBATIONS.items()]

    for img_name, img_fn, txt_name, txt_fn in tqdm(combinations, desc="Corruptions x Perturbations"):
        y_true = []
        y_pred = []
        samples_changed = []

        pbar = tqdm(loader, desc=f"Eval {img_name} x {txt_name}", leave=False)
        for idx, batch in enumerate(pbar):
            pil = batch['image']  # PIL.Image dal dataset
            text = batch['text']  # stringa
            label = batch['label'].item()

            # Applica solo perturbazioni che ritornano PIL.Image o testo
            pil_corr = img_fn(pil)       # img_fn deve restituire PIL.Image
            text_corr = txt_fn(text)     # txt_fn restituisce stringa

            pred, probs = predict_multimodal(model, pil_corr, text_corr, device, tokenizer=tokenizer)

            y_true.append(label)
            y_pred.append(pred)

            if pred != label:
                samples_changed.append({
                    "index": idx,
                    "true": label,
                    "pred": pred,
                    "image_variant": img_name,
                    "text_variant": txt_name
                })

        # compute metrics
        metrics = compute_classic_metrics(y_true, y_pred)
        cm = metrics['confusion_matrix']
        out_subdir = os.path.join(OUTPUT_DIR, f"{img_name}__{txt_name}")
        os.makedirs(out_subdir, exist_ok=True)
        plot_confusion_matrix(cm, labels=list(range(cm.shape[0])),
                              out_file=os.path.join(out_subdir, "confusion_matrix.png"))

        row = {
            "image_variant": img_name,
            "text_variant": txt_name,
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "num_samples": len(y_true),
            "num_changed": len(samples_changed)
        }
        rows.append(row)

        # save some changed examples
        with open(os.path.join(out_subdir, "summary.json"), "w") as f:
            json.dump({
                "metrics": row,
                "examples_changed": samples_changed[:50]
            }, f, indent=2)

    # save overall csv
    save_results_csv(rows, os.path.join(OUTPUT_DIR, "robustness_report.csv"))
    print("DONE. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
