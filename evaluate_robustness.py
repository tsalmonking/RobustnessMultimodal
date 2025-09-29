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
from transformers import BertTokenizer, AutoModel, CLIPVisionModel
from themis_model import get_Themis
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------- load model ----------
def load_model(device, weights_path=WEIGHTS_PATH):
    # inizializza Themis tramite la funzione helper
    model, tokenizer, processor = get_Themis(
        name_llm = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name_img_embed = "openai/clip-vit-base-patch32"
    )

    # carica pesi salvati (se compatibili)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception as _:
            # gestisce prefisso 'module.' se presente
            new_state = {}
            for k, v in state.items():
                key = k.replace('module.', '') if k.startswith('module.') else k
                new_state[key] = v
            model.load_state_dict(new_state)

    model.to(device).eval()
    return model, tokenizer, processor

# ---------- wrapper predictor (adattalo alla forward del tuo modello) ----------
def predict_multimodal(model, pil_img, text, device, tokenizer=None, preprocess=None ):
    """
    RETURN: pred_label (int), probs (np.array)
    Important: adapt this to how your model expects inputs.
    """
    # Default preprocess: torchvision transforms as in dataset
    if preprocess is None:
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    # text -> tokenization depends on model. If None, pass raw string in a dict
    if tokenizer:
        tok = tokenizer([text], return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            out = model(img_tensor, **tok)  # TODO: adattare a forward signature
    else:
        # fallback: assume model.forward accepts (image_tensor, text)
        with torch.no_grad():
            out = model(img_tensor, text)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs

# ---------- main evaluation ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(device)
    dataset = MultimodalDataset(csv_path=DATA_CSV, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    rows = []

    # tqdm interno: iterate over dataset
    pbar = tqdm(loader, desc=f"Eval {img_name} x {txt_name}", leave=False)
    for i, batch in enumerate(pbar):
        # ---- IMAGE ----
        pil = batch['image']  # batch_size=1, quindi pil è già una lista di 1 immagine
        if isinstance(pil, list):
            pil = pil[0]  # estrai PIL.Image

        # ---- LABEL ----
        label = int(batch['label'].item()) if isinstance(batch['label'], torch.Tensor) else int(batch['label'][0])

        # ---- TEXT ----
        text_inputs = batch['text']  # ora è un dict di tensori
        # Se vuoi passare al modello come batch
        # pred, probs = predict_multimodal(model, pil_corr, text_inputs, tokenizer=tokenizer)
        # Altrimenti se vuoi una singola stringa per la perturbation:
        text = tokenizer.decode(text_inputs['input_ids'][0], skip_special_tokens=True)

        # ---- APPLICA CORRUPTION / PERTURBAZIONE ----
        pil_corr = img_fn(pil)
        text_corr = txt_fn(text)

        # ---- PREDICT ----
        try:
            pred, probs = predict_multimodal(model, pil_corr, text_corr, tokenizer=tokenizer)
        except Exception as e:
            print("ERROR executing predict_multimodal. Adatta la funzione di prediction al tuo modello.\n", e)
            return

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
        plot_confusion_matrix(cm, labels=list(range(cm.shape[0])), out_file=os.path.join(out_subdir, "confusion_matrix.png"))

        row = {
            "image_variant": img_name,
            "text_variant": txt_name,
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "num_samples": len(y_true),
            "num_changed": len([1 for a,b in zip(y_true,y_pred) if a!=b])
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
