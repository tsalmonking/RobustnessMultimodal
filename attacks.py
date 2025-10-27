import os
import torch
import json
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

# Custom modules
from corruptions import IMAGE_CORRUPTIONS, TEXT_PERTURBATIONS
from config import OUTPUT_DIR, DEBUG_MODE
from utils import (
    compute_metrics,
    compute_robustness_metrics,
    plot_confusion_matrix,
    info,
)


def black_box(
    model,
    tokenizer,
    processor,
    accelerator,
    ids,
    texts,
    images,
    labels,
    clean_preds,
    device,
):
    combinations = [
        (img_name, img_fun, txt_name, txt_fun)
        for img_name, img_fun in IMAGE_CORRUPTIONS.items()
        for txt_name, txt_fun in TEXT_PERTURBATIONS.items()
    ]

    loss_fn = torch.nn.BCELoss()
    best_loss = -float("inf")
    best_outputs = None
    best_images = None
    best_texts = None
    best_img_name = None
    best_txt_name = None
    
    out_dir = os.path.join(OUTPUT_DIR, "black", "corr")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "texts"), exist_ok=True)

    for img_name, img_fun, txt_name, txt_fun in tqdm(
        combinations,
        desc="Combinations",
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        leave=False,
    ):
        # Applying perturbations
        corr_texts = txt_fun(texts)
        corr_pils = img_fun(images)

        # Texts tokenization
        if DEBUG_MODE:
            corr_text_inputs = tokenizer(
                corr_texts,
                padding="max_length",
                truncation=True,
                max_length=450,
                return_tensors="pt",
            )
        else:
            corr_text_inputs = tokenizer(
                corr_texts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Images processing
        corr_image_inputs = processor(images=corr_pils, return_tensors="pt")
        corr_image_inputs["pixel_values"] = corr_image_inputs["pixel_values"].unsqueeze(
            1
        )

        # Loss calculation
        with torch.no_grad():
            outputs = model(corr_image_inputs, corr_text_inputs)
            corr_preds = [1 if i > 0.5 else 0 for i in outputs]
            labels_preds = outputs.squeeze()
            labels_tensor = torch.tensor(labels, dtype=torch.float32).squeeze()
            loss_val = loss_fn(labels_preds, labels_tensor)
        
        for i in range(len(clean_preds)):
            if int(clean_preds[i]) != int(corr_preds[i]):
                images[i].save(os.path.join(out_dir, "images", f"{ids[i]}_clean.png"))
                corr_pils[i].save(
                    os.path.join(out_dir, "images", f"{ids[i]}_{img_name}.png")
                )
                json_path = os.path.join(out_dir, "texts", f"{ids[i]}_{txt_name}.json")
                data = {
                    "label": int(labels[i]),
                    "label_pred_clean": int(clean_preds[i]),
                    "label_pred_corr": int(corr_preds[i]),
                    "clean_text": texts[i],
                    "corr_txt": corr_texts[i],
                }

                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        # Searching for the best loss
        if loss_val.item() > best_loss:
            best_loss = loss_val.item()
            best_outputs = outputs.detach().clone()

    return best_outputs


def white_box(
    model,
    tokenizer,
    processor,
    accelerator,
    ids,
    texts,
    images,
    labels,
    clean_preds,
    device,
    iters,
    eps_img,
    alpha_img,
    eps_txt,
    alpha_txt,
    text_perturbation,
):
    # ----- CORRUPTED FORWARD -----
    corr_image_inputs, corr_text_inputs, best_text_corruption = PGDattack(
        model,
        tokenizer,
        processor,
        accelerator,
        images,
        texts,
        labels,
        device,
        iters,
        eps_img,
        alpha_img,
        eps_txt,
        alpha_txt,
        text_perturbation,
    )

    with torch.no_grad():
        outputs = model(corr_image_inputs, corr_text_inputs)
        corr_preds = [1 if i > 0.5 else 0 for i in outputs]

    # Changing in corr_image_inputs into a list of images to visualize them
    pixel_values = corr_image_inputs["pixel_values"].squeeze(1)
    best_images = []
    for img_tensor in pixel_values:
        # Convert [C, H, W] → [H, W, C]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        # If normalized, rescale to [0,255]
        img_np = (img_np * 255).clip(0, 255).astype("uint8")
        img_pil = Image.fromarray(img_np)
        best_images.append(img_pil)

    # Changing corr_Texts_inputs in a list of string to visualize them
    best_texts = tokenizer.batch_decode(
        corr_text_inputs["input_ids"], skip_special_tokens=True
    )

    # Saving best images and texts from the batch with a 50% of probability
    if text_perturbation == "true":
        out_dir = os.path.join(OUTPUT_DIR, "white", "corr")
    else:
        out_dir = os.path.join(OUTPUT_DIR, "white_img_only", "corr")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "texts"), exist_ok=True)
    for i in range(len(clean_preds)):
        if int(clean_preds[i]) != int(corr_preds[i]):
            images[i].save(os.path.join(out_dir, "images", f"{ids[i]}_clean.png"))
            best_images[i].save(os.path.join(out_dir, "images", f"{ids[i]}.png"))
            json_path = os.path.join(
                out_dir, "texts", f"{ids[i]}{best_text_corruption}.json"
            )
            data = {
                "label": int(labels[i]),
                "label_pred_clean": int(clean_preds[i]),
                "label_pred_corr": int(corr_preds[i]),
                "clean_text": texts[i],
                "corr_txt": best_texts[i],
            }

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    return outputs


def multimodal_attack(
    model,
    tokenizer,
    processor,
    accelerator,
    attack_mode,
    loader,
    device,
    iters=20,
    eps_img=8 / 255,
    alpha_img=2 / 255,
    eps_txt=3.0,
    alpha_txt=0.5,
    text_perturbation="false",
):
    info(f"\nStarting {attack_mode} robustness evaluation...")

    y_true = []
    y_pred_clean, y_pred_corr = [], []

    for ids, texts, images, labels in tqdm(loader, desc="Evaluating"):
        # ----- CLEAN FORWARD -----
        if DEBUG_MODE:
            clean_text_inputs = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=450,
                return_tensors="pt",
            )
        else:
            clean_text_inputs = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        clean_image_inputs = processor(images=images, return_tensors="pt")
        clean_image_inputs["pixel_values"] = clean_image_inputs[
            "pixel_values"
        ].unsqueeze(1)
        with torch.no_grad():
            clean_outputs = model(clean_image_inputs, clean_text_inputs)
        clean_preds = [1 if i > 0.5 else 0 for i in clean_outputs]
        y_pred_clean.extend(clean_preds)

        # ----- CORRUPTED FORWARD -----
        if attack_mode == "black":
            clean_outputs = black_box(
                model,
                tokenizer,
                processor,
                accelerator,
                ids,
                texts,
                images,
                labels,
                clean_preds,
                device,
            )
        else:
            clean_outputs = white_box(
                model,
                tokenizer,
                processor,
                accelerator,
                ids,
                texts,
                images,
                labels,
                clean_preds,
                device,
                iters,
                eps_img,
                alpha_img,
                eps_txt,
                alpha_txt,
                text_perturbation,
            )
        #torch.cuda.empty_cache()
        corr_preds = [1 if i > 0.5 else 0 for i in clean_outputs]
        y_pred_corr.extend(corr_preds)

        y_true.extend(labels)

    # ----- RESULTS -----
    # Classic metrics
    metrics_clean, cm_clean = compute_metrics(y_true, y_pred_clean)
    metrics_corr, cm_corr = compute_metrics(y_true, y_pred_corr)
    # Robustness metrics
    robustness_metrics = compute_robustness_metrics(y_true, y_pred_clean, y_pred_corr)

    # Save results
    if attack_mode == "white" and text_perturbation == "false":
        dir = os.path.join(OUTPUT_DIR, "white_img_only")
    else:
        dir = os.path.join(OUTPUT_DIR, attack_mode)

    clean_dir = os.path.join(dir, "clean")
    corr_dir = os.path.join(dir, "corr")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)
    with open(os.path.join(clean_dir, "metrics_clean.json"), "w") as f:
        json.dump({"metrics": metrics_clean}, f, indent=2)
    with open(os.path.join(corr_dir, "metrics_corr.json"), "w") as f:
        json.dump({"metrics": metrics_corr}, f, indent=2)
    with open(os.path.join(dir, "robustness_metrics.json"), "w") as f:
        json.dump({"metrics": robustness_metrics}, f, indent=2)

    print("Clean Confusion Matrix")
    print(cm_clean)
    print("Corr Confusion Matrix")
    print(cm_corr)
    print("\n--- Robustness Metrics ---")

    print(
        f"Adversarial Accuracy: {robustness_metrics['accuracy_on_corrupted'] * 100:.2f}%, percentage of inputs correctly classified after the adversarial perturbation."
    )
    print(
        f"Delta Accuracy: {robustness_metrics['delta_accuracy'] * 100:.2f}%, difference between clean inputs and corrupted inputs"
    )
    print(
        f"Flip Rate: {robustness_metrics['flip_rate'] * 100:.2f}%, percentage of inputs where the model's prediction changed after the perturbation."
    )
    print(
        f"Attack Success Rate (ASR): {robustness_metrics['attack_success_rate'] * 100:.2f}%, proportion of originally correct classifications that were flipped to an incorrect label"
    )

    plot_confusion_matrix(
        cm_clean,
        labels=range(cm_clean.shape[0]),
        out_file=os.path.join(clean_dir, "confusion_matrix.png"),
    )

    plot_confusion_matrix(
        cm_corr,
        labels=range(cm_corr.shape[0]),
        out_file=os.path.join(corr_dir, "confusion_matrix.png"),
    )


def forward_pass_manual(model, current_pv, current_emb):
    """Esegue il forward pass manuale come implementato nel codice."""
    b, k, c, h, w = current_pv.shape
    cur_pv_4d = current_pv.reshape(b * k, c, h, w)

    image_features = model.img_embed_model(pixel_values=cur_pv_4d).last_hidden_state
    image_features = model.image_proj(image_features)

    x = torch.cat((image_features, current_emb), dim=1)
    if model.merge_tokens is not None:
        x = model.patch_merger(x)
    for i in range(len(model.h)):
        x = model.h[i](x)[0]
    x = x.mean(dim=1)
    return model.lm_head(x).view(-1)


def PGDattack(
    model,
    tokenizer,
    processor,
    accelerator,
    images,
    texts,
    labels,
    device,
    iters,
    eps_img,
    alpha_img,
    eps_txt,
    alpha_txt,
    text_perturbation="False",
):
    """
    Perform multimodal PGD adversarial attack on both image and text embeddings (optional).
    If text_perturbation == "False", applies PGD only on the image and selects the worst textual corruption.
    """
    model = model.module if hasattr(model, "module") else model
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    labels_tensor = torch.tensor(labels, device=device).float()

    # Embedding matrix (vocabulary embeddings)
    embedding_matrix = model.emb.weight.data

    # -----------------------------
    # Initialize clean inputs
    # -----------------------------
    images_clean = processor(images=images, return_tensors="pt")
    images_clean["pixel_values"] = images_clean["pixel_values"].unsqueeze(1)
    pv = images_clean["pixel_values"].clone().detach()

    if DEBUG_MODE:
        texts_tok = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=450,
            return_tensors="pt",
        )
    else:
        texts_tok = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    emb_baseline = model.emb(texts_tok["input_ids"]).detach()

    # ==============================================================
    # Mode 1: PGD on both image and text embeddings (white-box)
    # ==============================================================

    if text_perturbation.lower() == "true":
        pv.requires_grad_(True)
        emb_adv = emb_baseline.clone().detach().requires_grad_(True)
        delta_img = torch.zeros_like(pv, requires_grad=True)

        best_loss = -float("inf")
        best_pv, best_emb = None, None

        for step in tqdm(
            range(int(iters)),
            desc="PGD multimodal iterations",
            leave=False,
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            cur_pv = pv + delta_img

            with accelerator.autocast():
                logits = forward_pass_manual(model, cur_pv, emb_adv)
                loss = loss_fn(logits, labels_tensor)
                # print(f"Iteration {step}/{iters} | Loss: {loss.item():.4f}")

            if loss.item() > best_loss:
                best_loss = loss.item()
                best_pv = cur_pv.detach().clone()
                best_emb = emb_adv.detach().clone()

            accelerator.backward(loss)
            if accelerator.scaler is not None:
                scale = accelerator.scaler.get_scale()
                delta_img.grad.data = delta_img.grad.data / scale

            # Update image perturbation (L_inf)
            with torch.no_grad():
                delta_img += alpha_img * torch.sign(delta_img.grad)
                delta_img.clamp_(-eps_img, eps_img)
            delta_img.grad.zero_()

            if accelerator.scaler is not None:
                scale = accelerator.scaler.get_scale()
                emb_adv.grad.data = emb_adv.grad.data / scale
            # Update text embeddings (L2)
            grad_emb = emb_adv.grad.detach()
            B, S, D = grad_emb.shape
            grad_emb_flat = grad_emb.view(B, -1)
            grad_norm = grad_emb_flat.norm(dim=1).clamp(min=1e-8)
            step_emb = alpha_txt * grad_emb / grad_norm.view(B, 1, 1)

            with torch.no_grad():
                emb_adv += step_emb

            # Project back to L2 ball
            delta_e = emb_adv.data - emb_baseline.data
            delta_e_flat_norm = delta_e.view(B, -1).norm(dim=1)
            exceed = delta_e_flat_norm > eps_txt
            if exceed.any():
                factor = eps_txt / delta_e_flat_norm[exceed]
                emb_adv.data[exceed] = (
                    emb_baseline.data[exceed] + delta_e[exceed] * factor[:, None, None]
                )
            delta_img.grad.zero_()
            emb_adv.grad.zero_()

        # Fallback in case best values were never updated
        if best_pv is None or best_emb is None:
            best_pv = (pv + delta_img).detach().clone()
            best_emb = emb_adv.detach().clone()

        images_adv = {"pixel_values": best_pv}

        # ----------------------------------------------------------
        # Project perturbed embeddings back to nearest token IDs
        # ----------------------------------------------------------
        final_emb = best_emb
        B, S, D = final_emb.shape
        final_emb_flat = final_emb.view(-1, D)

        final_emb_norm = F.normalize(final_emb_flat, p=2, dim=1)
        embedding_matrix_norm = F.normalize(embedding_matrix, p=2, dim=1)
        similarities = torch.matmul(final_emb_norm, embedding_matrix_norm.T)
        new_input_ids_flat = torch.argmax(similarities, dim=-1)
        new_input_ids = new_input_ids_flat.view(B, S)

        text_adv = {"input_ids": new_input_ids}

        return images_adv, text_adv, ""

    # ==============================================================
    # Mode 2: PGD only on the image (white-box) + black-box text search
    # ==============================================================

    else:
        best_corr_name = "clean"
        max_loss_corr = -float("inf")
        text_corruptions_results = {}

        # Search for the worst text corruption
        for name, txt_fun in tqdm(
            TEXT_PERTURBATIONS.items(),
            desc="Best text perturbation search",
            leave=False,
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            corr_texts = txt_fun(texts)
            if DEBUG_MODE:
                texts_tok_corr = tokenizer(
                    corr_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=450,
                    return_tensors="pt",
                )
            else:
                texts_tok_corr = tokenizer(
                    corr_texts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            emb_corr = model.emb(texts_tok_corr["input_ids"]).detach()

            with torch.no_grad():
                logits_corr = forward_pass_manual(model, pv, emb_corr)
                loss_corr = loss_fn(logits_corr, labels_tensor).item()

            text_corruptions_results[name] = {
                "texts_tok": texts_tok_corr,
                "loss": loss_corr,
            }

            if loss_corr > max_loss_corr:
                max_loss_corr = loss_corr
                best_corr_name = name

        best_text_tok = text_corruptions_results[best_corr_name]["texts_tok"]
        best_emb = model.emb(best_text_tok["input_ids"]).detach()

        # PGD on image only
        pv.requires_grad_(True)
        delta_img = torch.zeros_like(pv, requires_grad=True)

        for step in tqdm(
            range(int(iters)),
            desc="PGD image-only iterations",
            leave=False,
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            cur_pv = pv + delta_img

            with torch.enable_grad():
                logits = forward_pass_manual(model, cur_pv, best_emb)
                loss = loss_fn(logits, labels_tensor)
                # print(f"\nIteration {step}/{iters} | Loss: {loss.item():.4f}")

            model.zero_grad()
            loss.backward()

            # Update image perturbation (L_inf)
            delta_img.data = torch.clamp(
                delta_img.data + alpha_img * torch.sign(delta_img.grad),
                -eps_img,
                eps_img,
            )
            delta_img.grad.zero_()

        images_adv = {"pixel_values": (pv + delta_img).detach().clone()}
        text_adv = best_text_tok

        return images_adv, text_adv, "_" + best_corr_name
