import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torchattacks
import json
import glob
import argparse
import torchvision.transforms as T
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM

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
    plot_roc,
    compute_threshold,
    preds_fusion,
    use_model,
    save_results,
    img_corruption,
    bertattack,
    bertattack_text_only_single,
    txt_corruption,
    WrappedModel,
    cleanup_cuda
)
from config import (
    NAME_LLM, 
    NAME_IMG_EMBED, 
    WEIGHTS_PATH, 
    BATCH_SIZE, 
    N_TOKENS,
    THRESHOLD,
    TARGETED,
    SOURCE_LABEL,
    TARGET_LABEL,
    PGD_ITERS, 
    EPSILON, 
    K_BERT_ATTACK, 
    THRESHOLD_PRED_SCORE,
    MAX_WORDS_TO_ATTACK,
    MAX_CANDIDATES_PER_WORD,
    MAX_WORDS_FOR_IMPORTANCE,
)
import my_datasets

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
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--targeted", type=bool, default=TARGETED)
    parser.add_argument("--source_label", type=int, default=SOURCE_LABEL, choices=(0,1))
    parser.add_argument("--target_label", type=int, default=TARGET_LABEL, choices=(0,1))
    parser.add_argument("--set_params", type=bool, default=True)
    parser.add_argument("--pgd_iters", type=int, default=PGD_ITERS)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--alpha_factor", type=float, default=2.0)
    parser.add_argument("--results_path", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="Fakeddit", choices=list(dataset_classes.keys()))
    parser.add_argument("--k", type=int, default=K_BERT_ATTACK)
    parser.add_argument("--threshold_pred_score", type=bool, default=THRESHOLD_PRED_SCORE)
    parser.add_argument("--max_words_to_attack", type=int, default=MAX_WORDS_TO_ATTACK)
    parser.add_argument("--max_candidates_per_word", type=int, default=MAX_CANDIDATES_PER_WORD)
    parser.add_argument("--max_words_for_importance", type=int, default=MAX_WORDS_FOR_IMPORTANCE)
    args = parser.parse_args()

    device_mm = torch.device("cuda:0")
    device_txt = torch.device("cuda:1")
    device_img = torch.device("cuda:2")
    device_mlm = torch.device("cuda:3")

    model, tokenizer, processor = load_model(device_mm, args)
    txt_model, tokenizer_txt, processor_txt = load_model(device_txt, args, modality="text")
    img_model, tokenizer_img, processor_img = load_model(device_img, args, modality="image")

    # Threshold computations
    thr_multimodal_cross = args.threshold#compute_threshold(model, processor, tokenizer, dataset_classes, load_functions, args, device_mm, device_txt, device_img, "multimodal")
    thr_unimodal_txt = args.threshold#compute_threshold(txt_model, processor_txt, tokenizer_txt, dataset_classes, load_functions, args, device_mm, device_txt, device_img, "unimodal_txt")
    thr_unimodal_img = args.threshold#compute_threshold(img_model, processor_img, tokenizer_img, dataset_classes, load_functions, args, device_mm, device_txt, device_img, "unimodal_img")
    
    thr_multimodal_fusion_mean = args.threshold#compute_threshold(txt_model, processor, tokenizer, dataset_classes, load_functions, args, device_mm, device_txt, device_img, None, img_model, "mean", thr_multimodal_cross)
    thr_multimodal_fusion_min = args.threshold#compute_threshold(txt_model, processor, tokenizer, dataset_classes, load_functions, args, device_mm, device_txt, device_img, None, img_model, "min", thr_multimodal_cross)
    thr_multimodal_fusion_max = args.threshold#compute_threshold(txt_model, processor, tokenizer, dataset_classes, load_functions, args, device_mm, device_txt, device_img, None, img_model, "max", thr_multimodal_cross)

    # Load BERT model and tokenizer for text corruption
    bertattack_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    bertattack_mlm = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device_mlm)
    bertattack_mlm.eval()

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

    # Inizialization og scores and logits of txt only and img only
    unimodal_txt_outputs = []
    unimodal_img_outputs = []
    unimodal_txt_logits = []
    unimodal_img_logits = []
    unimodal_txt_corr_outputs = []
    unimodal_txt_corr_logits = []
    unimodal_img_corr_outputs = []
    unimodal_img_corr_logits = []

    # Logits of multimodal fusion
    multimodal_fusion_mean_outputs = []
    multimodal_fusion_min_outputs = []
    multimodal_fusion_max_outputs = []
    multimodal_fusion_mean_corr_outputs = []
    multimodal_fusion_min_corr_outputs = []
    multimodal_fusion_max_corr_outputs = []
    count = 0
    to_pil = T.ToPILImage()
    for images, labels, text, _, indices in tqdm(dataloader_test, desc="Evaluating", total=len(dataloader_test)):
        # if count == 10:   break
        # count += 1
        images = {k: v.to(device_mm) if torch.is_tensor(v) else v for k, v in images.items()}
        text = {k: v.to(device_mm) if torch.is_tensor(v) else v for k, v in text.items()}
        labels = labels.to(device_mm)
        images_for_img = {k: v.to(device_img) if torch.is_tensor(v) else v for k, v in images.items()}
        text_for_txt = {k: v.to(device_txt) if torch.is_tensor(v) else v for k, v in text.items()}

        multimodal_corr_txts_list = []
        multimodal_corr_imgs_pil_list = []
        unimodal_clean_imgs_pil_list = []
        unimodal_corr_imgs_list = []
        unimodal_corr_txts_list = []
        indices = indices.tolist()
        
        # Clean predictions
        with torch.no_grad():
            outputs, _ = model(images, text)
            preds = [args.target_label if i > thr_multimodal_cross else args.source_label for i in outputs.cpu().detach().numpy()]
            y_preds.extend(preds)
            multimodal_cross_outputs.extend(outputs.cpu().detach().numpy())
            y_true.extend(labels.cpu().numpy())

            txt_outputs, txt_logits = txt_model(images=None, texts=text_for_txt)
            txt_preds = [args.target_label if i > thr_unimodal_txt else args.source_label for i in txt_outputs.cpu().detach().numpy()]
            y_unimodal_txt_preds.extend(txt_preds)
            unimodal_txt_outputs.extend(txt_outputs.cpu().detach().numpy())
            unimodal_txt_logits.extend(txt_logits.cpu().detach().numpy())

            img_outputs, img_logits = img_model(images=images_for_img, texts=None)
            img_preds = [args.target_label if i > thr_unimodal_img else args.source_label for i in img_outputs.cpu().detach().numpy()]
            y_unimodal_img_preds.extend(img_preds)
            unimodal_img_outputs.extend(img_outputs.cpu().detach().numpy())
            unimodal_img_logits.extend(img_logits.cpu().detach().numpy())

            y_multimodal_fusion_mean_preds.extend(preds_fusion(txt_outputs, img_outputs, "mean", thr_multimodal_fusion_mean)[0])
            y_multimodal_fusion_min_preds.extend(preds_fusion(txt_outputs, img_outputs, "min", thr_multimodal_fusion_min)[0])
            y_multimodal_fusion_max_preds.extend(preds_fusion(txt_outputs, img_outputs, "max", thr_multimodal_fusion_max)[0])

            multimodal_fusion_mean_outputs.extend(preds_fusion(txt_outputs, img_outputs, "mean"))
            multimodal_fusion_min_outputs.extend(preds_fusion(txt_outputs, img_outputs, "min"))
            multimodal_fusion_max_outputs.extend(preds_fusion(txt_outputs, img_outputs, "max"))


        # print("labels:", labels)
        # print("multimodal scores:", multimodal_cross_outputs)
        # print("multimodal preds:", preds)
        # print("munimodal txt scores:", unimodal_txt_outputs)
        # print("unimodal txt logits:", unimodal_txt_logits)
        # print("unimodal img scores:", unimodal_img_outputs)
        # print("unimodal img logits:", unimodal_img_logits)
        
        # print("multimodal mean scores:", multimodal_fusion_mean_outputs)
        # print("multimodal mean preds:", y_multimodal_fusion_mean_preds)
        # print("multimodal min scores:", multimodal_fusion_min_outputs)
        # print("multimodal min preds:", y_multimodal_fusion_min_preds)
        # print("multimodal max scores:", multimodal_fusion_max_outputs)
        # print("multimodal max preds:", y_multimodal_fusion_max_preds)

        # Challenging the model
        for i, (pred, txt_pred, img_pred, output, label) in tqdm(enumerate(zip(preds, txt_preds, img_preds, outputs.cpu().detach().numpy(), labels.cpu().numpy().tolist())), desc="Challenging the model", total=len(labels), leave=False):
            # Clean news
            news = {
                "txt": dataset_test.texts[indices[i]],
                "img": Image.open(os.path.join(dataset_test.img_dir, dataset_test.imgs_path[indices[i]])).convert("RGB"),
            }
            unimodal_clean_imgs_pil_list.append(news["img"])
            # Only consider correctly classified samples
            if label == args.source_label:
                img_corr_news, ssim_pgd, proccess_img = img_corruption(model, tokenizer, processor, args, news, torch.tensor([label], device=device_mm))
                with torch.no_grad():
                    txt_corr_news, txt_similarity = bertattack(
                        model=model,
                        themis_tokenizer=tokenizer,
                        processor=processor,
                        args=args,
                        news=news,
                        label=label,
                        device=device_mm,
                        bert_tokenizer=bertattack_tokenizer,
                        mlm_model=bertattack_mlm,
                        mlm_device=device_mlm,
                    )
                    torch.cuda.empty_cache()

                # Ensure the text corruption is effective, if not use the original text as corrupted text
                if txt_similarity < 0.5:
                    txt_corr_news = news
                    txt_similarity = 1.0

                # Create multimodal corrupted news
                multimodal_corr_news = {
                    "txt": txt_corr_news["txt"],
                    "img": img_corr_news["img"],
                }
                # Get predictions on corrupted images only
                img_corr_pred, img_corr_output, _ = use_model(model, tokenizer, processor, args, img_corr_news, thr_multimodal_cross)
                # Get predictions on corrupted text only
                txt_corr_pred, txt_corr_output, _ = use_model(model, tokenizer, processor, args, txt_corr_news, thr_multimodal_cross)
                # Get predictions on corrupted images and text
                multimodal_corr_pred, multimodal_corr_output, _  = use_model(model, tokenizer, processor, args, multimodal_corr_news, thr_multimodal_cross)
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
                            "thr_multimodal_cross": thr_multimodal_cross,
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
                img_corr = to_pil(img_corr_news["img"].squeeze(0).cpu())
            else:
                img_corr = news["img"]
                txt_corr_news = news
                txt_similarity = 1.0
                ssim_pgd = 1.0
            multimodal_corr_txts_list.append(txt_corr_news["txt"])
            multimodal_corr_imgs_pil_list.append(img_corr)

            if label == args.source_label:
                unimodal_clean_image = processor_img(images=news["img"], return_tensors="pt", do_normalize=False)
                unimodal_clean_image = {k: v.to(device_img) if torch.is_tensor(v) else v for k, v in unimodal_clean_image.items()}
                wrapped_img_model = WrappedModel(img_model, fixed_txt=None, processor=processor_img).to(device_img)
                unimodal_img_attack = torchattacks.PGD(wrapped_img_model, eps=args.epsilon, alpha=args.epsilon / (args.pgd_iters * args.alpha_factor), steps=args.pgd_iters, random_start=True,)
                unimodal_corr_image = unimodal_img_attack(unimodal_clean_image["pixel_values"], torch.tensor([label], device=device_img).long())
                img_pil = to_pil(unimodal_corr_image.squeeze(0).cpu())
            else:
                img_pil = news["img"]
            unimodal_corr_imgs_list.append(img_pil)
            
            if label == args.source_label:
                with torch.no_grad():
                    unimodal_corr_txt, txt_similarity = bertattack_text_only_single(
                        model=txt_model,
                        themis_tokenizer=tokenizer_txt,
                        args=args,
                        txt=news["txt"],
                        label=label,
                        device=device_txt,
                        bert_tokenizer=bertattack_tokenizer,
                        mlm_model=bertattack_mlm,
                        mlm_device=device_mlm,
                    )
                    torch.cuda.empty_cache()

                if txt_similarity < 0.5:
                    unimodal_corr_txt = news["txt"]
                    txt_similarity = 1.0

                unimodal_corr_txts_list.append(unimodal_corr_txt)
            else:
                unimodal_corr_txts_list.append(news["txt"])

        # Tokenization of multimodal corrupted texts
        multimodal_corr_txts = tokenizer(multimodal_corr_txts_list, return_tensors="pt", padding="max_length", truncation=True, return_attention_mask=False, max_length=args.n_tokens).to(device_mm)
        multimodal_corr_txts = {"input_ids": multimodal_corr_txts.input_ids.unsqueeze(1)}
        # Tokenization of unimodal corrupted texts
        unimodal_corr_txts = tokenizer_txt(unimodal_corr_txts_list, return_tensors="pt", padding="max_length", truncation=True, return_attention_mask=False, max_length=args.n_tokens).to(device_txt)
        unimodal_corr_txts = {"input_ids": unimodal_corr_txts.input_ids.unsqueeze(1)}

        # Processing of multimodal corrupted images
        multimodal_corr_imgs = processor(images=multimodal_corr_imgs_pil_list, return_tensors="pt", do_normalize=False).to(device_mm)
        mean = torch.tensor(processor.image_mean, device=device_mm).view(1, -1, 1, 1)
        std = torch.tensor(processor.image_std, device=device_mm).view(1, -1, 1, 1)
        multimodal_corr_imgs = {"pixel_values": ((multimodal_corr_imgs["pixel_values"] - mean) / std)}
        multimodal_corr_imgs["pixel_values"] = multimodal_corr_imgs["pixel_values"].unsqueeze(1)
        images["pixel_values"] = images["pixel_values"].unsqueeze(1)
        multimodal_corr_imgs_copy = {k: v.to(device_mm) for k, v in multimodal_corr_imgs.items()}

        # Processing of unimodal corrupted images
        unimodal_corr_imgs = processor_img(images=unimodal_corr_imgs_list, return_tensors="pt", do_normalize=False).to(device_img)
        mean = torch.tensor(processor_img.image_mean, device=device_img).view(1, -1, 1, 1)
        std = torch.tensor(processor_img.image_std, device=device_img).view(1, -1, 1, 1)
        unimodal_corr_imgs = {"pixel_values": ((unimodal_corr_imgs["pixel_values"] - mean) / std)}
        unimodal_corr_imgs["pixel_values"] = unimodal_corr_imgs["pixel_values"].unsqueeze(1)
        unimodal_corr_imgs_copy = {k: v.to(device_img) for k, v in unimodal_corr_imgs.items()}

        # Get unimodal corrupted images
        # unimodal_clean_images = processor_img(images=unimodal_clean_imgs_pil_list, return_tensors="pt", do_normalize=False)
        # unimodal_clean_images = {k: v.to(device_img) if torch.is_tensor(v) else v for k, v in unimodal_clean_images.items()}
        # wrapped_img_model = WrappedModel(img_model, fixed_txt=None, processor=processor_img).to(device_img)
        # unimodal_img_attack = torchattacks.PGD(wrapped_img_model, eps=args.epsilon, alpha=args.epsilon / (args.pgd_iters * args.alpha_factor), steps=args.pgd_iters, random_start=True,)
        # unimodal_corr_images = unimodal_img_attack(unimodal_clean_images["pixel_values"], labels.to(device_img).long())
        # unimodal_corr_images_for_model = {"pixel_values": unimodal_corr_images.unsqueeze(1)}
        # print("unimodal clean:", unimodal_clean_images["pixel_values"].shape)
        # print("unimodal corr:", unimodal_corr_images.shape)
        # print("mean diff:", torch.mean(torch.abs(unimodal_clean_images["pixel_values"] - unimodal_corr_images)).item())
        # print("max diff:", torch.max(torch.abs(unimodal_clean_images["pixel_values"] - unimodal_corr_images)).item())

        # Get unimodal corrupted texts
        # unimodal_corr_txts_cpu, _ = bertattack_text_only(
        #     model=txt_model,
        #     themis_tokenizer=tokenizer_txt,
        #     args=args,
        #     dataset=dataset_test,
        #     indices=indices,
        #     labels=labels,
        #     device=device_txt,
        #     bert_tokenizer=bertattack_tokenizer,
        #     mlm_model=bertattack_mlm,
        #     mlm_device=device_mlm,
        # )
        # cleanup_cuda()
        #unimodal_corr_txts = {k: v.to(device_txt) for k, v in multimodal_corr_txts.items()}

        # Get predictions on corrupted samples in batch
        with torch.no_grad():
            batch_multimodal_cross_txt_corr_outputs, _ = model(images, multimodal_corr_txts)
            txt_corr_preds = [args.target_label if i > thr_multimodal_cross else args.source_label for i in batch_multimodal_cross_txt_corr_outputs.cpu().detach().numpy()]
            y_multimodal_cross_txt_corr_preds.extend(txt_corr_preds)
            multimodal_cross_txt_corr_outputs.extend(batch_multimodal_cross_txt_corr_outputs.cpu().detach().numpy())

            batch_multimodal_cross_img_corr_outputs, _ = model(multimodal_corr_imgs, text)
            img_corr_preds = [args.target_label if i > thr_multimodal_cross else args.source_label for i in batch_multimodal_cross_img_corr_outputs.cpu().detach().numpy()]
            y_multimodal_cross_img_corr_preds.extend(img_corr_preds)
            multimodal_cross_img_corr_outputs.extend(batch_multimodal_cross_img_corr_outputs.cpu().detach().numpy())

            multimodal_corr_imgs["pixel_values"] = multimodal_corr_imgs["pixel_values"].unsqueeze(1)
            batch_multimodal_cross_corr_outputs, _ = model(multimodal_corr_imgs, multimodal_corr_txts)
            multimodal_cross_corr_preds = [args.target_label if i > thr_multimodal_cross else args.source_label for i in batch_multimodal_cross_corr_outputs.cpu().detach().numpy()]
            y_multimodal_cross_corr_preds.extend(multimodal_cross_corr_preds)
            multimodal_cross_corr_outputs.extend(batch_multimodal_cross_corr_outputs.cpu().detach().numpy())

            unimodal_corr_txts = {
                k: v.to(device_txt, non_blocking=True) for k, v in unimodal_corr_txts.items()
            }
            txt_corr_outputs, txt_corr_logits = txt_model(images=None, texts=unimodal_corr_txts)
            unimodal_txt_corr_preds = [args.target_label if i > thr_unimodal_txt else args.source_label for i in txt_corr_outputs.cpu().detach().numpy()]
            y_unimodal_txt_corr_preds.extend(unimodal_txt_corr_preds)
            unimodal_txt_corr_outputs.extend(txt_corr_outputs.cpu().detach().numpy())
            unimodal_txt_corr_logits.extend(txt_corr_logits.cpu().detach().numpy())

            img_corr_outputs, img_corr_logits = img_model(images=unimodal_corr_imgs_copy, texts=None)
            unimodal_img_corr_preds = [args.target_label if i > thr_unimodal_img else args.source_label for i in img_corr_outputs.cpu().detach().numpy()]
            y_unimodal_img_corr_preds.extend(unimodal_img_corr_preds)
            unimodal_img_corr_outputs.extend(img_corr_outputs.cpu().detach().numpy())
            unimodal_img_corr_logits.extend(img_corr_logits.cpu().detach().numpy())

            y_multimodal_fusion_mean_corr_preds.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, "mean", thr_multimodal_fusion_mean)[0])
            y_multimodal_fusion_min_corr_preds.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, "min", thr_multimodal_fusion_min)[0])
            y_multimodal_fusion_max_corr_preds.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, "max", thr_multimodal_fusion_max)[0])

            multimodal_fusion_mean_corr_outputs.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, "mean"))
            multimodal_fusion_min_corr_outputs.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, "min"))
            multimodal_fusion_max_corr_outputs.extend(preds_fusion(txt_corr_outputs, img_corr_outputs, "max"))
        
        cleanup_cuda(
            unimodal_corr_txts,
            txt_corr_outputs,
            img_corr_outputs,
            txt_corr_logits,
            img_corr_logits,
            unimodal_corr_imgs,
        )

        cleanup_cuda(
            images,
            text,
            labels,
            images_for_img,
            text_for_txt,
            outputs,
            txt_outputs,
            img_outputs,
            multimodal_corr_txts,
            multimodal_corr_imgs,
            multimodal_corr_imgs_copy,
            batch_multimodal_cross_txt_corr_outputs,
            batch_multimodal_cross_img_corr_outputs,
            batch_multimodal_cross_corr_outputs,
        )

    # ----- RESULTS -----
    # Classic metrics on clean samples
    metrics_multimodal_cross_clean, fpr_cross_clean, tpr_cross_clean, cm_multimodal_cross_clean = compute_metrics(y_true, y_preds, multimodal_cross_outputs)
    metrics_unimodal_txt_clean, fpr_txt, tpr_txt, cm_unimodal_txt_clean = compute_metrics(y_true, y_unimodal_txt_preds, unimodal_txt_outputs)
    metrics_unimodal_img_clean, fpr_img, tpr_img, cm_unimodal_img_clean = compute_metrics(y_true, y_unimodal_img_preds, unimodal_img_outputs)
    metrics_multimodal_fusion_mean, fpr_fusion_mean, tpr_fusion_mean, cm_multimodal_fusion_mean = compute_metrics(y_true, y_multimodal_fusion_mean_preds, multimodal_fusion_mean_outputs)
    metrics_multimodal_fusion_min, fpr_fusion_min, tpr_fusion_min,cm_multimodal_fusion_min = compute_metrics(y_true, y_multimodal_fusion_min_preds, multimodal_fusion_min_outputs)
    metrics_multimodal_fusion_max, fpr_fusion_max, tpr_fusion_max, cm_multimodal_fusion_max = compute_metrics(y_true, y_multimodal_fusion_max_preds, multimodal_fusion_max_outputs)

    # Classic metrics on corrupted samples
    metrics_multimodal_cross_corr, fpr_cross_corr, tpr_cross_corr, cm_multimodal_cross_corr = compute_metrics(y_true, y_multimodal_cross_corr_preds, multimodal_cross_corr_outputs)
    metrics_unimodal_txt_corr, fpr_txt_corr, tpr_txt_corr, cm_unimodal_txt_corr = compute_metrics(y_true, y_unimodal_txt_corr_preds, unimodal_txt_corr_outputs)
    metrics_unimodal_img_corr, fpr_img_corr, tpr_img_corr, cm_unimodal_img_corr = compute_metrics(y_true, y_unimodal_img_corr_preds, unimodal_img_corr_outputs)
    metrics_multimodal_fusion_mean_corr, fpr_fusion_mean_corr, tpr_fusion_mean_corr, cm_multimodal_fusion_mean_corr = compute_metrics(y_true, y_multimodal_fusion_mean_corr_preds, multimodal_fusion_mean_corr_outputs)
    metrics_multimodal_fusion_min_corr, fpr_fusion_min_corr, tpr_fusion_min_corr, cm_multimodal_fusion_min_corr = compute_metrics(y_true, y_multimodal_fusion_min_corr_preds, multimodal_fusion_min_corr_outputs)
    metrics_multimodal_fusion_max_corr, fpr_fusion_max_corr, tpr_fusion_max_corr, cm_multimodal_fusion_max_corr = compute_metrics(y_true, y_multimodal_fusion_max_corr_preds, multimodal_fusion_max_corr_outputs)
    metrics_multimodal_txt_corr, fpr_cross_txt_corr, tpr_cross_txt_corr, cm_multimodal_txt_corr = compute_metrics(y_true, y_multimodal_cross_txt_corr_preds, multimodal_cross_txt_corr_outputs)
    metrics_multimodal_img_corr, fpr_cross_img_corr, tpr_cross_img_corr, cm_multimodal_img_corr = compute_metrics(y_true, y_multimodal_cross_img_corr_preds, multimodal_cross_img_corr_outputs)

    # Robustness metrics
    robustness_metrics_multimodal_cross = compute_robustness_metrics(y_true, y_preds, y_multimodal_cross_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)
    robustness_metrics_unimodal_txt = compute_robustness_metrics(y_true, y_unimodal_txt_preds, y_unimodal_txt_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)
    robustness_metrics_unimodal_img = compute_robustness_metrics(y_true, y_unimodal_img_preds, y_unimodal_img_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)
    robustness_metrics_multimodal_fusion_mean = compute_robustness_metrics(y_true, y_multimodal_fusion_mean_preds, y_multimodal_fusion_mean_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)
    robustness_metrics_multimodal_fusion_min = compute_robustness_metrics(y_true, y_multimodal_fusion_min_preds, y_multimodal_fusion_min_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)
    robustness_metrics_multimodal_fusion_max = compute_robustness_metrics(y_true, y_multimodal_fusion_max_preds, y_multimodal_fusion_max_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)
    robustness_metrics_multimodal_txt = compute_robustness_metrics(y_true, y_preds, y_multimodal_cross_txt_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)
    robustness_metrics_multimodal_img = compute_robustness_metrics(y_true, y_preds, y_multimodal_cross_img_corr_preds, targeted=args.targeted, source_label=args.source_label, target_label=args.target_label)

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

    # Plotting clean roc curves
    clean_rocs_info = {
        "Multimodal Cross": {
            "fpr": fpr_cross_clean,
            "tpr": tpr_cross_clean,
            "roc_auc": metrics_multimodal_cross_clean["auc"],
            "color": "blue"
        },
        "Multimodal Mean Fusion": {
            "fpr": fpr_fusion_mean,
            "tpr": tpr_fusion_mean,
            "roc_auc": metrics_multimodal_fusion_mean["auc"],
            "color": "yellow"
        },
        "Multimodal Min Fusion": {
            "fpr": fpr_fusion_min,
            "tpr": tpr_fusion_min,
            "roc_auc": metrics_multimodal_fusion_min["auc"],
            "color": "green"
        },
        "Multimodal Max Fusion": {
            "fpr": fpr_fusion_max,
            "tpr": tpr_fusion_max,
            "roc_auc": metrics_multimodal_fusion_max["auc"],
            "color": "red"
        },
        "Unimodal Text": {
            "fpr": fpr_txt,
            "tpr": tpr_txt,
            "roc_auc": metrics_unimodal_txt_clean["auc"],
            "color": "gray"
        },
        "Unimodal Image": {
            "fpr": fpr_img,
            "tpr": tpr_img,
            "roc_auc": metrics_unimodal_img_clean["auc"],
            "color": "pink"
        },
    }
    plot_roc(clean_rocs_info, os.path.join(clean_dir, "clean_roc_curves.png"))

    # Plotting multimodal corrupted roc curves
    corr_rocs_info = {
        "Multimodal Cross": {
            "fpr": fpr_cross_corr,
            "tpr": tpr_cross_corr,
            "roc_auc": metrics_multimodal_cross_corr["auc"],
            "color": "blue"
        },
        "Multimodal Mean Fusion": {
            "fpr": fpr_fusion_mean_corr,
            "tpr": tpr_fusion_mean_corr,
            "roc_auc": metrics_multimodal_fusion_mean_corr["auc"],
            "color": "yellow"
        },
        "Multimodal Min Fusion": {
            "fpr": fpr_fusion_min_corr,
            "tpr": tpr_fusion_min_corr,
            "roc_auc": metrics_multimodal_fusion_min_corr["auc"],
            "color": "green"
        },
        "Multimodal Max Fusion": {
            "fpr": fpr_fusion_max_corr,
            "tpr": tpr_fusion_max_corr,
            "roc_auc": metrics_multimodal_fusion_max_corr["auc"],
            "color": "red"
        },
        "Multimodal Cross Text Corr": {
            "fpr": fpr_cross_txt_corr,
            "tpr": tpr_cross_txt_corr,
            "roc_auc": metrics_multimodal_txt_corr["auc"],
            "color": "purple"
        },
        "Multimodal Cross Image Corr": {
            "fpr": fpr_cross_img_corr,
            "tpr": tpr_cross_img_corr,
            "roc_auc": metrics_multimodal_img_corr["auc"],
            "color": "brown"
        },
    }
    plot_roc(corr_rocs_info, os.path.join(corr_dir, "corr_roc_curves.png"))

    # Plotting unimodal corrupted roc curves
    corr_unimodal_rocs_info = {
        "Multimodal Cross Text Corr": {
            "fpr": fpr_cross_txt_corr,
            "tpr": tpr_cross_txt_corr,
            "roc_auc": metrics_multimodal_txt_corr["auc"],
            "color": "purple"
        },
        "Multimodal Cross Image Corr": {
            "fpr": fpr_cross_img_corr,
            "tpr": tpr_cross_img_corr,
            "roc_auc": metrics_multimodal_img_corr["auc"],
            "color": "brown"
        },
        "Unimodal Text Corr": {
            "fpr": fpr_txt_corr,
            "tpr": tpr_txt_corr,
            "roc_auc": metrics_unimodal_txt_corr["auc"],
            "color": "gray"
        },
        "Unimodal Image Corr": {
            "fpr": fpr_img_corr,
            "tpr": tpr_img_corr,
            "roc_auc": metrics_unimodal_img_corr["auc"],
            "color": "pink"
        },
    }
    plot_roc(corr_unimodal_rocs_info, os.path.join(corr_dir, "corr_unimodal_roc_curves.png"))

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
        unimodal_txt_logits,
        unimodal_img_logits,
        os.path.join(f"results/{args.dataset}/text_vs_image_clean.png"),
        "CLEAN"
    )

    plot_text_vs_image(
        y_true,
        unimodal_txt_corr_logits,
        unimodal_img_corr_logits,
        os.path.join(f"results/{args.dataset}/text_vs_image_corrupted.png"),
        "CORRUPTED"
    )

    plot_shift_arrows(
        y_true,
        unimodal_txt_logits,
        unimodal_img_logits,
        unimodal_txt_corr_logits,
        unimodal_img_corr_logits,
        os.path.join(f"results/{args.dataset}/shift_arrows.png"),
    )

    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        f.write(f"Model: {args.name_llm}\n")
        f.write(f"Image Embedder: {args.name_img_embed}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Number of Tokens: {args.n_tokens}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Merge Tokens: {args.merge_tokens}\n")
        f.write(f"LoRA Alpha: {args.lora_alpha}\n")
        f.write(f"LoRA R: {args.lora_r}\n")
        f.write(f"LoRA Dropout: {args.lora_dropout}\n")
        f.write(f"Use LoRA: {args.use_lora}\n")
        f.write(f"Is the Attack Targeted: {args.targeted}\n")
        if args.targeted:
            f.write(f"From Which Label: {args.source_label}\n")
            f.write(f"To Which Lable: {args.target_label}\n")
        f.write(f"PGD Iterations: {args.pgd_iters}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write(f"Alpha Factor: {args.alpha_factor}\n")
        f.write(f"K Number of Candidates Bertattack: {args.k}\n")
        f.write(f"Max Words to Attack:{args.max_words_to_attack}\n")
        f.write(f"Max Candidates per Word:{args.max_candidates_per_word}\n")


if __name__ == "__main__":
    main()
