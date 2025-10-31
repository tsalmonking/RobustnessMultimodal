import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
import torchattacks
import torchvision.transforms as T
import ollama
import json

from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image
from tqdm import tqdm

from utils import load_model
from config import NAME_LLM, NAME_IMG_EMBED
from prompt import LLM_CORRUPTER_PROMPT
from datasets import get_dataset, Recovery_Dataset, recovery_load_annotations_file

BATCH_SIZE = 4
N_TOKENS = 256

# PGD
PGD_ITERS = 20
EPSILON = 4 / 255
ALPHA = EPSILON / (PGD_ITERS * 2.0)

# DeepFool
DF_ITERS = 200
OVERSHOOT = 0.003

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

def use_model(model, tokenizer, processor, news):
    # Text tokenization
    token_txt = tokenizer(news["txt"], return_tensors="pt", padding='max_length', truncation=True, return_attention_mask=False, max_length=N_TOKENS)

    # Image processing
    if isinstance(news["img"], Image.Image):
        process_img = processor(images=news["img"], return_tensors="pt")
        process_img["pixel_values"] = process_img["pixel_values"].unsqueeze(1)
    else:
        process_img = {"pixel_values": news["img"]}
        if process_img["pixel_values"].dim() == 4:  # aggiungi questa riga
            process_img["pixel_values"] = process_img["pixel_values"].unsqueeze(1)

    # Using the model
    with torch.no_grad():
        outputs = model(process_img, token_txt)
        preds = [1 if i > 0.5 else 0 for i in outputs.cpu().detach().numpy()]
    
    return preds

def save_img(img, save_path):
    to_pil = T.ToPILImage()
    to_pil(img.squeeze(0).cpu().float()).save(save_path)

def img_corruption(model, tokenizer, processor, news, type_attack, label):
    token_txt = tokenizer(news["txt"], return_tensors="pt", padding='max_length', truncation=True, return_attention_mask=False, max_length=N_TOKENS)
    process_img = processor(images=news["img"], return_tensors="pt", do_normalize=False)
    wrapped_model = WrappedModel(model, token_txt, processor)
    if type_attack == "pgd":
        attack = torchattacks.PGD(wrapped_model, eps=EPSILON, alpha=ALPHA, steps=PGD_ITERS, random_start=True)
    elif type_attack == "deepfool":
        attack = torchattacks.DeepFool(wrapped_model, steps=DF_ITERS, overshoot=OVERSHOOT)
    corr_img = attack(process_img["pixel_values"], label)
    corr_news = {
        "txt": news["txt"],
        "img": corr_img
    }
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_val = ssim(preds=corr_img.float(), target=process_img["pixel_values"].float())

    return corr_news, ssim_val, process_img["pixel_values"]

def txt_corruption(news):
    response = ollama.chat(
        model="phi3:instruct",
        options={
            "temperature": 0.5,
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional adversarial news rewriter. "
                    "Your goal is to subtly modify a news passage to challenge a fake news detector "
                    "while keeping all factual details intact."
                )
            },
            {
                "role": "user",
                "content": LLM_CORRUPTER_PROMPT + news["txt"]
            }
        ]
    )
    corr_txt = response["message"]["content"].strip()

    corr_news = {
        "txt": corr_txt,
        "img": news["img"]
    }
    return corr_news

def save_results(idx, news, clean_img, img_corr_news_pgd, img_corr_news_df, preds, ssim_pgd, ssim_df):
    os.makedirs("results", exist_ok=True)
    result_dir = f"results/{idx}"
    os.makedirs(result_dir, exist_ok=True)

    save_img(clean_img, os.path.join(result_dir, "clean_img.png"))
    save_img(img_corr_news_pgd["img"], os.path.join(result_dir, "pgd_img.png"))
    save_img(img_corr_news_df["img"], os.path.join(result_dir, "deepfool_img.png"))

    result_data = {
        "index": idx,
        "true_label": preds["label_true"],
        "pred_clean": preds["clean"],
        "pred_img_corr": preds["img_corr"],
        "pred_txt_corr": preds["txt_corr"],
        "pred_multimodal_corr": preds["multimodal_corr"],
        "SSIM_pgd": ssim_pgd,
        "SSIM_deepfool": ssim_df,
        "original_txt": news["txt"],
        "corr_txt": img_corr_news_pgd["txt"]
    }

    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=4)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        device,
        weights_path="../model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt",
        name_llm=NAME_LLM,
        name_img_embed=NAME_IMG_EMBED,
    )
    dataset_test = get_dataset(Recovery_Dataset, recovery_load_annotations_file, N_TOKENS, processor, tokenizer, 
                "../Data/ReCOVery/test.csv",
                "../Data/ReCOVery/images")
    
    subset = torch.utils.data.Subset(dataset_test, list(range(len(dataset_test) // 10)))
    
    dataloader_test = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,generator=torch.Generator(device='cuda'))

    preds = []
    accumulated_labels = []
    with torch.no_grad():
        for images, labels, text, _ in dataloader_test:            
            outputs = model(images, text)
            preds.extend(outputs.cpu().detach().numpy())
            accumulated_labels.extend(labels.cpu().numpy())

        preds = [1 if i > 0.5 else 0 for i in preds]
    
    for i, (pred, label) in tqdm(enumerate(zip(preds, accumulated_labels)), desc="Looking for a multimodality change", leave=False):
        if pred == label:
            news = {
                "txt": dataset_test.texts[i],
                "img": Image.open(os.path.join(dataset_test.img_dir, dataset_test.imgs_path[i])).convert("RGB")
            }
            img_corr_news_pgd, ssim_pgd, _ = img_corruption(model, tokenizer, processor, news, "pgd", torch.tensor([label]))
            txt_corr_news = txt_corruption(news)
            multimodal_corr_news, ssim_df, process_img = img_corruption(model, tokenizer, processor, txt_corr_news, "deepfool", torch.tensor([label]))

            #pred = use_model(model, tokenizer, processor, news)[0]
            img_corr_pred= use_model(model, tokenizer, processor, img_corr_news_pgd)[0]
            txt_corr_pred = use_model(model, tokenizer, processor, txt_corr_news)[0]
            multimodal_corr_pred = use_model(model, tokenizer, processor, multimodal_corr_news)[0]

            if (img_corr_pred == label and txt_corr_pred == label and multimodal_corr_pred != label):
                save_results(i, news, process_img, img_corr_news_pgd, multimodal_corr_news,
                    preds={
                        "label_true": label,
                        "clean": pred,
                        "img_corr": img_corr_pred,
                        "txt_corr": txt_corr_pred,
                        "multimodal_corr": multimodal_corr_pred
                    },
                    ssim_pgd=ssim_pgd,
                    ssim_df=ssim_df
                )

if __name__ == "__main__":
    main()