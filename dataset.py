import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class RecoveryDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer, processor, max_length=64):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs["pixel_values"] = image_inputs["pixel_values"].unsqueeze(0)  # (1,3,H,W)

        text_inputs = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        label = torch.tensor(row["label"], dtype=torch.float)

        return image_inputs["pixel_values"].squeeze(0), label, text_inputs["input_ids"].squeeze(0)

