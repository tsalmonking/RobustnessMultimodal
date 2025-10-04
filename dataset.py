import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

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

        # --- testo (title + text già preprocessati nel CSV) ---
        text = str(row["title"]) + " " + str(row["text"])

        # --- immagine ---
        img_path = os.path.join(self.image_dir, f"{row['id']}.jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), (255, 255, 255))  # fallback bianco

        # --- label ---
        label = int(row["label"])

        return {
            "image": image,   # PIL.Image
            "text": text,     # stringa
            "label": label,   # int
            "index": idx
        }
