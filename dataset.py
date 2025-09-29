import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import requests
from io import BytesIO

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, tokenizer=None, transform=None, download_images=True, image_dir="images"):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.download_images = download_images
        self.image_dir = image_dir

        if self.download_images and not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---- TEXT ----
        text = str(row["title"]) + " " + str(row["body_text"])

        # ---- IMAGE ----
        image_url = row["image"]
        image = None
        image_path = os.path.join(self.image_dir, f"{row['news_id']}.jpg")

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        elif self.download_images and isinstance(image_url, str) and image_url.startswith("http"):
            try:
                response = requests.get(image_url, timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(image_path)  # cache
            except:
                image = Image.new("RGB", (224, 224), (255, 255, 255))  # placeholder
        else:
            image = Image.new("RGB", (224, 224), (255, 255, 255))  # placeholder

        if self.transform:
            image = self.transform(image)

        # ---- LABEL ----
        label = int(row["reliability"])

        return {
          "text": text,       # singola stringa
          "image": image,     # PIL.Image.Image
          "label": label,
          "index": idx
        }
