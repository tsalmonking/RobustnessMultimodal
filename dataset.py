import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


# recovery dataset class well commented
class RecoveryDataset(Dataset):
    """A PyTorch Dataset class for loading images and corresponding text data for recovery tasks.
    Each sample consists of an image, a text input, and a label.
    Args:
        csv_file (str): Path to the CSV file containing image IDs, text data, and labels.
        image_dir (str): Directory where images are stored.
    Returns:
        tuple: (text, image, label) where text is a string, image is a PIL image and label is an integer.
    """

    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id']}.jpg")

        id = row["id"]
        image = Image.open(img_path).convert("RGB")
        text = row["text"] + " " + row["title"]
        label = row["label"]

        return id, text, image, label
