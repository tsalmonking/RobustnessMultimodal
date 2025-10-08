import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch


# recovery dataset class well commented
class RecoveryDataset(Dataset):
    """A PyTorch Dataset class for loading images and corresponding text data for recovery tasks.
    Each sample consists of an image, a text input, and a label.
    Args:
        csv_file (str): Path to the CSV file containing image IDs, text data, and labels.
        image_dir (str): Directory where images are stored.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text data.
        processor (transformers.PreTrainedProcessor): Processor for handling image data.
        max_length (int): Maximum length for text tokenization. Default is 64.
    """

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
        image_inputs["pixel_values"] = image_inputs["pixel_values"].unsqueeze(0)

        text_inputs = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        label = torch.tensor(row["label"], dtype=torch.float)

        # Squeeze to remove the batch dimension added by the processor and tokenizer
        return (
            image_inputs["pixel_values"].squeeze(0),
            label,
            text_inputs["input_ids"].squeeze(0),
        )
