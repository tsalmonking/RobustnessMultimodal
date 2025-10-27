import os
import torch
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Custom modules
from dataset import RecoveryDataset
from utils import multimodal_collate, load_model
from attacks import multimodal_attack
from config import NAME_LLM, NAME_IMG_EMBED, WEIGHTS_PATH, OUTPUT_DIR, DEBUG_MODE

# PGD default parameters
if DEBUG_MODE:
    ITERS = 2
else:
    ITERS = 80
EPS_IMG = 4 / 255.0
ALPHA_IMG = EPS_IMG / (ITERS * 1.25)
EPS_TEXT = 5.0
ALPHA_TEXT = EPS_TEXT / (ITERS * 1.0)

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--attack-mode",
    choices=["black", "white"],
    default="black",
    help="Attack mode: 'black' (use standard corruptions) or 'white' (use gradient-based PGD).",
)
# white-box specific params (used only if attack-mode=white)
parser.add_argument("--pgd-iters", type=int, default=ITERS)
parser.add_argument("--eps-img", type=float, default=EPS_IMG)
parser.add_argument("--alpha-img", type=float, default=ALPHA_IMG)
parser.add_argument("--eps-text", type=float, default=EPS_TEXT)
parser.add_argument("--alpha-text", type=float, default=ALPHA_TEXT)
parser.add_argument(
    "--text-perturbation",
    choices=["true", "false"],
    default="false",
    help="Whether to perturb text embeddings in white-box attack. Warning: the text embeddings perturbation could create incoherent texts.",
)

args = parser.parse_args()

DATASET_PATH = "Data/ReCOVery"
DATA_CSV = os.path.join(DATASET_PATH, "recovery.csv")
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
if DEBUG_MODE:
    BATCH_SIZE = 2
else:
    BATCH_SIZE = 16


# Main evaluation loop
def main():
    # Prepare output directory and device
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator = Accelerator()
    device = accelerator.device
    
    # Loading the model
    model, tokenizer, processor = load_model(
        device,
        weights_path=WEIGHTS_PATH,
        name_llm=NAME_LLM,
        name_img_embed=NAME_IMG_EMBED,
    )

    # The dataset with images and texts
    dataset = RecoveryDataset(csv_file=DATA_CSV, image_dir=IMAGES_DIR)

    # for debug loader gets smaller
    subset = torch.utils.data.Subset(dataset, list(range(len(dataset) // 100)))

    if DEBUG_MODE:
        loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multimodal_collate
    )
    else:
        # DataLoader for batching
        loader = DataLoader(
            subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multimodal_collate
        )

    # Preparing for processing optimizating GPU use
    model, loader = accelerator.prepare(model, loader)

    multimodal_attack(model, tokenizer, processor, accelerator, "black", loader, device)

    multimodal_attack(
        model,
        tokenizer,
        processor,
        accelerator,
        "white",
        loader,
        device,
        args.pgd_iters,
        args.eps_img,
        args.alpha_img,
        args.eps_text,
        args.alpha_text,
    )

    multimodal_attack(
        model,
        tokenizer,
        processor,
        accelerator,
        "white",
        loader,
        device,
        args.pgd_iters,
        args.eps_img,
        args.alpha_img,
        args.eps_text,
        args.alpha_text,
        "true",
    )

    # A ring alarm to signal the end of the process
    # winsound.Beep(1000, 1000)


if __name__ == "__main__":
    main()
