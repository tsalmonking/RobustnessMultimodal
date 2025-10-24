import os
import torch
import argparse
from torch.utils.data import DataLoader

# Custom modules
from dataset import RecoveryDataset
from utils import multimodal_collate, load_model
from attacks import multimodal_attack
from config import NAME_LLM, NAME_IMG_EMBED, WEIGHTS_PATH, OUTPUT_DIR

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--attack-mode",
    choices=["black", "white"],
    default="black",
    help="Attack mode: 'black' (use standard corruptions) or 'white' (use gradient-based PGD).",
)
# white-box specific params (used only if attack-mode=white)
parser.add_argument("--pgd-iters", type=int, default=40)
parser.add_argument("--eps-img", type=float, default=8 / 255.0)
parser.add_argument("--alpha-img", type=float, default=8 / 40 * 1.25)
parser.add_argument("--eps-text", type=float, default=5.0)
parser.add_argument("--alpha-text", type=float, default=5 / 40 * 1.0)
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
BATCH_SIZE = 4


# Main evaluation loop
def main():
    # Prepare output directory and device
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    #subset = torch.utils.data.Subset(dataset, list(range(len(dataset) // 300)))
    # DataLoader for batching
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multimodal_collate
    )

    multimodal_attack(model, tokenizer, processor, "black", loader, device)

    multimodal_attack(
        model,
        tokenizer,
        processor,
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
