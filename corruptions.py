import torch
import torch.nn.functional as F
import random

# --- IMAGES (Torch tensor) ---


# Add image corruptions and text perturbations
def add_gaussian_noise(x, std=0.05):
    """x: (B, C, H, W)"""
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0, 1)


# Blur
def blur(x, kernel_size=3):
    """Applica un blur medio semplice"""
    padding = kernel_size // 2
    weight = torch.ones((x.shape[1], 1, kernel_size, kernel_size), device=x.device)
    weight /= kernel_size**2
    return F.conv2d(x, weight, padding=padding, groups=x.shape[1])


# Brightness adjustment
def brightness(x, factor=0.2):
    """Aumenta o riduce la luminosità"""
    delta = (torch.rand(1).item() * 2 - 1) * factor
    return torch.clamp(x + delta, 0, 1)


# Horizontal
def flip_horizontal(x):
    return torch.flip(x, dims=[3])


# Vertical flip
def flip_vertical(x):
    return torch.flip(x, dims=[2])


# JPEG compression simulation
def jpeg_compression(x, quality=50):
    """Simula una compressione JPEG grezza"""
    # downsample + upsample
    h, w = x.shape[2:]
    scale = max(1, int(100 / quality))
    small = F.interpolate(x, size=(h // scale, w // scale), mode="bilinear")
    return F.interpolate(small, size=(h, w), mode="bilinear")


# Standard image corruptions
STANDARD_CORRUPTIONS = {
    "none": lambda x: x,
    "noise": add_gaussian_noise,
    "blur": blur,
    "brightness": brightness,
    "flip_h": flip_horizontal,
    "flip_v": flip_vertical,
    "jpeg": jpeg_compression,
}

# --- TESTO (stringhe) ---


# Drop random word
def drop_random_word(text, p=0.15):
    words = text.split()
    if len(words) <= 2:
        return text
    keep = [w for w in words if random.random() > p]
    return " ".join(keep) if keep else text


# Random case
def random_case(text):
    return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in text)


# Swap adjacent words
def swap_adjacent_words(text):
    words = text.split()
    if len(words) < 2:
        return text
    i = random.randint(0, len(words) - 2)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


# Repeat a random word
def repeat_word(text):
    words = text.split()
    if not words:
        return text
    i = random.randint(0, len(words) - 1)
    words.insert(i, words[i])
    return " ".join(words)


# Standard text perturbations
TEXT_PERTURBATIONS = {
    "none": lambda t: t,
    "drop_word": drop_random_word,
    "case": random_case,
    "swap": swap_adjacent_words,
    "repeat": repeat_word,
}
