from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np

from config import DEBUG_MODE

# --- IMAGES (PIL list) ---


def add_gaussian_noise(images, std=0.05):
    """Add Gaussian noise to a list of PIL images."""
    corrupted = []
    for img in images:
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1)
        noisy = Image.fromarray((np_img * 255).astype(np.uint8))
        corrupted.append(noisy)
    return corrupted


def blur(images, radius=1):
    """Apply a blur to a list of PIL images."""
    return [img.filter(ImageFilter.GaussianBlur(radius)) for img in images]


def brightness(images, factor=0.2):
    """Randomly vary the brightness."""
    corrupted = []
    for img in images:
        enhancer = ImageEnhance.Brightness(img)
        delta = 1 + ((random.random() * 2 - 1) * factor)
        corrupted.append(enhancer.enhance(delta))
    return corrupted


def flip_horizontal(images):
    """Flip horizontally."""
    return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]


def flip_vertical(images):
    """Flip vertically."""
    return [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]


def jpeg_compression(images, quality=50):
    """Simulate JPEG compression (down-up sampling)"""
    corrupted = []
    for img in images:
        # Convert to RGB just to be safe
        img = img.convert("RGB")
        # Salva temporaneamente in memoria (simula compressione)
        from io import BytesIO

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        corrupted.append(compressed)
    return corrupted


# Dizionario corruzioni
if DEBUG_MODE:
    IMAGE_CORRUPTIONS = {
        # "none": lambda x: x,
        "noise": add_gaussian_noise,
        "blur": blur,
        # "brightness": brightness,
        # "flip_h": flip_horizontal,
        # "flip_v": flip_vertical,
        # "jpeg": jpeg_compression,
    }
else:
    IMAGE_CORRUPTIONS = {
        # "none": lambda x: x,
        "noise": add_gaussian_noise,
        "blur": blur,
        "brightness": brightness,
        "flip_h": flip_horizontal,
        "flip_v": flip_vertical,
        "jpeg": jpeg_compression,
    }

# --- TESTO (stringhe) ---


# Drop random word
def drop_random_word(texts, p=0.15):
    """Drop each word with probability p"""
    perturbed_texts = []
    for text in texts:
        words = text.split()
        if len(words) <= 2:
            perturbed_texts.append(text)
            continue
        keep = [w for w in words if random.random() > p]
        perturbed_texts.append(" ".join(keep) if keep else text)
    return perturbed_texts


# Random case
def random_case(texts):
    """Randomly change the case of characters."""
    perturbed_texts = []
    for text in texts:
        perturbed_texts.append(
            "".join(c.upper() if random.random() > 0.5 else c.lower() for c in text)
        )
    return perturbed_texts


# Swap adjacent words
def swap_adjacent_words(texts):
    """Swap adjacent words in each text."""
    perturbed_texts = []
    for text in texts:
        words = text.split()
        if len(words) < 2:
            perturbed_texts.append(text)
            continue
        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
        perturbed_texts.append(" ".join(words))
    return perturbed_texts


# Repeat a random word
def repeat_word(texts):
    """Repeat a random word in each text."""
    perturbed_texts = []
    for text in texts:
        words = text.split()
        if not words:
            perturbed_texts.append(text)
            continue
        i = random.randint(0, len(words) - 1)
        words.insert(i, words[i])
        perturbed_texts.append(" ".join(words))
    return perturbed_texts


# Standard text perturbations
if DEBUG_MODE:
    TEXT_PERTURBATIONS = {
        # "none": lambda t: t,
        # "drop_word": drop_random_word,
        "case": random_case,
        # "swap": swap_adjacent_words,
        # "repeat": repeat_word,
    }
else:
    TEXT_PERTURBATIONS = {
        # "none": lambda t: t,
        "drop_word": drop_random_word,
        "case": random_case,
        "swap": swap_adjacent_words,
        "repeat": repeat_word,
    }
