"""
Utility funkcije za augmentaciju podataka (slike, bbox-ovi, optical flow).
"""

import random
import numpy as np
import torch
from torchvision import transforms
from typing import List, Tuple

from constants import (
    AUG_BRIGHTNESS_RANGE, AUG_CONTRAST_RANGE, AUG_SATURATION_RANGE,
    AUG_GRAYSCALE_PROB, AUG_TEMPORAL_JITTER_RANGE, AUG_ERASING_PROB,
    AUG_ERASING_SCALE, IMAGENET_MEAN, IMAGENET_STD, INV_255
)


class AugmentationParams:
    """
    Struktura za držanje augmentacijskih parametara koji se primjenjuju konzistentno kroz sekvencu.
    """
    def __init__(self, is_train: bool):
        self.is_train = is_train
        
        if is_train:
            # Uzorkuj parametre jednom za cijelu sekvencu
            self.brightness = 1.0 + random.uniform(*AUG_BRIGHTNESS_RANGE)
            self.contrast = 1.0 + random.uniform(*AUG_CONTRAST_RANGE)
            self.saturation = 1.0 + random.uniform(*AUG_SATURATION_RANGE)
            self.do_grayscale = random.random() < AUG_GRAYSCALE_PROB
            self.do_hflip = random.random() < 0.5
        else:
            # Val/test: bez augmentacije
            self.brightness = 1.0
            self.contrast = 1.0
            self.saturation = 1.0
            self.do_grayscale = False
            self.do_hflip = False


def apply_temporal_jitter(
    start_idx: int,
    n_frames: int,
    seq_len: int
) -> int:
    """
    Primjenjuje nasumični pomak prozora uzorkovanja (temporal jitter).
    
    Args:
        start_idx: Početni indeks sekvence
        n_frames: Ukupan broj frameova u videu
        seq_len: Dužina sekvence
        
    Returns:
        Novi start indeks s jitter-om
    """
    lo = -min(AUG_TEMPORAL_JITTER_RANGE, start_idx)
    hi = min(AUG_TEMPORAL_JITTER_RANGE, n_frames - seq_len - 1 - start_idx)
    
    if lo < hi:
        jitter = random.randint(lo, hi)
        return start_idx + jitter
        
    return start_idx


def apply_color_jitter(
    img_array: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    do_grayscale: bool
) -> np.ndarray:
    """
    Primjenjuje color jitter augmentaciju na numpy array sliku.
    
    Args:
        img_array: (3, H, W) float32 array u [0, 1]
        brightness: Brightness factor
        contrast: Contrast factor
        saturation: Saturation factor
        do_grayscale: Primijeniti grayscale konverziju
        
    Returns:
        Augmentirana slika (3, H, W) float32
    """
    # Brightness
    img_array = img_array * np.float32(brightness)
    
    # Contrast
    mean_val = img_array.mean()
    img_array = np.float32(contrast) * (img_array - mean_val) + mean_val
    
    # Saturation
    gray = img_array[0:1] * 0.299 + img_array[1:2] * 0.587 + img_array[2:3] * 0.114
    img_array = np.float32(saturation) * (img_array - gray) + gray
    
    # Grayscale
    if do_grayscale:
        img_array[:] = gray
        
    # Clip to valid range
    np.clip(img_array, 0.0, 1.0, out=img_array)
    
    return img_array


def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """
    Primjenjuje ImageNet normalizaciju na sliku.
    
    Args:
        img_array: (3, H, W) float32 array u [0, 1]
        
    Returns:
        Normalizirana slika (3, H, W) float32
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    
    return (img_array - mean) / std


def create_random_erasing_transform(prob: float = AUG_ERASING_PROB) -> transforms.RandomErasing:
    """
    Kreira RandomErasing transform za augmentaciju.
    
    Args:
        prob: Vjerojatnost primjene
        
    Returns:
        RandomErasing transform
    """
    return transforms.RandomErasing(p=prob, scale=AUG_ERASING_SCALE)


def apply_horizontal_flip_to_flow(flow: np.ndarray) -> np.ndarray:
    """
    Primjenjuje horizontalni flip na optical flow.
    
    Args:
        flow: (2, H, W) numpy array
        
    Returns:
        Flipped flow (2, H, W)
    """
    # Flip horizontalno
    flow = np.flip(flow, axis=2).copy()
    # Negate dx kanal
    flow[0] = -flow[0]
    
    return flow


def sample_temporal_mask_indices(
    seq_len: int,
    prob: float
) -> List[int]:
    """
    Uzorkuje indekse frameova koji će biti maskirani (zamijenjeni s prethodnim).
    
    Args:
        seq_len: Dužina sekvence
        prob: Vjerojatnost primjene temporal maskiranja
        
    Returns:
        Lista indeksa za maskiranje
    """
    if prob <= 0 or random.random() >= prob or seq_len <= 2:
        return []
        
    n_mask = random.randint(1, 2)
    # Ne maskiraj prvi ni zadnji frame
    candidates = list(range(1, seq_len - 1))
    
    if candidates:
        return random.sample(candidates, min(n_mask, len(candidates)))
        
    return []
