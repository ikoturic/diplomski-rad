"""
Utility funkcije za obradu optical flow podataka.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple
from functools import lru_cache

from constants import FLOW_CACHE_MAX_SIZE, BBOX_IDX_FLOW_MAG, BBOX_IDX_FLOW_ANGLE, BBOX_IDX_FLOW_STD, EPS_SMALL


@lru_cache(maxsize=FLOW_CACHE_MAX_SIZE)
def _cached_load_flow(path: str) -> np.ndarray:
    """Učitaj optički tok s diska (keširano)."""
    return np.load(path).astype(np.float32)


def load_flow_for_frame(
    video_name: str,
    frame_idx: int,
    optical_flow_dir: str,
    is_first_frame: bool = False
) -> Optional[np.ndarray]:
    """
    Učitava optical flow za određeni frame.
    
    Args:
        video_name: Naziv videa
        frame_idx: Indeks framea
        optical_flow_dir: Direktorij s flow podacima
        is_first_frame: Je li ovo prvi frame (nema prethodnog)
        
    Returns:
        Flow array (2, H, W) ili None ako ne postoji ili je prvi frame
    """
    if is_first_frame:
        return None
        
    flow_path = os.path.join(
        optical_flow_dir, video_name, f"flow_{frame_idx - 1:06d}.npy"
    )
    
    if os.path.isfile(flow_path):
        return _cached_load_flow(flow_path)
        
    return None


def resize_flow_if_needed(
    flow: np.ndarray,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    Resize flow na kanonsku veličinu ako je potrebno.
    
    Args:
        flow: (2, H, W) numpy array
        target_h, target_w: Ciljna veličina
        
    Returns:
        Resized flow (2, target_h, target_w)
    """
    if flow.shape[1] != target_h or flow.shape[2] != target_w:
        return np.stack([
            cv2.resize(flow[0], (target_w, target_h), interpolation=cv2.INTER_AREA),
            cv2.resize(flow[1], (target_w, target_h), interpolation=cv2.INTER_AREA),
        ], axis=0)
        
    return flow


def subtract_ego_motion(flow: np.ndarray) -> np.ndarray:
    """
    Oduzima ego-motion (medijan flow) od flow mape.
    
    Args:
        flow: (2, H, W) numpy array
        
    Returns:
        Residual flow (2, H, W)
    """
    ego_dx = np.median(flow[0])
    ego_dy = np.median(flow[1])
    
    flow[0] -= ego_dx
    flow[1] -= ego_dy
    
    return flow


def extract_flow_statistics_for_bbox(
    flow_map: torch.Tensor,
    bbox_norm: np.ndarray,
    flow_h: int,
    flow_w: int
) -> Tuple[float, float, float]:
    """
    Ekstrahira statistike flow-a unutar bbox-a.
    
    Args:
        flow_map: (2, Hf, Wf) flow tensor
        bbox_norm: [x1, y1, x2, y2] normalizirani bbox
        flow_h, flow_w: Dimenzije flow mape
        
    Returns:
        (flow_mag, flow_angle, flow_std): Magnitude, angle, std deviation
    """
    # Bbox u flow pixel koordinate
    x1f = int(bbox_norm[0] * flow_w)
    y1f = int(bbox_norm[1] * flow_h)
    x2f = max(x1f + 1, int(bbox_norm[2] * flow_w))
    y2f = max(y1f + 1, int(bbox_norm[3] * flow_h))
    
    # Clamp
    x1f = max(0, min(x1f, flow_w - 1))
    y1f = max(0, min(y1f, flow_h - 1))
    x2f = max(x1f + 1, min(x2f, flow_w))
    y2f = max(y1f + 1, min(y2f, flow_h))
    
    # Crop flow
    crop = flow_map[:, y1f:y2f, x1f:x2f]
    dx_crop = crop[0]
    dy_crop = crop[1]
    
    # Magnitude
    mag = torch.sqrt(dx_crop**2 + dy_crop**2 + EPS_SMALL)
    flow_mag = mag.mean().item()
    
    # Angle (mean direction)
    flow_angle = torch.atan2(dy_crop.mean(), dx_crop.mean()).item()
    
    # Std deviation
    flow_std = mag.std().item() if mag.numel() > 1 else 0.0
    
    return flow_mag, flow_angle, flow_std


def add_flow_features_to_bbox(
    features: np.ndarray,
    flow_map: torch.Tensor,
    bbox_norm: np.ndarray,
    flow_h: int,
    flow_w: int
) -> np.ndarray:
    """
    Dodaje flow statistike u bbox feature vektor.
    
    Args:
        features: (17,) numpy array
        flow_map: (2, Hf, Wf) flow tensor
        bbox_norm: [x1, y1, x2, y2] normalizirani bbox
        flow_h, flow_w: Dimenzije flow mape
        
    Returns:
        Ažurirani features array
    """
    flow_mag, flow_angle, flow_std = extract_flow_statistics_for_bbox(
        flow_map, bbox_norm, flow_h, flow_w
    )
    
    features[BBOX_IDX_FLOW_MAG] = flow_mag
    features[BBOX_IDX_FLOW_ANGLE] = flow_angle
    features[BBOX_IDX_FLOW_STD] = flow_std
    
    return features


def extract_per_object_flow_patches(
    flow_map: np.ndarray,
    bboxes_norm: List[np.ndarray],
    obj_mask: List[bool],
    pool_size: int,
    max_objects: int
) -> np.ndarray:
    """
    Ekstrahira flow patches za svaki objekt.
    
    Args:
        flow_map: (2, Hf, Wf) numpy array
        bboxes_norm: Lista normaliziranih bbox-ova
        obj_mask: Lista bool maskova (True = stvarni objekt)
        pool_size: Veličina pooled patch-a
        max_objects: Maksimalan broj objekata
        
    Returns:
        (max_objects, 2, pool_size, pool_size) numpy array
    """
    flow_h, flow_w = flow_map.shape[1], flow_map.shape[2]
    obj_flow = np.zeros((max_objects, 2, pool_size, pool_size), dtype=np.float32)
    
    for j in range(len(bboxes_norm)):
        if not obj_mask[j]:
            continue
            
        bbox = bboxes_norm[j]
        
        # Bbox u flow pixel koordinate
        x1f = int(bbox[0] * flow_w)
        y1f = int(bbox[1] * flow_h)
        x2f = max(x1f + 1, int(bbox[2] * flow_w))
        y2f = max(y1f + 1, int(bbox[3] * flow_h))
        
        x1f = max(0, min(x1f, flow_w - 1))
        y1f = max(0, min(y1f, flow_h - 1))
        x2f = max(x1f + 1, min(x2f, flow_w))
        y2f = max(y1f + 1, min(y2f, flow_h))
        
        # Crop i resize
        crop_dx = flow_map[0, y1f:y2f, x1f:x2f]
        crop_dy = flow_map[1, y1f:y2f, x1f:x2f]
        
        obj_flow[j, 0] = cv2.resize(crop_dx, (pool_size, pool_size), interpolation=cv2.INTER_AREA)
        obj_flow[j, 1] = cv2.resize(crop_dy, (pool_size, pool_size), interpolation=cv2.INTER_AREA)
        
    return obj_flow


def process_flow_sequence(
    flow_dir: str,
    video_name: str,
    start_idx: int,
    seq_len: int,
    target_h: int,
    target_w: int,
    temporal_masked_indices: List[int]
) -> List[np.ndarray]:
    """
    Učitava i procesira sekvencu optical flow frameova.
    
    Args:
        flow_dir: Direktorij s flow podacima
        video_name: Naziv videa
        start_idx: Početni indeks
        seq_len: Dužina sekvence
        target_h, target_w: Ciljna veličina flow mape
        temporal_masked_indices: Indeksi frameova koji su maskirani
        
    Returns:
        Lista flow numpy arrays (svaki (2, target_h, target_w))
    """
    flow_shape = (2, target_h, target_w)
    all_flows = []
    
    for t in range(start_idx, start_idx + seq_len):
        if t == start_idx:
            # Prvi frame: nema prethodnog → nula flow
            flow = None
        else:
            flow = load_flow_for_frame(video_name, t, flow_dir)
            
        all_flows.append(flow)
        
    # Zamijeni None i maskirane s nulama; resize na kanonsku veličinu
    for i in range(len(all_flows)):
        if all_flows[i] is None or i in temporal_masked_indices:
            all_flows[i] = np.zeros(flow_shape, dtype=np.float32)
        else:
            all_flows[i] = resize_flow_if_needed(all_flows[i], target_h, target_w)
            
    # Oduzmi ego-motion
    for i in range(len(all_flows)):
        if not (i in temporal_masked_indices or np.all(all_flows[i] == 0)):
            all_flows[i] = subtract_ego_motion(all_flows[i])
            
    return all_flows
