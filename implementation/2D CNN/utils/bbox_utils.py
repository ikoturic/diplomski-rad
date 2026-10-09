"""
Utility funkcije za obradu bounding box-ova i ekstrakciju značajki.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from constants import (
    BBOX_IDX_CX, BBOX_IDX_CY, BBOX_IDX_W, BBOX_IDX_H,
    BBOX_IDX_AREA, BBOX_IDX_ASPECT, BBOX_IDX_DX, BBOX_IDX_DY,
    BBOX_IDX_DW, BBOX_IDX_DH, BBOX_IDX_CONF, BBOX_IDX_APPROACH_RATE,
    BBOX_IDX_EXPANSION_RATE, BBOX_IDX_TTC, BBOX_IDX_FLOW_MAG,
    BBOX_IDX_FLOW_ANGLE, BBOX_IDX_FLOW_STD, MIN_NN_MATCHING_DISTANCE_SQ,
    MAX_TTC_VALUE, EPS_SMALL, EPS_MEDIUM, BBOX_FEATURE_DIM
)


def normalize_bbox_coords(
    bbox: List[float], 
    img_width: int, 
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Normalizira bbox koordinate na raspon [0, 1].
    
    Args:
        bbox: [x1, y1, x2, y2] u pixelima
        img_width: Originalna širina slike
        img_height: Originalna visina slike
        
    Returns:
        (x1n, y1n, x2n, y2n): Normalizirane koordinate u [0, 1]
    """
    x1, y1, x2, y2 = bbox
    inv_w = 1.0 / img_width
    inv_h = 1.0 / img_height
    
    x1n = max(0.0, min(1.0, x1 * inv_w))
    y1n = max(0.0, min(1.0, y1 * inv_h))
    x2n = max(0.0, min(1.0, x2 * inv_w))
    y2n = max(0.0, min(1.0, y2 * inv_h))
    
    return x1n, y1n, x2n, y2n


def compute_bbox_properties(
    x1n: float, 
    y1n: float, 
    x2n: float, 
    y2n: float
) -> Tuple[float, float, float, float, float, float]:
    """
    Izračunava osnovna svojstva bbox-a iz normaliziranih koordinata.
    
    Args:
        x1n, y1n, x2n, y2n: Normalizirane koordinate
        
    Returns:
        (cx, cy, w, h, area, aspect): Centar, dimenzije, površina, omjer
    """
    cx = (x1n + x2n) * 0.5
    cy = (y1n + y2n) * 0.5
    w = max(x2n - x1n, EPS_MEDIUM)
    h = max(y2n - y1n, EPS_MEDIUM)
    area = w * h
    aspect = w / h
    
    return cx, cy, w, h, area, aspect


def find_previous_object_match(
    cx: float,
    cy: float,
    category: int,
    prev_objects_list: List[Dict],
) -> Optional[Dict]:
    """
    Pronalazi najbolji match iz prethodnog framea koristeći NN matching po poziciji.
    
    Args:
        cx, cy: Centar trenutnog objekta
        category: Kategorija objekta
        prev_objects_list: Lista objekata iz prethodnog framea
        
    Returns:
        Najbolji match ili None ako nema dobrog matcha
    """
    if not prev_objects_list:
        return None
        
    best_d2 = MIN_NN_MATCHING_DISTANCE_SQ
    best_prev = None
    
    for prev in prev_objects_list:
        if prev["cat"] != category or prev["matched"]:
            continue
            
        d2 = (cx - prev["cx"]) ** 2 + (cy - prev["cy"]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_prev = prev
            
    if best_prev is not None:
        best_prev["matched"] = True
        
    return best_prev


def compute_velocity_features(
    cx: float,
    cy: float,
    w: float,
    h: float,
    track_id: int,
    category: int,
    prev_objects_by_id: Dict[int, Dict],
    prev_objects_list: List[Dict]
) -> Tuple[float, float, float, float]:
    """
    Izračunava velocity značajke (dx, dy, dw, dh) koristeći tracking ili NN matching.
    
    Args:
        cx, cy, w, h: Trenutne dimenzije objekta
        track_id: ID za tracking
        category: Kategorija objekta
        prev_objects_by_id: Mapa track_id -> objekt za brzo pretraživanje
        prev_objects_list: Lista objekata iz prethodnog framea za NN matching
        
    Returns:
        (dx, dy, dw, dh): Brzine pomaka i veličine
    """
    dx = dy = dw = dh = 0.0
    
    # Prvo pokušaj tracking po ID-u
    if track_id >= 0 and track_id in prev_objects_by_id:
        prev = prev_objects_by_id[track_id]
        dx = cx - prev["cx"]
        dy = cy - prev["cy"]
        dw = w - prev["w"]
        dh = h - prev["h"]
    else:
        # Fallback: NN matching po poziciji
        prev = find_previous_object_match(cx, cy, category, prev_objects_list)
        if prev is not None:
            dx = cx - prev["cx"]
            dy = cy - prev["cy"]
            dw = w - prev["w"]
            dh = h - prev["h"]
            
    return dx, dy, dw, dh


def compute_approach_rate(
    cx: float,
    cy: float,
    dx: float,
    dy: float
) -> float:
    """
    Izračunava brzinu približavanja centru slike.
    
    Negativna vrijednost = objekt se približava centru.
    
    Args:
        cx, cy: Normalizirani centar objekta
        dx, dy: Brzina pomaka
        
    Returns:
        Brzina približavanja
    """
    offset_x = cx - 0.5
    offset_y = cy - 0.5
    distance = max((offset_x**2 + offset_y**2)**0.5, EPS_MEDIUM)
    approach_rate = -(dx * offset_x + dy * offset_y) / distance
    
    return approach_rate


def compute_expansion_rate(
    w: float,
    h: float,
    dw: float,
    dh: float
) -> float:
    """
    Izračunava relativnu brzinu rasta bbox-a.
    
    Args:
        w, h: Trenutne dimenzije
        dw, dh: Promjene dimenzija
        
    Returns:
        Expansion rate
    """
    expansion_rate = dw / max(w, EPS_MEDIUM) + dh / max(h, EPS_MEDIUM)
    return expansion_rate


def compute_time_to_collision(
    area: float,
    expansion_rate: float
) -> float:
    """
    Izračunava Time-to-Collision proxy na temelju brzine rasta.
    
    Args:
        area: Površina bbox-a
        expansion_rate: Brzina ekspanzije
        
    Returns:
        TTC vrijednost (ograničena na MAX_TTC_VALUE)
    """
    if expansion_rate > EPS_LARGE:
        ttc = area / expansion_rate
    else:
        ttc = MAX_TTC_VALUE
        
    return min(ttc, MAX_TTC_VALUE)


def extract_bbox_features(
    bbox: List[float],
    track_id: int,
    category: int,
    confidence: float,
    img_width: int,
    img_height: int,
    prev_objects_by_id: Dict[int, Dict],
    prev_objects_list: List[Dict]
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Ekstrahira sve bbox značajke iz jednog objekta.
    
    Args:
        bbox: [x1, y1, x2, y2] koordinate u pixelima
        track_id: Tracking ID objekta
        category: Kategorija objekta
        confidence: Detection confidence
        img_width, img_height: Dimenzije originalne slike
        prev_objects_by_id: Mapa ID -> objekt iz prethodnog framea
        prev_objects_list: Lista objekata iz prethodnog framea
        
    Returns:
        (bbox_norm, features, curr_object): Normalizirani bbox, značajke, trenutni objekt
    """
    # Normaliziraj koordinate
    x1n, y1n, x2n, y2n = normalize_bbox_coords(bbox, img_width, img_height)
    
    # Osnovna svojstva
    cx, cy, w, h, area, aspect = compute_bbox_properties(x1n, y1n, x2n, y2n)
    
    # Velocity značajke
    dx, dy, dw, dh = compute_velocity_features(
        cx, cy, w, h, track_id, category, prev_objects_by_id, prev_objects_list
    )
    
    # Dinamičke značajke
    approach_rate = compute_approach_rate(cx, cy, dx, dy)
    expansion_rate = compute_expansion_rate(w, h, dw, dh)
    ttc = compute_time_to_collision(area, expansion_rate)
    
    # Bbox koordinate za ROI Align
    bbox_norm = np.array([x1n, y1n, x2n, y2n], dtype=np.float32)
    
    # Feature vektor (14 značajki + 3 za flow koje se popunjavaju kasnije)
    features = np.zeros(BBOX_FEATURE_DIM, dtype=np.float32)
    features[BBOX_IDX_CX] = cx
    features[BBOX_IDX_CY] = cy
    features[BBOX_IDX_W] = w
    features[BBOX_IDX_H] = h
    features[BBOX_IDX_AREA] = area
    features[BBOX_IDX_ASPECT] = aspect
    features[BBOX_IDX_DX] = dx
    features[BBOX_IDX_DY] = dy
    features[BBOX_IDX_DW] = dw
    features[BBOX_IDX_DH] = dh
    features[BBOX_IDX_CONF] = confidence
    features[BBOX_IDX_APPROACH_RATE] = approach_rate
    features[BBOX_IDX_EXPANSION_RATE] = expansion_rate
    features[BBOX_IDX_TTC] = ttc
    
    # Trenutni objekt za sljedeći frame
    curr_object = {
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "cat": category,
        "matched": False,
        "track_id": track_id,
    }
    
    return bbox_norm, features, curr_object


def apply_horizontal_flip_to_bbox(
    bbox_norm: np.ndarray,
    features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Primjenjuje horizontalni flip na bbox koordinate i značajke.
    
    Args:
        bbox_norm: Normalizirani bbox [x1, y1, x2, y2]
        features: Feature vektor
        
    Returns:
        (bbox_flipped, features_flipped): Flipped bbox i značajke
    """
    # Flip bbox koordinata
    x1_old = bbox_norm[0]
    x2_old = bbox_norm[2]
    bbox_norm[0] = 1.0 - x2_old
    bbox_norm[2] = 1.0 - x1_old
    
    # Flip cx i obrnuti horizontalni pomak
    features[BBOX_IDX_CX] = 1.0 - features[BBOX_IDX_CX]
    features[BBOX_IDX_DX] = -features[BBOX_IDX_DX]
    features[BBOX_IDX_APPROACH_RATE] = -features[BBOX_IDX_APPROACH_RATE]
    features[BBOX_IDX_FLOW_ANGLE] = -features[BBOX_IDX_FLOW_ANGLE]
    
    return bbox_norm, features
