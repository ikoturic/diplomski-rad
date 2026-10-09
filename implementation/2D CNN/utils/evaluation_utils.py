"""
Utility funkcije za evaluaciju modela.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix,
)
from scipy.ndimage import median_filter as scipy_median_filter


# ========================================================================= #
#  Post-processing metode                                                     #
# ========================================================================= #

POSTPROCESS_CHOICES = [
    "none",
    "ma3", "ma5", "ma7", "ma9",
    "median3", "median5", "median7", "median11",
]


def apply_postprocess(
    video_probs: Dict[str, List[Tuple[int, float]]],
    method: str
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Primijeni post-processing na svaki video.
    
    Args:
        video_probs: Dict[video_name] -> [(frame_idx, prob)]
        method: Metoda post-processinga
        
    Returns:
        Procesiran video_probs dict
    """
    if method == "none":
        return video_probs

    result = {}
    for vn, frame_probs in video_probs.items():
        sorted_fp = sorted(frame_probs, key=lambda x: x[0])
        frames = [f[0] for f in sorted_fp]
        probs = np.array([f[1] for f in sorted_fp])

        if method.startswith("ma"):
            # Moving average
            w = int(method[2:])
            kernel = np.ones(w) / w
            probs = np.convolve(probs, kernel, mode='same')
        elif method.startswith("median"):
            # Median filter
            k = int(method[6:])
            probs = scipy_median_filter(probs, size=k)

        result[vn] = [(frames[i], float(probs[i])) for i in range(len(frames))]
        
    return result


def postprocess_label(method: str) -> str:
    """Čitljiv naziv post-processing metode za grafove."""
    if method == "none":
        return ""
    if method.startswith("ma"):
        return f" + Moving Average (w={method[2:]})"
    if method.startswith("median"):
        return f" + Median Filter (k={method[6:]})"
    return f" + {method}"


# ========================================================================= #
#  Metrike                                                                    #
# ========================================================================= #

def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Izračunava sve evaluacijske metrike.
    
    Args:
        probs: Predviđene vjerojatnosti
        labels: Ground truth labele
        threshold: Prag za binarizaciju
        
    Returns:
        Dict s metrikama
    """
    preds = (probs >= threshold).astype(float)
    
    metrics = {
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    
    # AUC metrike (mogu failati ako nema pozitivnih/negativnih primjera)
    try:
        metrics["auc_roc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc_roc"] = 0.0
        
    try:
        metrics["average_precision"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["average_precision"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }
    
    return metrics


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Pronalazi najbolji threshold po F1 metrice.
    
    Args:
        probs: Predviđene vjerojatnosti
        labels: Ground truth labele
        
    Returns:
        (best_threshold, best_f1)
    """
    best_f1, best_thr = 0.0, 0.5
    
    for thr in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= thr).astype(float)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            
    return float(best_thr), float(best_f1)


def threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray
) -> List[Dict[str, float]]:
    """
    Evaluira metriku za raspon threshold vrijednosti.
    
    Args:
        probs: Predviđene vjerojatnosti
        labels: Ground truth labele
        
    Returns:
        Lista dict-ova s metrikama za svaki threshold
    """
    results = []
    
    for thr in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= thr).astype(float)
        results.append({
            "threshold": round(float(thr), 2),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "accuracy": float(accuracy_score(labels, preds)),
        })
        
    return results


# ========================================================================= #
#  Video-level evaluacija                                                     #
# ========================================================================= #

def compute_video_level_confusion(
    video_probs: Dict[str, List[Tuple[int, float]]],
    all_metadata: Dict,
    threshold: float = 0.5,
    tolerance_frames: int = 10,
    before_frames: int = None,
    after_frames: int = None
) -> Dict:
    """
    Video-level confusion matrix.
    
    Raspon oko anomaly_start:
      - Ako before_frames i after_frames su None: koristi ±tolerance_frames (simetričan)
      - Inače: onset mora biti u [anomaly_start - before_frames, anomaly_start + after_frames]
      
    Args:
        video_probs: Dict[video_name] -> [(frame_idx, prob)]
        all_metadata: Metadata svih videa
        threshold: Prag za detekciju
        tolerance_frames: Tolerancija za simetrični raspon
        before_frames: Custom tolerancija prije anomaly_start
        after_frames: Custom tolerancija nakon anomaly_start
        
    Returns:
        Dict s video-level metrikama i TP/FP/FN listama
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    tp_videos, fp_videos, fn_videos = [], [], []

    for vn in sorted(video_probs.keys()):
        meta = all_metadata.get(vn)
        if meta is None:
            continue

        anomaly_start = meta.get("anomaly_start", -1)
        anomaly_end = meta.get("anomaly_end", -1)
        has_anomaly = anomaly_start >= 0 and anomaly_end >= 0

        # Nađi sve 0→1 onsets
        preds = sorted(video_probs[vn], key=lambda x: x[0])
        onsets = []
        prev = 0
        for frame_idx, prob in preds:
            curr = 1 if prob >= threshold else 0
            if prev == 0 and curr == 1:
                onsets.append(frame_idx)
            prev = curr

        if has_anomaly:
            # Provjeri je li ijedan onset u dozvoljenom rasponu
            if before_frames is not None and after_frames is not None:
                lo = anomaly_start - before_frames
                hi = anomaly_start + after_frames
                detected = any(lo <= o <= hi for o in onsets)
            else:
                detected = any(abs(o - anomaly_start) <= tolerance_frames for o in onsets)
                
            if detected:
                tp += 1
                tp_videos.append(vn)
            else:
                fn += 1
                fn_videos.append(vn)
        else:
            # Normalan video
            if len(onsets) == 0:
                tn += 1
            else:
                fp += 1
                fp_videos.append(vn)

    total = tp + fp + fn + tn
    result = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "total": total,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tolerance_frames": tolerance_frames,
        "tp_videos": tp_videos,
        "fp_videos": fp_videos,
        "fn_videos": fn_videos,
    }
    
    if before_frames is not None:
        result["before_frames"] = before_frames
        result["after_frames"] = after_frames
        
    return result
