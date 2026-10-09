"""
Evaluacija 3D CNN modela na test/val setu.

Pokretanje:
    cd "3D CNN"
    python evaluate_3d.py --checkpoint checkpoints_3d/v1/best_model.pth --split test
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import yaml
import numpy as np
import torch
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_3d import CNN3D_Danger
from dataloader import create_test_dataloader


# ========================================================================= #
#  Per-sample evaluacija                                                      #
# ========================================================================= #

@torch.inference_mode()
def evaluate_model(model, loader, device):
    """Evaluira model na cijelom loaderu, vraća predikcije i oznake."""
    model.eval()
    all_probs_gpu = []
    all_labels_gpu = []
    all_video_names = []

    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for batch in tqdm(loader, desc="Evaluacija"):
        assert "frames" in batch, "3D CNN model zahtijeva raw frames!"
        frames = batch["frames"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"]
        video_names = batch["video_name"]
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(
                frames, bbox_features=bbox_features,
                categories=categories, obj_mask=obj_mask, flow=flow,
            )

        all_probs_gpu.append(torch.sigmoid(logits))
        all_labels_gpu.append((target >= 0.5).float())
        all_video_names.extend(video_names)

    all_probs = torch.cat(all_probs_gpu).float().cpu().numpy()
    all_labels = torch.cat(all_labels_gpu).numpy()
    return all_probs, all_labels, all_video_names


# ========================================================================= #
#  Compute metrike                                                            #
# ========================================================================= #

def compute_metrics(probs, labels, threshold=0.5):
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
    try:
        metrics["auc_roc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc_roc"] = 0.0
    try:
        metrics["average_precision"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["average_precision"] = 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }
    return metrics


def find_best_threshold(probs, labels):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= thr).astype(float)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr), float(best_f1)


def per_video_analysis(probs, labels, video_names):
    vid_data = defaultdict(lambda: {"probs": [], "labels": []})
    for p, l, v in zip(probs, labels, video_names):
        vid_data[v]["probs"].append(p)
        vid_data[v]["labels"].append(l)
    results = []
    for vid, data in vid_data.items():
        results.append({
            "video": vid,
            "n_samples": len(data["probs"]),
            "n_positive": sum(1 for l in data["labels"] if l >= 0.5),
            "has_anomaly": int(any(l >= 0.5 for l in data["labels"])),
            "avg_prob": float(np.mean(data["probs"])),
            "max_prob": float(np.max(data["probs"])),
        })
    return sorted(results, key=lambda x: x["max_prob"], reverse=True)


def threshold_sweep(probs, labels):
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
#  Grafovi                                                                    #
# ========================================================================= #

def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Krivulja (3D CNN)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, "r-", lw=2, label=f"AP = {ap:.4f}")
    baseline = labels.sum() / len(labels)
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.4, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Krivulja (3D CNN)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm_dict, save_path, threshold=0.5):
    cm = np.array([[cm_dict["TN"], cm_dict["FP"]],
                    [cm_dict["FN"], cm_dict["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normalno", "Opasno"])
    ax.set_yticklabels(["Normalno", "Opasno"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (thr={threshold:.2f})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ========================================================================= #
#  Video-level confusion matrix                                               #
# ========================================================================= #

def analyze_delays(video_probs, all_metadata, threshold=0.5):
    """Delay analiza — referentna točka = anomaly_start."""
    delays = []
    missed_events = 0
    all_onsets_relative = []  # svi onseti (za all-onsets histogram)
    total_anomaly_videos = 0

    for vn in sorted(video_probs.keys()):
        meta = all_metadata.get(vn)
        if meta is None:
            continue
        anomaly_start = meta.get("anomaly_start", -1)
        anomaly_end   = meta.get("anomaly_end", -1)
        if anomaly_start < 0 or anomaly_end < 0:
            continue

        total_anomaly_videos += 1
        preds = sorted(video_probs[vn], key=lambda x: x[0])

        onsets = []
        prev = 0
        for frame_idx, prob in preds:
            curr = 1 if prob >= threshold else 0
            if prev == 0 and curr == 1:
                onsets.append(frame_idx)
            prev = curr

        if not onsets:
            missed_events += 1
            continue

        best = min(onsets, key=lambda x: abs(x - anomaly_start))
        delays.append(best - anomaly_start)
        for o in onsets:
            all_onsets_relative.append(o - anomaly_start)

    return {
        "delays": np.array(delays),
        "all_onsets_relative": np.array(all_onsets_relative),
        "missed_events": missed_events,
        "total_anomaly_videos": total_anomaly_videos,
    }


def plot_delay_histogram(delays, save_path):
    """Histogram samo najboljeg onseta po videu."""
    if len(delays) == 0:
        return
    min_d = int(np.floor(delays.min()))
    max_d = int(np.ceil(delays.max()))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delays, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="anomaly_start")
    ax.axvline(delays.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean={delays.mean():.1f}f")
    ax.set_xlabel("Delay (frameovi, neg=rano, poz=kasno)")
    ax.set_ylabel("Broj videa")
    ax.set_title("Kašnjenje detekcije — najbolji onset po videu (3D CNN)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_delay_histogram_all(all_onsets_relative, save_path):
    """Histogram svih onseta (relativno na anomaly_start)."""
    if len(all_onsets_relative) == 0:
        return
    min_d = int(np.floor(all_onsets_relative.min()))
    max_d = int(np.ceil(all_onsets_relative.max()))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_onsets_relative, bins=bins, color="mediumseagreen", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="anomaly_start")
    ax.axvline(all_onsets_relative.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean={all_onsets_relative.mean():.1f}f")
    ax.set_xlabel("Onset - anomaly_start (frameovi)")
    ax.set_ylabel("Broj onseta")
    ax.set_title(f"Svi onseti detekcije ({len(all_onsets_relative)} ukupno, 3D CNN)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_video_level_confusion(video_probs, all_metadata, threshold=0.5,
                                   tolerance_frames=10, before_frames=None, after_frames=None):
    """
    Video-level confusion matrix.
    - Simetričan: onset u [anomaly_start - tolerance_frames, anomaly_start + tolerance_frames]
    - Asimetričan: onset u [anomaly_start - before_frames, anomaly_start + after_frames]
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    tp_videos, fp_videos, fn_videos = [], [], []

    for vn in sorted(video_probs.keys()):
        meta = all_metadata.get(vn)
        if meta is None:
            continue

        anomaly_start = meta.get("anomaly_start", -1)
        anomaly_end   = meta.get("anomaly_end", -1)
        has_anomaly   = anomaly_start >= 0 and anomaly_end >= 0

        preds = sorted(video_probs[vn], key=lambda x: x[0])
        onsets = []
        prev = 0
        for frame_idx, prob in preds:
            curr = 1 if prob >= threshold else 0
            if prev == 0 and curr == 1:
                onsets.append(frame_idx)
            prev = curr

        if has_anomaly:
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
            if len(onsets) == 0:
                tn += 1
            else:
                fp += 1
                fp_videos.append(vn)

    total = tp + fp + fn + tn
    result = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "total": total,
        "accuracy":  (tp + tn) / max(total, 1),
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
        "f1":        2 * tp / max(2 * tp + fp + fn, 1),
        "tolerance_frames": tolerance_frames,
        "tp_videos": tp_videos,
        "fp_videos": fp_videos,
        "fn_videos": fn_videos,
    }
    if before_frames is not None:
        result["before_frames"] = before_frames
        result["after_frames"] = after_frames
    return result


def plot_video_confusion_matrix(vcm, save_path, title=None):
    cm = np.array([[vcm["TN"], vcm["FP"]],
                   [vcm["FN"], vcm["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normalno", "Opasno"])
    ax.set_yticklabels(["Normalno", "Opasno"])
    ax.set_xlabel("Model predikcija (onset)")
    ax.set_ylabel("GT (video)")
    if title is None:
        tol_sec = vcm["tolerance_frames"] / 10
        title = f"Video-level CM (3D CNN, \u00b1{tol_sec:.1f}s tolerancija)"
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ========================================================================= #
#  Main                                                                       #
# ========================================================================= #

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D CNN danger prediction model")
    parser.add_argument("--config", default="config_3d.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--eval_stride", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    # Auto-detect config iz checkpoint source/ direktorija
    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_config = os.path.join(ckpt_dir, "source", "config_3d.yaml")
    config_path = ckpt_config if os.path.exists(ckpt_config) else args.config
    print(f"Config: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.eval_stride is not None:
        print(f"Override stride: {config.get('stride', 1)} -> {args.eval_stride}")
        config["stride"] = args.eval_stride

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uređaj: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    # --- Model ---
    model = CNN3D_Danger(
        backbone_3d=config.get("backbone_3d", "r2plus1d_18"),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.5),
        freeze_3d=config.get("freeze_3d", False),
        freeze_early=config.get("freeze_early", True),
        use_grad_checkpoint=False,  # Nepotrebno za evaluaciju
        max_objects=config.get("max_objects", 15),
        bbox_feat_dim=config.get("bbox_feat_dim", 64),
        category_embed_dim=config.get("category_embed_dim", 16),
        use_object_branch=config.get("use_object_branch", True),
        use_object_interaction=config.get("use_object_interaction", False),
        object_interaction_heads=config.get("object_interaction_heads", 4),
        object_interaction_layers=config.get("object_interaction_layers", 1),
        obj_temporal_layers=config.get("obj_temporal_layers", 1),
        obj_temporal_dropout=config.get("obj_temporal_dropout", 0.0),
        attention_heads=config.get("attention_heads", 4),
        use_causal_attention=config.get("use_causal_attention", True),
        use_flow=config.get("use_global_flow", config.get("use_optical_flow", False)),
        flow_feat_dim=config.get("flow_feat_dim", 128),
        freeze_layer3=config.get("freeze_layer3", False),
        use_temporal_aux=config.get("use_temporal_aux", False),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"Checkpoint: {args.checkpoint}  (epoch {epoch})")

    # --- Data ---
    assert config.get("cnn_cache_dir") is None, "3D CNN model ne podržava CNN cache!"
    eval_bs = config.get("val_batch_size", 16)
    config["batch_size"] = eval_bs
    config["num_workers"] = 4
    config["persistent_workers"] = True
    config["prefetch_factor"] = 2
    loader = create_test_dataloader(config, split=args.split)
    print(f"Evaluacija na {args.split} setu ({len(loader.dataset)} uzoraka, bs={eval_bs})")

    # --- Evaluate ---
    probs, labels, video_names = evaluate_model(model, loader, device)

    # --- Metrics ---
    metrics = compute_metrics(probs, labels, threshold=0.5)
    best_thr, best_f1 = find_best_threshold(probs, labels)
    metrics_best_thr = compute_metrics(probs, labels, threshold=best_thr)

    print(f"\n{'='*60}")
    print(f"REZULTATI ({args.split} set) — 3D CNN Model")
    print(f"{'='*60}")
    print(f"  Ukupno uzoraka:  {metrics['n_samples']}")
    print(f"  Pozitivni:       {metrics['n_positive']} ({metrics['n_positive']/max(metrics['n_samples'],1)*100:.1f}%)")
    print(f"  Negativni:       {metrics['n_negative']} ({metrics['n_negative']/max(metrics['n_samples'],1)*100:.1f}%)")
    print(f"\n  AUC-ROC:         {metrics['auc_roc']:.4f}")
    print(f"  Average Prec:    {metrics['average_precision']:.4f}")
    print(f"\n  --- Threshold = 0.5 ---")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  TP={cm['TP']}  FP={cm['FP']}  FN={cm['FN']}  TN={cm['TN']}")
    print(f"\n  --- Najbolji threshold = {best_thr:.2f} ---")
    print(f"  F1:         {metrics_best_thr['f1']:.4f}")
    print(f"  Precision:  {metrics_best_thr['precision']:.4f}")
    print(f"  Recall:     {metrics_best_thr['recall']:.4f}")

    # --- Save ---
    save_dir = args.save_dir or os.path.dirname(args.checkpoint)
    os.makedirs(save_dir, exist_ok=True)

    thr_sweep = threshold_sweep(probs, labels)
    vid_analysis = per_video_analysis(probs, labels, video_names)

    results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "model": "CNN3D_Danger",
        "backbone_3d": config.get("backbone_3d", "r2plus1d_18"),
        "metrics_thr_0.5": metrics,
        "best_threshold": best_thr,
        "metrics_best_thr": metrics_best_thr,
        "threshold_sweep": thr_sweep,
        "per_video_top20": vid_analysis[:20],
        "total_videos": len(set(video_names)),
    }
    json_path = os.path.join(save_dir, f"test_results_3d_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON: {json_path}")

    try:
        plot_roc_curve(labels, probs, os.path.join(save_dir, "roc_curve_3d.png"))
        plot_pr_curve(labels, probs, os.path.join(save_dir, "pr_curve_3d.png"))
        plot_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix_3d_050.png"), 0.5)
        plot_confusion_matrix(
            metrics_best_thr["confusion_matrix"],
            os.path.join(save_dir, f"confusion_matrix_3d_{best_thr:.2f}.png"),
            best_thr,
        )
        print(f"  Grafovi spremljeni u: {save_dir}/")
    except Exception as e:
        print(f"  UPOZORENJE: Plot greška: {e}")

    # --- Metadata + video_probs ---
    metadata_dir = config.get("metadata_dir", "../../dataset")
    meta_json = os.path.join(
        os.path.dirname(os.path.abspath(config_path)), metadata_dir,
        f"metadata_{args.split}.json",
    )
    all_metadata = {}
    if os.path.exists(meta_json):
        with open(meta_json, encoding="utf-8") as f:
            all_metadata = json.load(f)
        print(f"  Metadata: {meta_json} ({len(all_metadata)} videa)")
    else:
        print(f"  UPOZORENJE: Metadata nije pronađena na {meta_json}")

    if all_metadata:
        seq_len = config.get("seq_len", 24)
        video_probs = defaultdict(list)
        video_sample_count = defaultdict(int)
        for prob, vn in zip(probs, video_names):
            fidx = seq_len + video_sample_count[vn]
            video_sample_count[vn] += 1
            video_probs[vn].append((fidx, float(prob)))

        # --- Delay analiza ---
        delay_info = analyze_delays(video_probs, all_metadata, threshold=args.threshold)
        delays     = delay_info["delays"]
        all_onsets = delay_info["all_onsets_relative"]
        print(f"\n{'='*60}")
        print(f"DELAY ANALIZA (threshold={args.threshold:.2f})")
        print(f"{'='*60}")
        print(f"  Ukupno anomaly videa:  {delay_info['total_anomaly_videos']}")
        print(f"  Propušteni eventi:     {delay_info['missed_events']}")
        if len(delays) > 0:
            print(f"  Mean delay (best):     {delays.mean():.2f}f  "
                  f"median={float(np.median(delays)):.1f}f")
            print(f"  % rano (delay < 0):    {(delays < 0).mean()*100:.1f}%")
            print(f"  % na vrijeme (<=0):    {(delays <= 0).mean()*100:.1f}%")
        try:
            plot_delay_histogram(delays, os.path.join(save_dir, "delay_histogram.png"))
            plot_delay_histogram_all(all_onsets, os.path.join(save_dir, "delay_histogram_all_onsets.png"))
            print(f"  Histogrami: delay_histogram.png, delay_histogram_all_onsets.png")
        except Exception as e:
            print(f"  UPOZORENJE: Delay histogram greška: {e}")

        # --- Video-level confusion matrix ---
        vcm_configs = [
            {"label": "±1s",       "file": "video_cm_pm1s.png",
             "kwargs": {"tolerance_frames": 10}},
            {"label": "±2s",       "file": "video_cm_pm2s.png",
             "kwargs": {"tolerance_frames": 20}},
            {"label": "-2s..GT",   "file": "video_cm_before2s.png",
             "kwargs": {"before_frames": 20, "after_frames": 0}},
        ]
        print(f"\n{'='*60}")
        print(f"VIDEO-LEVEL CONFUSION MATRIX (threshold={args.threshold:.2f})")
        print(f"{'='*60}")
        vcm_results = {}
        for vcfg in vcm_configs:
            vcm = compute_video_level_confusion(
                video_probs, all_metadata, threshold=args.threshold, **vcfg["kwargs"]
            )
            print(f"  {vcfg['label']:10s}: "
                  f"TP={vcm['TP']} FP={vcm['FP']} FN={vcm['FN']} TN={vcm['TN']} "
                  f"| P={vcm['precision']:.3f} R={vcm['recall']:.3f} "
                  f"F1={vcm['f1']:.3f} Acc={vcm['accuracy']:.3f}")
            vcm_results[vcfg["label"]] = {
                k: v for k, v in vcm.items()
                if k not in ("tp_videos", "fp_videos", "fn_videos")
            }
            try:
                title = f"Video-level CM (3D CNN, {vcfg['label']})"
                plot_video_confusion_matrix(
                    vcm, os.path.join(save_dir, vcfg["file"]), title=title,
                )
            except Exception as e:
                print(f"  UPOZORENJE: VCM plot greška ({vcfg['label']}): {e}")

        results["delay_analysis"] = {
            "total_anomaly_videos": delay_info["total_anomaly_videos"],
            "missed_events": delay_info["missed_events"],
            "mean_delay_best": float(delays.mean()) if len(delays) > 0 else None,
            "median_delay_best": float(np.median(delays)) if len(delays) > 0 else None,
        }
        results["video_level_confusion"] = vcm_results
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  VCM slike: video_cm_pm1s.png, video_cm_pm2s.png, video_cm_before2s.png")

    print(f"\nEvaluacija završena!")


if __name__ == "__main__":
    main()
