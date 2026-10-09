"""
Evaluacija bez rampe — čisto binarni GT labeli.

GT = 1 za frameove u [anomaly_start, anomaly_end], inače GT = 0.
Delay analiza mjeri kašnjenje od anomaly_start (ne od 50% rampe).

Pokretanje:
    python evaluate.py --checkpoint checkpoints/v25/best_model.pth
    python evaluate.py --checkpoint checkpoints/v25/best_model.pth --threshold 0.4
"""

import os
import json
import pickle
import argparse
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
from tqdm import tqdm

from model import CNN_LSTM_BBox
from dataloader import create_test_dataloader
from utils import (
    apply_postprocess, postprocess_label, compute_metrics,
    find_best_threshold, threshold_sweep, compute_video_level_confusion,
    POSTPROCESS_CHOICES
)


def run_inference(model, loader, device, seq_len):
    """Pokreni inferenciju i grupiraj rezultate po videu s frame indeksima."""
    model.eval()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    video_probs = defaultdict(list)
    video_sample_count = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            frames = batch["frames"].to(device, non_blocking=True) if "frames" in batch else None
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            bbox_features = batch["bbox_features"].to(device, non_blocking=True)
            categories = batch["categories"].to(device, non_blocking=True)
            obj_mask = batch["obj_mask"].to(device, non_blocking=True)
            vid_names = batch["video_name"]
            flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
            cached_fm = batch["feat_maps"].to(device, non_blocking=True) if "feat_maps" in batch else None
            cached_gf = batch["global_feats"].to(device, non_blocking=True) if "feat_maps" in batch else None

            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(frames, bboxes, bbox_features, categories, obj_mask, flow=flow,
                               cached_feat_maps=cached_fm, cached_global_feats=cached_gf)
            probs = torch.sigmoid(logits).float().cpu().numpy()

            for i in range(len(vid_names)):
                vn = vid_names[i]
                frame_idx = seq_len + video_sample_count[vn]
                video_sample_count[vn] += 1
                video_probs[vn].append((frame_idx, float(probs[i])))

    return video_probs


def build_binary_labels(video_probs, all_metadata):
    """Izgradi binarni GT (bez rampe) za svaki uzorak koristeći metadata."""
    all_probs = []
    all_labels = []
    all_video_names = []

    for vn in sorted(video_probs.keys()):
        meta = all_metadata.get(vn)
        if meta is None:
            continue
        anomaly_start = meta.get("anomaly_start", -1)
        anomaly_end = meta.get("anomaly_end", -1)

        preds = sorted(video_probs[vn], key=lambda x: x[0])
        for frame_idx, prob in preds:
            all_probs.append(prob)
            # Binarni GT: 1 ako je frame u [anomaly_start, anomaly_end], inače 0
            if anomaly_start >= 0 and anomaly_end >= 0 and anomaly_start <= frame_idx <= anomaly_end:
                all_labels.append(1.0)
            else:
                all_labels.append(0.0)
            all_video_names.append(vn)

    return np.array(all_probs), np.array(all_labels), all_video_names


# ========================================================================= #
#  Delay analiza (referentna točka = anomaly_start)                          #
# ========================================================================= #

def analyze_delays_no_ramp(video_probs, all_metadata, threshold=0.5):
    """Delay analiza s anomaly_start kao referentnom točkom."""
    delays = []
    missed_events = 0
    false_alarm_distances = []
    total_anomaly_videos = 0

    for vn in sorted(video_probs.keys()):
        meta = all_metadata.get(vn)
        if meta is None:
            continue
        anomaly_start = meta.get("anomaly_start", -1)
        anomaly_end = meta.get("anomaly_end", -1)
        if anomaly_start < 0 or anomaly_end < 0:
            continue

        total_anomaly_videos += 1

        # Referentna točka = anomaly_start (bez rampe)
        gt_ref = float(anomaly_start)

        preds = sorted(video_probs[vn], key=lambda x: x[0])

        # Nađi sve 0→1 tranzicije
        model_onsets = []
        prev_pred = 0
        for frame_idx, prob in preds:
            curr_pred = 1 if prob >= threshold else 0
            if prev_pred == 0 and curr_pred == 1:
                model_onsets.append(frame_idx)
            prev_pred = curr_pred

        if not model_onsets:
            missed_events += 1
            continue

        best_onset = min(model_onsets, key=lambda x: abs(x - gt_ref))
        delay = best_onset - gt_ref
        delays.append(delay)

        for onset in model_onsets:
            if onset != best_onset:
                false_alarm_distances.append(onset - gt_ref)

    return {
        "delays": np.array(delays),
        "missed_events": missed_events,
        "false_alarm_distances": false_alarm_distances,
        "total_anomaly_videos": total_anomaly_videos,
    }


# ========================================================================= #
#  Grafovi                                                                    #
# ========================================================================= #

def plot_roc_curve(labels, probs, save_path, title_extra=""):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Slučajni klasifikator")
    ax.set_xlabel("Stopa lažno pozitivnih")
    ax.set_ylabel("Stopa istinski pozitivnih")
    ax.set_title(f"ROC krivulja{title_extra}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_pr_curve(labels, probs, save_path, title_extra=""):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, "r-", lw=2, label=f"AP = {ap:.4f}")
    baseline = labels.sum() / len(labels)
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.4, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall krivulja{title_extra}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_score_distribution(labels, probs, save_path, title_extra=""):
    pos_probs = probs[labels >= 0.5]
    neg_probs = probs[labels < 0.5]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(neg_probs, bins=50, alpha=0.6, color="green", label="Normalno", density=True)
    ax.hist(pos_probs, bins=50, alpha=0.6, color="red", label="Opasno", density=True)
    ax.set_xlabel("Predviđena vjerojatnost"); ax.set_ylabel("Gustoća")
    ax.set_title(f"Distribucija izlaznih ocjena{title_extra}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_confusion_matrix(cm_dict, save_path, threshold=0.5, title_extra=""):
    cm = np.array([[cm_dict["TN"], cm_dict["FP"]],
                   [cm_dict["FN"], cm_dict["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normalno", "Opasno"])
    ax.set_yticklabels(["Normalno", "Opasno"])
    ax.set_xlabel("Predviđeno"); ax.set_ylabel("Stvarno")
    ax.set_title(f"Matrica zabune (prag={threshold:.2f}){title_extra}")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_video_confusion_matrix(vcm, save_path, title=None):
    cm = np.array([[vcm["TN"], vcm["FP"]],
                   [vcm["FN"], vcm["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normalno", "Opasno"])
    ax.set_yticklabels(["Normalno", "Opasno"])
    ax.set_xlabel("Predikcija modela"); ax.set_ylabel("Stvarna oznaka (video)")
    if title is None:
        tol_sec = vcm["tolerance_frames"] / 10
        title = f"Matrica zabune na razini videozapisa (\u00b1{tol_sec:.1f}s tolerancija)"
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_delay_histogram(delays, save_path, title_suffix="", title_extra=""):
    min_d = int(np.floor(delays.min()))
    max_d = int(np.ceil(delays.max()))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delays, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Početak anomalije")
    ax.axvline(delays.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Srednja vrijednost = {delays.mean():.1f}")
    ax.set_xlabel("Kašnjenje (okviri; neg. = rano, poz. = kasno)")
    ax.set_ylabel("Broj detekcija")
    ax.set_title(f"Distribucija kašnjenja (samo detekcije najbliže stvarnim oznakama){title_extra}")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_delay_histogram_all(delays, false_alarm_dists, save_path, title_extra=""):
    all_d = np.concatenate([delays, np.array(false_alarm_dists)]) if false_alarm_dists else delays
    min_d = int(np.floor(all_d.min()))
    max_d = int(np.ceil(all_d.max()))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_d, bins=bins, color="mediumseagreen", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Početak anomalije")
    ax.axvline(all_d.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Srednja vrijednost = {all_d.mean():.1f}")
    ax.set_xlabel("Kašnjenje (okviri)")
    ax.set_ylabel("Broj detekcija")
    ax.set_title(f"Distribucija kašnjenja (sve detekcije, ukupno {len(all_d)}){title_extra}")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_delay_cdf(delays, save_path, title_extra=""):
    sorted_d = np.sort(delays)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_d, cdf, color="steelblue", linewidth=2)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Kašnjenje (okviri)"); ax.set_ylabel("Kumulativni udio")
    ax.set_title(f"Kumulativna distribucija kašnjenja (ref. = početak anomalije){title_extra}")
    ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_cumulative_late(delays, save_path, title_extra=""):
    max_d = int(np.ceil(delays.max())) if delays.max() > 0 else 0
    d_range = np.arange(0, max_d + 1)
    cum_pcts = [(delays >= d).sum() / len(delays) * 100 for d in d_range]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(d_range, cum_pcts, color="coral", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Minimalno kašnjenje (okviri)")
    ax.set_ylabel("Udio događaja s kašnjenjem ≥ N (%)")
    ax.set_title(f"Kumulativno kašnjenje (ref. = početak anomalije){title_extra}")
    ax.set_ylim(0, 105); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


# ========================================================================= #
#  Main                                                                       #
# ========================================================================= #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--postprocess", default="none", choices=POSTPROCESS_CHOICES,
                        help="Post-processing metoda (default: none)")
    parser.add_argument("--frames_root", default=None, help="Override frames_root iz configa")
    parser.add_argument("--metadata_dir", default=None, help="Override metadata_dir iz configa")
    parser.add_argument("--annotations_dir", default=None, help="Override annotations_dir iz configa")
    parser.add_argument("--yolo_detections_dir", default=None, help="Override yolo_detections_dir")
    parser.add_argument("--optical_flow_dir", default=None, help="Override optical_flow_dir")
    parser.add_argument("--disable_flow", action="store_true", help="Isključi optical flow bez obzira na config")
    parser.add_argument("--cnn_cache_dir", default=None, help="Override cnn_cache_dir")
    parser.add_argument("--disable_cnn_cache", action="store_true", help="Isključi cnn cache bez obzira na config")
    parser.add_argument("--test_split_file", default=None,
                        help="Override test split fajla (npr. ../dataset/dota_test_split.txt za DoTA-only evaluaciju)")
    args = parser.parse_args()

    # --- Config ---
    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_config = os.path.join(ckpt_dir, "source", "config.yaml")
    config_path = ckpt_config if os.path.exists(ckpt_config) else "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Runtime override configa za evaluaciju na drugom datasetu ---
    if args.frames_root is not None:
        config["frames_root"] = args.frames_root
    if args.metadata_dir is not None:
        config["metadata_dir"] = args.metadata_dir
    if args.annotations_dir is not None:
        config["annotations_dir"] = args.annotations_dir
    if args.yolo_detections_dir is not None:
        config["yolo_detections_dir"] = args.yolo_detections_dir
    if args.optical_flow_dir is not None:
        config["optical_flow_dir"] = args.optical_flow_dir
        config["use_optical_flow"] = True
        config["use_global_flow"] = True
    if args.disable_flow:
        config["use_optical_flow"] = False
        config["use_global_flow"] = False
    if args.cnn_cache_dir is not None:
        config["cnn_cache_dir"] = args.cnn_cache_dir
    if args.disable_cnn_cache:
        config["cnn_cache_dir"] = None
    if args.test_split_file is not None:
        config["test_split_file"] = args.test_split_file
        print(f"Test split override: {args.test_split_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Model ---
    model = CNN_LSTM_BBox(
        hidden_dim=config.get("hidden_dim", 256),
        lstm_layers=config.get("lstm_layers", 1),
        dropout=config.get("dropout", 0.5),
        lstm_dropout=config.get("lstm_dropout", 0.0),
        backbone=config.get("backbone", "resnet50"),
        freeze_cnn=config.get("freeze_cnn", True),
        unfreeze_layer3=config.get("unfreeze_layer3", False),
        unfreeze_layer4=config.get("unfreeze_layer4", False),
        max_objects=config.get("max_objects", 10),
        roi_output_size=config.get("roi_output_size", 3),
        roi_feat_dim=config.get("roi_feat_dim", 128),
        bbox_feat_dim=config.get("bbox_feat_dim", 64),
        category_embed_dim=config.get("category_embed_dim", 16),
        use_attention=config.get("use_attention", True),
        attention_heads=config.get("attention_heads", 4),
        use_flow=config.get("use_global_flow", config.get("use_optical_flow", False)),
        flow_feat_dim=config.get("flow_feat_dim", 128),
        use_object_interaction=config.get("use_object_interaction", False),
        object_interaction_heads=config.get("object_interaction_heads", 4),
        object_interaction_layers=config.get("object_interaction_layers", 1),
        use_per_object_flow=config.get("use_per_object_flow", False),
        obj_flow_pool_size=config.get("obj_flow_pool_size", 5),
        obj_flow_feat_dim=config.get("obj_flow_feat_dim", 128),
        bidirectional=config.get("bidirectional", False),
        use_causal_attention=config.get("use_causal_attention", True),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Checkpoint: {args.checkpoint}  (epoch {epoch})")
    print(f"Threshold: {args.threshold}")
    print(f"Evaluacija BEZ RAMPE — GT = [anomaly_start, anomaly_end]")
    if args.postprocess != "none":
        print(f"Post-processing: {args.postprocess}")

    # --- DataLoader ---
    seq_len = config.get("seq_len", 24)
    config["batch_size"] = 32
    config["num_workers"] = 4
    config["persistent_workers"] = True
    config["prefetch_factor"] = 4
    config["pin_memory"] = True
    loader = create_test_dataloader(config, split=args.split)

    # --- Inference ---
    cache_key_src = "|".join([
        str(args.split),
        str(config.get("frames_root", "")),
        str(config.get("metadata_dir", "")),
        str(config.get("yolo_detections_dir", "")),
        str(config.get("optical_flow_dir", "")) if config.get("use_optical_flow", False) else "flow_off",
        str(config.get("cnn_cache_dir", "")),
        str(config.get("test_split_file", "")),  # različit hash za DoTA-only evaluaciju
    ])
    cache_hash = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()[:10]
    probs_cache = os.path.join(ckpt_dir, f"video_probs_{cache_hash}.pkl")
    if os.path.isfile(probs_cache):
        print(f"Učitavam cached predikcije: {probs_cache}")
        with open(probs_cache, "rb") as f:
            video_probs = pickle.load(f)
    else:
        video_probs = run_inference(model, loader, device, seq_len)
        with open(probs_cache, "wb") as f:
            pickle.dump(dict(video_probs), f)
        print(f"Predikcije spremljene: {probs_cache}")

    # --- Metadata ---
    metadata_dir = config.get("metadata_dir", "../dataset")
    all_metadata = {}
    for fname in ["metadata_train.json", "metadata_val.json", "metadata_test.json"]:
        p = os.path.join(metadata_dir, fname)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                all_metadata.update(json.load(f))

    # --- Post-processing ---
    pp_label = postprocess_label(args.postprocess)
    if args.postprocess != "none":
        print(f"\nPost-processing: {args.postprocess}{pp_label}")
        video_probs = apply_postprocess(video_probs, args.postprocess)

    # --- Binarni GT (bez rampe) ---
    probs, labels, video_names = build_binary_labels(video_probs, all_metadata)
    print(f"\nUkupno uzoraka: {len(labels)}")
    print(f"Pozitivni (u anomaliji): {int(labels.sum())} ({labels.mean()*100:.1f}%)")
    print(f"Negativni: {int((1-labels).sum())} ({(1-labels).mean()*100:.1f}%)")

    # --- Klasifikacijske metrike ---
    metrics_05 = compute_metrics(probs, labels, threshold=0.5)
    best_thr, best_f1 = find_best_threshold(probs, labels)
    metrics_best = compute_metrics(probs, labels, threshold=best_thr)
    metrics_custom = compute_metrics(probs, labels, threshold=args.threshold) if args.threshold != 0.5 else None

    print(f"\n{'='*60}")
    print(f"REZULTATI BEZ RAMPE ({args.split} set)")
    print(f"{'='*60}")
    print(f"  AUC-ROC:         {metrics_05['auc_roc']:.4f}")
    print(f"  Average Prec:    {metrics_05['average_precision']:.4f}")
    print(f"\n  --- Threshold = 0.5 ---")
    print(f"  Accuracy:   {metrics_05['accuracy']:.4f}")
    print(f"  Precision:  {metrics_05['precision']:.4f}")
    print(f"  Recall:     {metrics_05['recall']:.4f}")
    print(f"  F1:         {metrics_05['f1']:.4f}")
    cm = metrics_05["confusion_matrix"]
    print(f"  TP={cm['TP']}  FP={cm['FP']}  FN={cm['FN']}  TN={cm['TN']}")
    print(f"\n  --- Najbolji threshold = {best_thr:.2f} ---")
    print(f"  Accuracy:   {metrics_best['accuracy']:.4f}")
    print(f"  Precision:  {metrics_best['precision']:.4f}")
    print(f"  Recall:     {metrics_best['recall']:.4f}")
    print(f"  F1:         {metrics_best['f1']:.4f}")

    # --- Video-level confusion matrices ---
    vcm_configs = [
        {"name": "±1.0s", "kwargs": {"tolerance_frames": 10}, "fname": "video_cm_pm1s"},
        {"name": "±2.0s", "kwargs": {"tolerance_frames": 20}, "fname": "video_cm_pm2s"},
        {"name": "-2.0s do početka anomalije", "kwargs": {"before_frames": 20, "after_frames": 0}, "fname": "video_cm_before2s"},
    ]
    vcm_results = {}
    for vcfg in vcm_configs:
        vcm = compute_video_level_confusion(video_probs, all_metadata, threshold=args.threshold, **vcfg["kwargs"])
        vcm_results[vcfg["fname"]] = vcm
        print(f"\n{'='*60}")
        print(f"VIDEO-LEVEL CONFUSION MATRIX ({vcfg['name']})")
        print(f"{'='*60}")
        print(f"  TP={vcm['TP']}  FP={vcm['FP']}  FN={vcm['FN']}  TN={vcm['TN']}")
        print(f"  Accuracy:   {vcm['accuracy']:.4f}")
        print(f"  Precision:  {vcm['precision']:.4f}")
        print(f"  Recall:     {vcm['recall']:.4f}")
        print(f"  F1:         {vcm['f1']:.4f}")

    # --- Delay analiza (ref = anomaly_start) ---
    delay_res = analyze_delays_no_ramp(video_probs, all_metadata, threshold=args.threshold)
    delays = delay_res["delays"]
    missed = delay_res["missed_events"]
    fa_dists = delay_res["false_alarm_distances"]
    total_av = delay_res["total_anomaly_videos"]

    print(f"\n{'='*60}")
    print(f"DELAY ANALIZA BEZ RAMPE (ref = anomaly_start)")
    print(f"{'='*60}")
    print(f"  Videa s anomalijom: {total_av}")
    print(f"  Detektirano: {len(delays)}")
    print(f"  Promašeno: {missed}")
    print(f"  Dodatni onseti: {len(fa_dists)}")

    if len(delays) > 0:
        early = (delays < 0).sum()
        on_time = (delays == 0).sum()
        late = (delays > 0).sum()
        print(f"\n  Mean:   {delays.mean():.1f}")
        print(f"  Median: {np.median(delays):.1f}")
        print(f"  Std:    {delays.std():.1f}")
        print(f"  Min:    {delays.min():.1f}")
        print(f"  Max:    {delays.max():.1f}")
        print(f"\n  Rano:   {early} ({early/len(delays)*100:.1f}%)")
        print(f"  Točno:  {on_time} ({on_time/len(delays)*100:.1f}%)")
        print(f"  Kasno:  {late} ({late/len(delays)*100:.1f}%)")

    # --- Spremi grafove ---
    if args.postprocess != "none":
        out_dir = os.path.join(ckpt_dir, f"no_ramp_{args.postprocess}")
    else:
        out_dir = os.path.join(ckpt_dir, "no_ramp")
    os.makedirs(out_dir, exist_ok=True)

    try:
        plot_roc_curve(labels, probs, os.path.join(out_dir, "roc_curve.png"), pp_label)
        plot_pr_curve(labels, probs, os.path.join(out_dir, "pr_curve.png"), pp_label)
        plot_score_distribution(labels, probs, os.path.join(out_dir, "score_dist.png"), pp_label)
        plot_confusion_matrix(metrics_05["confusion_matrix"],
                              os.path.join(out_dir, f"confusion_matrix_050.png"), 0.5, pp_label)
        plot_confusion_matrix(metrics_best["confusion_matrix"],
                              os.path.join(out_dir, f"confusion_matrix_{best_thr:.2f}.png"), best_thr, pp_label)
        if metrics_custom:
            plot_confusion_matrix(metrics_custom["confusion_matrix"],
                                  os.path.join(out_dir, f"confusion_matrix_{args.threshold:.2f}.png"),
                                  args.threshold, pp_label)
        for vcfg in vcm_configs:
            vname = vcfg["fname"]
            vcm_i = vcm_results[vname]
            plot_video_confusion_matrix(vcm_i, os.path.join(out_dir, f"{vname}.png"),
                                        title=f"Matrica zabune na razini videozapisa ({vcfg['name']}){pp_label}")
        print(f"\n  Klasifikacijski grafovi: {out_dir}/")
    except Exception as e:
        print(f"  UPOZORENJE: Plot greška: {e}")

    if len(delays) > 0:
        try:
            plot_delay_histogram(delays, os.path.join(out_dir, "delay_histogram.png"),
                                 "", pp_label)
            plot_delay_histogram_all(delays, fa_dists, os.path.join(out_dir, "delay_histogram_all_onsets.png"), pp_label)
            plot_delay_cdf(delays, os.path.join(out_dir, "delay_cdf.png"), pp_label)
            plot_cumulative_late(delays, os.path.join(out_dir, "delay_cumulative_late.png"), pp_label)
            print(f"  Delay grafovi: {out_dir}/")
        except Exception as e:
            print(f"  UPOZORENJE: Delay plot greška: {e}")

    # --- JSON ---
    thr_sweep = threshold_sweep(probs, labels)
    results = {
        "mode": "no_ramp",
        "postprocess": args.postprocess,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "threshold": args.threshold,
        "metrics_thr_0.5": metrics_05,
        "best_threshold": best_thr,
        "metrics_best_thr": metrics_best,
        "threshold_sweep": thr_sweep,
        "delay_analysis": {
            "reference_point": "anomaly_start",
            "total_anomaly_videos": total_av,
            "detected": len(delays),
            "missed": int(missed),
            "delays_mean": float(delays.mean()) if len(delays) > 0 else None,
            "delays_median": float(np.median(delays)) if len(delays) > 0 else None,
            "delays_std": float(delays.std()) if len(delays) > 0 else None,
            "delays_min": float(delays.min()) if len(delays) > 0 else None,
            "delays_max": float(delays.max()) if len(delays) > 0 else None,
            "early_count": int(early) if len(delays) > 0 else 0,
            "on_time_count": int(on_time) if len(delays) > 0 else 0,
            "late_count": int(late) if len(delays) > 0 else 0,
            "all_delays": [float(d) for d in delays],
            "false_alarm_count": len(fa_dists),
            "false_alarm_distances": [float(d) for d in fa_dists],
        },
        "video_level_confusion": {
            vname: {
                "description": vcfg["name"],
                "TP": vcm_results[vname]["TP"], "FP": vcm_results[vname]["FP"],
                "FN": vcm_results[vname]["FN"], "TN": vcm_results[vname]["TN"],
                "accuracy": vcm_results[vname]["accuracy"],
                "precision": vcm_results[vname]["precision"],
                "recall": vcm_results[vname]["recall"],
                "f1": vcm_results[vname]["f1"],
                "fn_videos": vcm_results[vname]["fn_videos"],
                "fp_videos": vcm_results[vname]["fp_videos"],
            } for vcfg, vname in [(c, c["fname"]) for c in vcm_configs]
        },
    }
    json_path = os.path.join(out_dir, "results_no_ramp.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")
    print(f"\nGotovo!")


if __name__ == "__main__":
    main()
