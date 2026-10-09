import os
import sys
import json
import pickle
import random
import argparse
from collections import defaultdict

import yaml
import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix,
)
from scipy.ndimage import median_filter as scipy_median_filter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_3d import CNN3D_Danger
from dataloader import create_test_dataloader
from utils import (
    POSTPROCESS_CHOICES, apply_postprocess, postprocess_label,
    compute_metrics, find_best_threshold, threshold_sweep,
    compute_video_level_confusion
)

def _trailing_ma(probs, window):
    cs = np.cumsum(np.insert(probs, 0, 0.0))
    out = np.empty_like(probs, dtype=float)
    for i in range(len(probs)):
        start = max(0, i - window + 1)
        out[i] = (cs[i + 1] - cs[start]) / (i + 1 - start)
    return out

def _ema(probs, alpha):
    out = np.empty_like(probs, dtype=float)
    out[0] = float(probs[0])
    for i in range(1, len(probs)):
        out[i] = alpha * probs[i] + (1.0 - alpha) * out[i - 1]
    return out

def _hysteresis(probs, high_thr, low_thr):
    out = np.zeros(len(probs), dtype=float)
    active = False
    for i, p in enumerate(probs):
        if not active and p >= high_thr:
            active = True
        elif active and p < low_thr:
            active = False
        out[i] = 1.0 if active else 0.0
    return out

def apply_postprocess(video_probs, method):
    if method == "none":
        return video_probs

    result = {}
    for vn, frame_probs in video_probs.items():
        sorted_fp = sorted(frame_probs, key=lambda x: x[0])
        frames = [f[0] for f in sorted_fp]
        probs = np.array([f[1] for f in sorted_fp])

        if method.startswith("tma"):
            probs = _trailing_ma(probs, int(method[3:]))
        elif method.startswith("ma"):
            w = int(method[2:])
            kernel = np.ones(w) / w
            probs = np.convolve(probs, kernel, mode="same")
        elif method.startswith("median"):
            k = int(method[6:])
            probs = scipy_median_filter(probs, size=k)
        elif method.startswith("ema") and "_hyst" not in method:
            alpha = int(method[3:]) / 10.0
            probs = _ema(probs, alpha)
        elif method == "ema02_hyst_05_02":
            probs = _hysteresis(_ema(probs, 0.2), 0.5, 0.2)
        elif method == "ema03_hyst_05_02":
            probs = _hysteresis(_ema(probs, 0.3), 0.5, 0.2)
        elif method == "tma5_hyst_05_02":
            probs = _hysteresis(_trailing_ma(probs, 5), 0.5, 0.2)
        elif method == "hyst_05_02":
            probs = _hysteresis(probs, 0.5, 0.2)
        elif method == "hyst_06_03":
            probs = _hysteresis(probs, 0.6, 0.3)
        elif method == "hyst_07_03":
            probs = _hysteresis(probs, 0.7, 0.3)

        result[vn] = [(frames[i], float(probs[i])) for i in range(len(frames))]
    return result

def postprocess_label(method):
    if method == "none":
        return ""
    if method.startswith("tma"):
        return f" + Trailing MA (w={method[3:]})"
    if method.startswith("ma"):
        return f" + Moving Average (w={method[2:]})"
    if method.startswith("median"):
        return f" + Median Filter (k={method[6:]})"
    if method == "ema02_hyst_05_02":
        return " + EMA(0.2)+Hyst(0.5/0.2)"
    if method == "ema03_hyst_05_02":
        return " + EMA(0.3)+Hyst(0.5/0.2)"
    if method == "tma5_hyst_05_02":
        return " + TrailMA(5)+Hyst(0.5/0.2)"
    if method.startswith("ema"):
        return f" + EMA (alpha=0.{method[3:]})"
    if method.startswith("hyst_"):
        h, l = method.split("_")[1:]
        return f" + Hysteresis(H=0.{h}, L=0.{l})"
    return f" + {method}"

def run_inference(model, loader, device, seq_len):
    model.eval()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    video_probs = defaultdict(list)
    video_sample_count = defaultdict(int)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            frames = batch["frames"].to(device, non_blocking=True)
            bbox_features = batch["bbox_features"].to(device, non_blocking=True)
            categories = batch["categories"].to(device, non_blocking=True)
            obj_mask = batch["obj_mask"].to(device, non_blocking=True)
            flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
            vid_names = batch["video_name"]

            with autocast(device_type=device.type, dtype=amp_dtype,
                          enabled=(device.type == "cuda")):
                logits = model(
                    frames, bbox_features=bbox_features,
                    categories=categories, obj_mask=obj_mask, flow=flow,
                )
            probs = torch.sigmoid(logits).float().cpu().numpy()

            for i, vn in enumerate(vid_names):
                frame_idx = seq_len + video_sample_count[vn]
                video_sample_count[vn] += 1
                video_probs[vn].append((frame_idx, float(probs[i])))

    return video_probs

def _meta_num_frames(meta):
    if not meta:
        return 0
    L = (meta.get("num_frames") or meta.get("total_frames")
         or meta.get("n_frames") or meta.get("length") or 0)
    if not L and "frames" in meta and isinstance(meta["frames"], dict) and meta["frames"]:
        L = max(int(x) for x in meta["frames"].keys()) + 1
    return int(L)

def subsample_video_window(video_probs, all_metadata, window_len, seed, only_yt=True):

    if window_len <= 0:
        return video_probs, {}
    out = {}
    chosen = {}
    for vn, frame_probs in video_probs.items():
        if only_yt and not vn.startswith("yt_"):
            out[vn] = list(frame_probs)
            continue
        n_frames = _meta_num_frames(all_metadata.get(vn))
        if n_frames <= 0:
            n_frames = max((fi for fi, _ in frame_probs), default=0) + 1
        if n_frames <= window_len:
            out[vn] = list(frame_probs)
            chosen[vn] = (0, n_frames)
            continue
        rng = random.Random(f"{seed}|{vn}")
        start = rng.randint(0, n_frames - window_len)
        end = start + window_len
        kept = [(fi, p) for (fi, p) in frame_probs if start <= fi < end]
        out[vn] = kept
        chosen[vn] = (start, end)
    return out, chosen

def build_binary_labels(video_probs, all_metadata):

    all_probs, all_labels, all_video_names = [], [], []
    for vn in sorted(video_probs.keys()):
        meta = all_metadata.get(vn)
        if meta is None:
            continue
        a_start = meta.get("anomaly_start", -1)
        a_end = meta.get("anomaly_end", -1)

        preds = sorted(video_probs[vn], key=lambda x: x[0])
        for frame_idx, prob in preds:
            all_probs.append(prob)
            if a_start >= 0 and a_end >= 0 and a_start <= frame_idx <= a_end:
                all_labels.append(1.0)
            else:
                all_labels.append(0.0)
            all_video_names.append(vn)

    return np.array(all_probs), np.array(all_labels), all_video_names

def analyze_delays_no_ramp(video_probs, all_metadata, threshold=0.5):
    delays = []
    missed_events = 0
    false_alarm_distances = []
    total_anomaly_videos = 0

    for vn in sorted(video_probs.keys()):
        meta = all_metadata.get(vn)
        if meta is None:
            continue
        a_start = meta.get("anomaly_start", -1)
        a_end = meta.get("anomaly_end", -1)
        if a_start < 0 or a_end < 0:
            continue

        total_anomaly_videos += 1
        gt_ref = float(a_start)

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

        best = min(onsets, key=lambda x: abs(x - gt_ref))
        delays.append(best - gt_ref)
        for o in onsets:
            if o != best:
                false_alarm_distances.append(o - gt_ref)

    return {
        "delays": np.array(delays),
        "missed_events": missed_events,
        "false_alarm_distances": false_alarm_distances,
        "total_anomaly_videos": total_anomaly_videos,
    }

def plot_roc_curve(labels, probs, save_path, title_extra=""):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Slučajni klasifikator")
    ax.set_xlabel("Stopa lažno pozitivnih"); ax.set_ylabel("Stopa istinski pozitivnih")
    ax.set_title(f"ROC krivulja (3D CNN){title_extra}")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
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
    ax.set_title(f"Precision-Recall krivulja (3D CNN){title_extra}")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
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
    ax.set_title(f"Matrica zabune (prag={threshold:.2f}, 3D CNN){title_extra}")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def plot_video_confusion_matrix(vcm, save_path, title):
    cm = np.array([[vcm["TN"], vcm["FP"]],
                   [vcm["FN"], vcm["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normalno", "Opasno"])
    ax.set_yticklabels(["Normalno", "Opasno"])
    ax.set_xlabel("Predikcija modela"); ax.set_ylabel("Stvarna oznaka (video)")
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
    ax.set_title(f"Distribucija kašnjenja (samo detekcije najbliže stvarnim oznakama, 3D CNN){title_extra}")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def plot_delay_histogram_all(delays, fa_dists, save_path, title_extra=""):
    all_d = np.concatenate([delays, np.array(fa_dists)]) if fa_dists else delays
    min_d = int(np.floor(all_d.min()))
    max_d = int(np.ceil(all_d.max()))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_d, bins=bins, color="mediumseagreen", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Početak anomalije")
    ax.axvline(all_d.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Srednja vrijednost = {all_d.mean():.1f}")
    ax.set_xlabel("Kašnjenje (okviri)"); ax.set_ylabel("Broj detekcija")
    ax.set_title(f"Distribucija kašnjenja (sve detekcije, ukupno {len(all_d)}, 3D CNN){title_extra}")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--postprocess", default="none", choices=POSTPROCESS_CHOICES)
    parser.add_argument("--no_cache", action="store_true",
                        help="Ignoriraj cache i ponovno pokreni inference.")
    parser.add_argument("--window-len", type=int, default=0,
                        help="Ako > 0, video se reduce na nasumicni prozor te duljine (frameovi).")
    parser.add_argument("--window-seed", type=int, default=42,
                        help="Seed za odabir starta prozora po videu.")
    parser.add_argument("--window-all", action="store_true",
                        help="Primijeni window i na DoTA videe (default: samo yt_).")
    parser.add_argument("--test_split_file", default=None,
                        help="Override test split fajla (npr. ../../dataset/dota_test_split.txt za DoTA-only evaluaciju)")
    args = parser.parse_args()

    ckpt_dir = os.path.dirname(args.checkpoint)
    snap_cfg = os.path.join(ckpt_dir, "source", "config_3d.yaml")
    config_path = snap_cfg if os.path.exists(snap_cfg) else "config_3d.yaml"
    print(f"Config: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uređaj: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = CNN3D_Danger(
        backbone_3d=config.get("backbone_3d", "r2plus1d_18"),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.5),
        freeze_3d=config.get("freeze_3d", False),
        freeze_early=config.get("freeze_early", True),
        use_grad_checkpoint=False,
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
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"  [INFO] Ignorirani ključevi iz checkpointa (auxiliary glave): {unexpected}")
    if missing:
        print(f"  [WARN] Nedostajući ključevi u checkpointu: {missing}")
    epoch = ckpt.get("epoch", "?")
    print(f"Checkpoint: {args.checkpoint}   epoch={epoch}")
    print(f"Threshold: {args.threshold}")
    print(f"Evaluacija BEZ RAMPE — GT = [anomaly_start, anomaly_end]")
    if args.postprocess != "none":
        print(f"Post-processing: {args.postprocess}")

    seq_len = config.get("seq_len", 24)
    config["stride"] = 1
    config["batch_size"] = config.get("val_batch_size", 192)
    config["num_workers"] = 6
    config["persistent_workers"] = True
    config["prefetch_factor"] = 4
    config["pin_memory"] = True
    if args.test_split_file is not None:
        config["test_split_file"] = args.test_split_file
        print(f"Test split override: {args.test_split_file}")
    loader = create_test_dataloader(config, split=args.split)

    split_suffix = "_dota" if config.get("test_split_file") else ""
    probs_cache = os.path.join(ckpt_dir, f"video_probs_{args.split}{split_suffix}.pkl")
    if os.path.isfile(probs_cache) and not args.no_cache:
        print(f"Učitavam cached predikcije: {probs_cache}")
        with open(probs_cache, "rb") as f:
            video_probs = pickle.load(f)
    else:
        video_probs = run_inference(model, loader, device, seq_len)
        with open(probs_cache, "wb") as f:
            pickle.dump(dict(video_probs), f)
        print(f"Predikcije spremljene: {probs_cache}")

    metadata_dir = config.get("metadata_dir", "../../dataset")
    all_metadata = {}
    for fname in ["metadata_train.json", "metadata_val.json", "metadata_test.json"]:
        p = os.path.join(metadata_dir, fname)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                all_metadata.update(json.load(f))

    if args.window_len > 0:
        n_before = sum(len(v) for v in video_probs.values())
        video_probs, chosen = subsample_video_window(
            video_probs, all_metadata, args.window_len, args.window_seed,
            only_yt=not args.window_all)
        n_after = sum(len(v) for v in video_probs.values())
        n_truncated = sum(1 for vn, (s, e) in chosen.items()
                          if (e - s) == args.window_len)
        scope = "sve" if args.window_all else "samo yt_"
        print(f"\nWindow subsample: window_len={args.window_len} seed={args.window_seed} ({scope})")
        print(f"  Truncated videa: {n_truncated} / {len(chosen)}")
        print(f"  Predikcije: {n_before} -> {n_after}")

    pp_label = postprocess_label(args.postprocess)
    if args.postprocess != "none":
        print(f"\nPost-processing: {args.postprocess}{pp_label}")
        video_probs = apply_postprocess(video_probs, args.postprocess)

    probs, labels, _ = build_binary_labels(video_probs, all_metadata)
    print(f"\nUkupno uzoraka: {len(labels)}")
    print(f"Pozitivni: {int(labels.sum())} ({labels.mean()*100:.1f}%)")
    print(f"Negativni: {int((1-labels).sum())} ({(1-labels).mean()*100:.1f}%)")

    metrics_05 = compute_metrics(probs, labels, threshold=0.5)
    best_thr, best_f1 = find_best_threshold(probs, labels)
    metrics_best = compute_metrics(probs, labels, threshold=best_thr)
    metrics_custom = compute_metrics(probs, labels, threshold=args.threshold) if args.threshold != 0.5 else None

    print(f"\n{'='*60}")
    print(f"FRAME-LEVEL REZULTATI BEZ RAMPE ({args.split} set)")
    print(f"{'='*60}")
    print(f"  AUC-ROC:      {metrics_05['auc_roc']:.4f}")
    print(f"  Average Prec: {metrics_05['average_precision']:.4f}")
    print(f"\n  --- Threshold = 0.5 ---")
    print(f"  Acc  {metrics_05['accuracy']:.4f}  P  {metrics_05['precision']:.4f}  "
          f"R  {metrics_05['recall']:.4f}  F1  {metrics_05['f1']:.4f}")
    cm = metrics_05["confusion_matrix"]
    print(f"  TP={cm['TP']}  FP={cm['FP']}  FN={cm['FN']}  TN={cm['TN']}")
    print(f"\n  --- Najbolji threshold = {best_thr:.2f} ---")
    print(f"  Acc  {metrics_best['accuracy']:.4f}  P  {metrics_best['precision']:.4f}  "
          f"R  {metrics_best['recall']:.4f}  F1  {metrics_best['f1']:.4f}")

    vcm_configs = [
        {"name": "±1.0s", "kwargs": {"tolerance_frames": 10}, "fname": "video_cm_pm1s"},
        {"name": "±2.0s", "kwargs": {"tolerance_frames": 20}, "fname": "video_cm_pm2s"},
        {"name": "-2.0s do početka anomalije",
         "kwargs": {"before_frames": 20, "after_frames": 0},
         "fname": "video_cm_before2s"},
    ]
    vcm_results = {}
    for vcfg in vcm_configs:
        vcm = compute_video_level_confusion(
            video_probs, all_metadata, threshold=args.threshold, **vcfg["kwargs"])
        vcm_results[vcfg["fname"]] = vcm
        print(f"\n{'='*60}")
        print(f"VIDEO-LEVEL CONFUSION MATRIX ({vcfg['name']})")
        print(f"{'='*60}")
        print(f"  TP={vcm['TP']}  FP={vcm['FP']}  FN={vcm['FN']}  TN={vcm['TN']}")
        print(f"  Acc {vcm['accuracy']:.4f}  P {vcm['precision']:.4f}  "
              f"R {vcm['recall']:.4f}  F1 {vcm['f1']:.4f}")

    delay_res = analyze_delays_no_ramp(video_probs, all_metadata, threshold=args.threshold)
    delays = delay_res["delays"]
    missed = delay_res["missed_events"]
    fa_dists = delay_res["false_alarm_distances"]
    total_av = delay_res["total_anomaly_videos"]

    print(f"\n{'='*60}")
    print(f"DELAY ANALIZA (ref = anomaly_start)")
    print(f"{'='*60}")
    print(f"  Videa s anomalijom: {total_av}")
    print(f"  Detektirano: {len(delays)}")
    print(f"  Promašeno: {missed}")
    print(f"  Dodatni onseti: {len(fa_dists)}")

    early = on_time = late = 0
    if len(delays) > 0:
        early = int((delays < 0).sum())
        on_time = int((delays == 0).sum())
        late = int((delays > 0).sum())
        print(f"\n  Mean: {delays.mean():.1f}   Median: {np.median(delays):.1f}   "
              f"Std: {delays.std():.1f}   Min: {delays.min():.1f}   Max: {delays.max():.1f}")
        print(f"  Rano:  {early} ({early/len(delays)*100:.1f}%)")
        print(f"  Točno: {on_time} ({on_time/len(delays)*100:.1f}%)")
        print(f"  Kasno: {late} ({late/len(delays)*100:.1f}%)")

    suffix = ""
    if args.postprocess != "none":
        suffix += f"_{args.postprocess}"
    if args.window_len > 0:
        suffix += f"_w{args.window_len}s{args.window_seed}"
    out_dir = os.path.join(ckpt_dir, f"no_ramp{suffix}") if suffix else os.path.join(ckpt_dir, "no_ramp")
    os.makedirs(out_dir, exist_ok=True)

    try:
        plot_roc_curve(labels, probs, os.path.join(out_dir, "roc_curve.png"), pp_label)
        plot_pr_curve(labels, probs, os.path.join(out_dir, "pr_curve.png"), pp_label)
        plot_confusion_matrix(metrics_05["confusion_matrix"],
                              os.path.join(out_dir, "confusion_matrix_050.png"), 0.5, pp_label)
        plot_confusion_matrix(metrics_best["confusion_matrix"],
                              os.path.join(out_dir, f"confusion_matrix_{best_thr:.2f}.png"),
                              best_thr, pp_label)
        if metrics_custom:
            plot_confusion_matrix(metrics_custom["confusion_matrix"],
                                  os.path.join(out_dir, f"confusion_matrix_{args.threshold:.2f}.png"),
                                  args.threshold, pp_label)
        for vcfg in vcm_configs:
            vname = vcfg["fname"]
            plot_video_confusion_matrix(
                vcm_results[vname],
                os.path.join(out_dir, f"{vname}.png"),
                title=f"Matrica zabune na razini videozapisa ({vcfg['name']}, 3D CNN){pp_label}",
            )
        print(f"\n  Klasifikacijski grafovi: {out_dir}/")
    except Exception as e:
        print(f"  UPOZORENJE: plot greška: {e}")

    if len(delays) > 0:
        try:
            plot_delay_histogram(delays, os.path.join(out_dir, "delay_histogram.png"),
                                 " (najbliži onset)", pp_label)
            plot_delay_histogram_all(delays, fa_dists,
                                     os.path.join(out_dir, "delay_histogram_all_onsets.png"), pp_label)
            print(f"  Delay grafovi: {out_dir}/")
        except Exception as e:
            print(f"  UPOZORENJE: delay plot greška: {e}")

    thr_sweep = threshold_sweep(probs, labels)
    results = {
        "mode": "no_ramp",
        "model": "CNN3D_Danger",
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
            "early_count": early,
            "on_time_count": on_time,
            "late_count": late,
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
            }
            for vcfg, vname in [(c, c["fname"]) for c in vcm_configs]
        },
    }
    json_path = os.path.join(out_dir, "results_no_ramp.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")
    print("\nGotovo!")

if __name__ == "__main__":
    main()
