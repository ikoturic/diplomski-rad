"""
Evaluacija CNN+LSTM+BBox modela na test setu.

Pokretanje:
    python evaluate.py --config config.yaml --checkpoint checkpoints/v4/best_model.pth

Metrike:
    - AUC-ROC, Average Precision (AP)
    - Precision, Recall, F1 na raznim thresholdima
    - ROC i PR krivulje
    - Confusion matrix
    - Per-video analiza
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
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

from model import CNN_LSTM_BBox
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

    # BFloat16 na Ampere+ GPU-ovima, inače Float16
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for batch in tqdm(loader, desc="Evaluacija"):
        frames = batch["frames"].to(device, non_blocking=True)
        bboxes = batch["bboxes"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"]
        video_names = batch["video_name"]
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(frames, bboxes, bbox_features, categories, obj_mask, flow=flow)

        all_probs_gpu.append(torch.sigmoid(logits))
        all_labels_gpu.append((target >= 0.5).float())
        all_video_names.extend(video_names)

    # Jedan CPU transfer na kraju
    all_probs = torch.cat(all_probs_gpu).float().cpu().numpy()
    all_labels = torch.cat(all_labels_gpu).numpy()

    return all_probs, all_labels, all_video_names


# ========================================================================= #
#  Compute metrike                                                            #
# ========================================================================= #

def compute_metrics(probs, labels, threshold=0.5):
    """Računa sve binarne metrike."""
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
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }

    return metrics


def find_best_threshold(probs, labels):
    """Pronalazi threshold koji maksimizira F1."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= thr).astype(float)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr), float(best_f1)


def per_video_analysis(probs, labels, video_names):
    """Agregira metrike po videu – prosječna predikcija i label za svaki video."""
    vid_data = defaultdict(lambda: {"probs": [], "labels": []})
    for p, l, v in zip(probs, labels, video_names):
        vid_data[v]["probs"].append(p)
        vid_data[v]["labels"].append(l)

    results = []
    for vid, data in vid_data.items():
        avg_prob = float(np.mean(data["probs"]))
        max_prob = float(np.max(data["probs"]))
        has_anomaly = int(any(l >= 0.5 for l in data["labels"]))
        n_samples = len(data["probs"])
        n_positive = sum(1 for l in data["labels"] if l >= 0.5)
        results.append({
            "video": vid,
            "n_samples": n_samples,
            "n_positive": n_positive,
            "has_anomaly": has_anomaly,
            "avg_prob": avg_prob,
            "max_prob": max_prob,
        })
    return sorted(results, key=lambda x: x["max_prob"], reverse=True)


# ========================================================================= #
#  Grafovi                                                                    #
# ========================================================================= #

def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Krivulja")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(labels, probs, save_path):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, "r-", lw=2, label=f"AP = {ap:.4f}")
    baseline = labels.sum() / len(labels)
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.4, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Krivulja")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_score_distribution(labels, probs, save_path):
    pos_probs = probs[labels >= 0.5]
    neg_probs = probs[labels < 0.5]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(neg_probs, bins=50, alpha=0.6, color="green", label="Normalno", density=True)
    ax.hist(pos_probs, bins=50, alpha=0.6, color="red", label="Opasno", density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Distribucija Score-ova")
    ax.legend()
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
#  Threshold sweep                                                            #
# ========================================================================= #

def threshold_sweep(probs, labels):
    """Računa metrike za raspon thresholdova."""
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
#  Main                                                                       #
# ========================================================================= #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--eval_stride", type=int, default=None,
                        help="Override stride za evaluaciju (default: koristi config)")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--backbone", default=None,
                        help="Override backbone (resnet18/resnet50)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Override hidden_dim")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override backbone ako je zadan
    if args.backbone is not None:
        config["backbone"] = args.backbone
    if args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim

    # Override stride za evaluaciju ako je zadan
    if args.eval_stride is not None:
        print(f"Override stride: {config.get('stride', 12)} -> {args.eval_stride}")
        config["stride"] = args.eval_stride

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uređaj: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

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
        use_flow=config.get("use_optical_flow", False),
        flow_feat_dim=config.get("flow_feat_dim", 128),
        use_object_interaction=config.get("use_object_interaction", False),
        object_interaction_heads=config.get("object_interaction_heads", 4),
        object_interaction_layers=config.get("object_interaction_layers", 1),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"Checkpoint: {args.checkpoint}  (epoch {epoch})")

    # --- Data (samo traženi split, brže nego gradnja svih) ---
    # Za eval: manji batch da workeri stignu prefetch-ati (svaki uzorak = 24 framea + flow)
    eval_bs = config.get("eval_batch_size", config.get("batch_size", 8))
    config["batch_size"] = eval_bs
    # Više workera za eval I/O
    config["num_workers"] = config.get("val_num_workers", config.get("num_workers", 4))
    loader = create_test_dataloader(config, split=args.split)
    print(f"Evaluacija na {args.split} setu ({len(loader.dataset)} uzoraka, bs={eval_bs})")

    # --- Evaluate ---
    probs, labels, video_names = evaluate_model(model, loader, device)

    # --- Metrics at default threshold ---
    metrics = compute_metrics(probs, labels, threshold=0.5)
    best_thr, best_f1 = find_best_threshold(probs, labels)
    metrics_best_thr = compute_metrics(probs, labels, threshold=best_thr)

    print(f"\n{'='*60}")
    print(f"REZULTATI ({args.split} set)")
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
    print(f"  Accuracy:   {metrics_best_thr['accuracy']:.4f}")
    print(f"  Precision:  {metrics_best_thr['precision']:.4f}")
    print(f"  Recall:     {metrics_best_thr['recall']:.4f}")
    print(f"  F1:         {metrics_best_thr['f1']:.4f}")

    # --- Save ---
    save_dir = args.save_dir or os.path.dirname(args.checkpoint)
    os.makedirs(save_dir, exist_ok=True)

    # Save JSON
    thr_sweep = threshold_sweep(probs, labels)
    vid_analysis = per_video_analysis(probs, labels, video_names)

    results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "metrics_thr_0.5": metrics,
        "best_threshold": best_thr,
        "metrics_best_thr": metrics_best_thr,
        "threshold_sweep": thr_sweep,
        "per_video_top20": vid_analysis[:20],
        "total_videos": len(set(video_names)),
    }
    json_path = os.path.join(save_dir, f"test_results_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON: {json_path}")

    # Plot curves
    try:
        plot_roc_curve(labels, probs, os.path.join(save_dir, "roc_curve.png"))
        print(f"  ROC curve:   {save_dir}/roc_curve.png")

        plot_pr_curve(labels, probs, os.path.join(save_dir, "pr_curve.png"))
        print(f"  PR curve:    {save_dir}/pr_curve.png")

        plot_score_distribution(labels, probs, os.path.join(save_dir, "score_dist.png"))
        print(f"  Score dist:  {save_dir}/score_dist.png")

        plot_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix_050.png"), 0.5)
        plot_confusion_matrix(
            metrics_best_thr["confusion_matrix"],
            os.path.join(save_dir, f"confusion_matrix_{best_thr:.2f}.png"),
            best_thr,
        )
        print(f"  Conf matrix: {save_dir}/confusion_matrix_*.png")
    except Exception as e:
        print(f"  UPOZORENJE: Plot greška: {e}")

    print(f"\nEvaluacija završena!")


if __name__ == "__main__":
    main()
