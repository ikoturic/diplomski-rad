"""
Pretrening skripta za 2D CNN+LSTM+BBox model na TAU-106K skupu.

Analogno 3D CNN pretrening skripti (3D CNN/train_3d.py s config_tau_pretrain.yaml).
Trenira ResNet18 backbone zajedno s cijelim modelom na TAU-106K binarnim labelama,
potom sprema samo CNN backbone (cnn_early / cnn_layer3 / cnn_layer4) za transfer
na DoTA fine-tuning.

Pokretanje:
    cd diplomski
    python train_tau_pretrain_2d.py --config config_tau_pretrain_2d.yaml

Transfer na DoTA (v27):
    Dodaj u config.yaml:
        pretrained_backbone: "checkpoints/tau_pretrain_2d/tau_pretrain_backbone.pt"
"""

import os
import sys
import json
import time
import random
import shutil
import argparse
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CNN_LSTM_BBox, count_parameters, count_all_parameters
from dataloader import DoTANextFrameDataset, create_dataloaders, CATEGORY_MAP

import glob
from torch.utils.data import DataLoader


# ========================================================================= #
#  Focal Loss                                                                 #
# ========================================================================= #

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * bce).mean()


# ========================================================================= #
#  Dataloaders s no_ramp=True                                                 #
# ========================================================================= #

def create_pretrain_dataloaders(config):
    """
    Gradi train/val DataLoader-e za pretrening na TAU-106K.
    Razlika od standardnog create_dataloaders: no_ramp=True za sve splitove
    jer TAU anotacije nemaju rampu – samo binarne oznake 0/1.
    """
    frames_root = config["frames_root"]
    metadata_dir = config["metadata_dir"]
    annotations_dir = config.get("annotations_dir", "")
    yolo_dir = config.get("yolo_detections_dir", None)

    # Učitaj metadata
    all_metadata = {}
    for mf in ["metadata_train.json", "metadata_val.json", "metadata_test.json"]:
        p = os.path.join(metadata_dir, mf)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                all_metadata.update(json.load(f))

    # Splitovi: TAU pretrain koristi train_split.txt za sve, bez posebnog testa
    def _read_split(path):
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    all_vids = _read_split(os.path.join(metadata_dir, "train_split.txt"))
    # Fallback: pokušaj val_split.txt ako postoji
    val_from_file = _read_split(os.path.join(metadata_dir, "val_split.txt"))

    if val_from_file:
        train_vids = all_vids
        val_vids = val_from_file
        print(f"\nSplit (iz fajlova): Train={len(train_vids)}  Val={len(val_vids)}")
    else:
        # 80/20 split
        random.seed(config.get("seed", 42))
        random.shuffle(all_vids)
        split = int(len(all_vids) * 0.95)
        train_vids = all_vids[:split]
        val_vids = all_vids[split:]
        print(f"\nSplit (95/5): Train={len(train_vids)}  Val={len(val_vids)}")

    common = dict(
        seq_len=config.get("seq_len", 24),
        img_size=tuple(config.get("img_size", [224, 224])),
        danger_horizon=config.get("danger_horizon", 8),
        max_objects=config.get("max_objects", 15),
        orig_img_size=tuple(config.get("orig_img_size", [1280, 720])),
        yolo_detections_dir=yolo_dir,
        no_ramp=True,   # TAU anotacije su binarne
    )

    train_ds = DoTANextFrameDataset(
        frames_root, annotations_dir, all_metadata, train_vids,
        phase="train", augment=True,
        stride=config.get("stride", 1),
        stride_safe=config.get("stride_safe", 10),
        stride_danger=config.get("stride_danger", 5),
        **common,
    )
    # Val: koristimo val_stride umjesto stride=1 (inače 400K+ uzoraka → sate validacije)
    val_stride = config.get("val_stride", config.get("stride_safe", 10))
    val_ds = DoTANextFrameDataset(
        frames_root, annotations_dir, all_metadata, val_vids,
        phase="val", augment=False,
        stride=val_stride,
        **common,
    )

    bs = config.get("batch_size", 96)
    nw = config.get("num_workers", 4)
    persist = config.get("persistent_workers", True) and nw > 0
    prefetch = config.get("prefetch_factor", 4) if nw > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=persist, prefetch_factor=prefetch,
    )
    val_bs = config.get("val_batch_size", 128)
    val_nw = config.get("val_num_workers", 0)
    # persistent_workers zahtijeva val_nw > 0; pin_memory s val_nw=0 izbjegava Windows crash
    val_persist = val_nw > 0 and config.get("persistent_workers", True)
    val_prefetch = config.get("prefetch_factor", 4) if val_nw > 0 else None
    val_loader = DataLoader(
        val_ds, batch_size=val_bs, shuffle=False,
        num_workers=val_nw, pin_memory=(val_nw == 0),
        persistent_workers=val_persist, prefetch_factor=val_prefetch,
    )
    return train_loader, val_loader


# ========================================================================= #
#  Trening jedne epohe                                                       #
# ========================================================================= #

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    accum_steps=1, amp_dtype=torch.float16):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0

    pbar = tqdm(loader, desc=f"  Train ep {epoch}")
    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        frames = batch["frames"].to(device, non_blocking=True)
        bboxes = batch["bboxes"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(frames, bboxes, bbox_features, categories, obj_mask, flow=flow)
            loss = criterion(logits, target) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = target.shape[0]
        running_loss += loss.item() * accum_steps * bs
        n_samples += bs

        preds = (torch.sigmoid(logits) > 0.5).float()
        binary_labels = (target >= 0.5).float()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(binary_labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    n = max(n_samples, 1)
    return {
        "loss": running_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


# ========================================================================= #
#  Validacija                                                                 #
# ========================================================================= #

@torch.inference_mode()
def validate(model, loader, criterion, device, amp_dtype=torch.float16):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    n_samples = 0

    for batch in tqdm(loader, desc="  Val"):
        frames = batch["frames"].to(device, non_blocking=True)
        bboxes = batch["bboxes"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(frames, bboxes, bbox_features, categories, obj_mask, flow=flow)
            loss = criterion(logits, target)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        binary_labels = (target >= 0.5).float()
        bs = target.shape[0]
        running_loss += loss.item() * bs
        n_samples += bs
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(binary_labels.cpu().tolist())

    n = max(n_samples, 1)
    metrics = {
        "loss": running_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


# ========================================================================= #
#  Checkpoint                                                                 #
# ========================================================================= #

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_auc, history):
    state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": state_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_auc": best_auc,
        "history": history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt.get("best_auc", 0.0), ckpt.get("history", [])


# ========================================================================= #
#  Grafovi                                                                    #
# ========================================================================= #

def plot_training_curves(history, save_dir):
    epochs = [e["epoch"] for e in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [e["train"]["loss"] for e in history], "b-o", ms=3, label="Train")
    axes[0].plot(epochs, [e["val"]["loss"] for e in history], "r-o", ms=3, label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [e["train"]["f1"] for e in history], "b-o", ms=3, label="Train F1")
    axes[1].plot(epochs, [e["val"]["f1"] for e in history], "r-o", ms=3, label="Val F1")
    axes[1].set_title("F1 Score"); axes[1].set_ylim([0, 1]); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [e["val"].get("auc", 0) for e in history], "g-o", ms=3, label="Val AUC")
    axes[2].set_title("Validation AUC"); axes[2].set_ylim([0, 1]); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Grafovi: {save_dir}/training_curves.png")


# ========================================================================= #
#  Main                                                                       #
# ========================================================================= #

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUređaj: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0)
        print(f"  VRAM: {getattr(mem, 'total_memory', 0) / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # --- Checkpoint folder (fiksni, bez auto-versioning) ---
    ckpt_dir = config.get("checkpoint_dir", "checkpoints/tau_pretrain_2d")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint folder: {ckpt_dir}")

    # --- Spremi snapshot izvornog koda ---
    source_dir = os.path.join(ckpt_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    for _src in ["model.py", "dataloader.py", "train_tau_pretrain_2d.py", "config_tau_pretrain_2d.yaml"]:
        _src_path = os.path.join(_script_dir, _src)
        if os.path.isfile(_src_path):
            shutil.copy2(_src_path, os.path.join(source_dir, _src))
    print(f"  Izvorni kod spremljen: {source_dir}/")

    # --- Podaci ---
    print("\nPriprema podataka (TAU-106K, no_ramp=True)...")
    train_loader, val_loader = create_pretrain_dataloaders(config)
    if len(train_loader.dataset) == 0:
        print("GREŠKA: Prazan dataset!")
        return

    # --- Model (ista arhitektura kao v25, ali freeze_cnn=False) ---
    model = CNN_LSTM_BBox(
        hidden_dim=config.get("hidden_dim", 256),
        lstm_layers=config.get("lstm_layers", 2),
        dropout=config.get("dropout", 0.3),
        lstm_dropout=config.get("lstm_dropout", 0.1),
        backbone=config.get("backbone", "resnet18"),
        freeze_cnn=False,                           # Pretrening: sve otključano
        unfreeze_layer3=False,
        unfreeze_layer4=False,
        max_objects=config.get("max_objects", 15),
        roi_output_size=config.get("roi_output_size", 3),
        roi_feat_dim=config.get("roi_feat_dim", 128),
        bbox_feat_dim=config.get("bbox_feat_dim", 64),
        category_embed_dim=config.get("category_embed_dim", 16),
        use_attention=config.get("use_attention", True),
        attention_heads=config.get("attention_heads", 4),
        use_flow=False,                             # Bez optičkog toka za pretrening
        flow_feat_dim=config.get("flow_feat_dim", 128),
        use_object_interaction=config.get("use_object_interaction", True),
        object_interaction_heads=config.get("object_interaction_heads", 4),
        object_interaction_layers=config.get("object_interaction_layers", 1),
        use_per_object_flow=False,
        bidirectional=config.get("bidirectional", False),
        use_causal_attention=config.get("use_causal_attention", True),
    ).to(device)

    raw_model = model

    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"\nModel: CNN_LSTM_BBox (pretrening na TAU-106K)")
    print(f"  Backbone: {config.get('backbone', 'resnet18')} (potpuno otključan)")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total:     {total:,}")
    print(f"  Frozen:    {total - trainable:,} (trebalo bi biti 0)")

    # --- Loss ---
    gamma = config.get("focal_gamma", 1.0)
    alpha = config.get("focal_alpha", 0.6)
    ls = config.get("label_smoothing", 0.02)
    criterion = FocalBCEWithLogitsLoss(gamma=gamma, alpha=alpha, label_smoothing=ls)
    print(f"\n  Loss: Focal BCE (gamma={gamma}, alpha={alpha}, label_smooth={ls})")

    # --- Optimizer: CNN backbone dobiva manji LR (fine-tune ImageNet težina) ---
    base_lr = config.get("learning_rate", 0.0002)
    cnn_lr = base_lr * config.get("cnn_lr_factor", 0.5)

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(k) for k in ("cnn_early.", "cnn_layer3.", "cnn_layer4.")):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = [{"params": head_params, "lr": base_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cnn_lr})

    optimizer = optim.AdamW(param_groups, weight_decay=config.get("weight_decay", 0.01))
    print(f"  Optimizer: AdamW  base_lr={base_lr:.2e}  cnn_lr={cnn_lr:.2e}  wd={config.get('weight_decay', 0.01)}")

    # --- Scheduler ---
    warmup_epochs = config.get("warmup_epochs", 1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get("cosine_T0", 10),
        T_mult=config.get("cosine_T_mult", 1),
        eta_min=base_lr * 0.01,
    )
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs],
        )
        print(f"  Warmup: {warmup_epochs} epohe (linearan 1%→100% LR)")

    # --- AMP ---
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=(device.type == "cuda" and not use_bf16))
    print(f"  AMP: {'BFloat16' if use_bf16 else 'Float16'}")

    # --- Resume ---
    start_epoch = 1
    best_auc = 0.0
    history = []
    resume_path = config.get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        print(f"\nResume from: {resume_path}")
        start_epoch, best_auc, history = load_checkpoint(
            resume_path, raw_model, optimizer, scheduler, scaler, device,
        )
        start_epoch += 1
        print(f"  Continuing from epoch {start_epoch}, best AUC={best_auc:.4f}")

    # --- Training loop ---
    num_epochs = config.get("num_epochs", 10)
    patience = config.get("early_stopping_patience", 5)
    epochs_no_improve = 0
    save_every = config.get("save_every", 5)
    accum_steps = config.get("gradient_accumulation_steps", 1)

    print(f"\nTrening: {num_epochs} epoha, patience={patience}")
    print("-" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        torch.cuda.empty_cache()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            accum_steps=accum_steps, amp_dtype=amp_dtype,
        )

        torch.cuda.empty_cache()
        val_metrics = validate(model, val_loader, criterion, device, amp_dtype=amp_dtype)

        if warmup_epochs > 0:
            scheduler.step()
        else:
            scheduler.step(epoch)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        entry = {
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
            "time_sec": elapsed,
        }
        history.append(entry)

        print(f"\nEpoha {epoch}/{num_epochs}  ({elapsed:.0f}s)  lr={lr:.2e}")
        print(f"  Train - loss: {train_metrics['loss']:.4f}  "
              f"acc: {train_metrics['accuracy']:.4f}  F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}  "
              f"acc: {val_metrics['accuracy']:.4f}  F1: {val_metrics['f1']:.4f}  "
              f"AUC: {val_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pth"),
                model, optimizer, scheduler, scaler, epoch, best_auc, history,
            )
            print(f"  >>> Novi best model! AUC={best_auc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if epoch % save_every == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth"),
                model, optimizer, scheduler, scaler, epoch, best_auc, history,
            )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping nakon {patience} epoha bez poboljšanja.")
            break

    # --- Spremi final model ---
    save_checkpoint(
        os.path.join(ckpt_dir, "final_model.pth"),
        model, optimizer, scheduler, scaler, epoch, best_auc, history,
    )

    # --- Spremi samo CNN backbone za transfer na DoTA ---
    if config.get("save_backbone_only", True):
        backbone_path = os.path.join(ckpt_dir, "tau_pretrain_backbone.pt")
        state_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        backbone_state = {
            k: v for k, v in state_model.state_dict().items()
            if any(k.startswith(p) for p in ("cnn_early.", "cnn_layer3.", "cnn_layer4."))
        }
        torch.save({"backbone_state_dict": backbone_state, "best_auc": best_auc}, backbone_path)
        print(f"\n  Backbone checkpoint: {backbone_path}")
        print(f"  ({len(backbone_state)} parametarskih tenzora)")
        abs_path = os.path.abspath(backbone_path).replace("\\", "/")
        print(f"\n  Dodaj u config.yaml za v27:")
        print(f'    pretrained_backbone: "{abs_path}"')

    with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nPretrening završen!")
    print(f"  Best AUC (TAU-106K): {best_auc:.4f}")
    print(f"  Checkpointi: {ckpt_dir}")

    plot_training_curves(history, ckpt_dir)


# ========================================================================= #
#  CLI                                                                        #
# ========================================================================= #

def main():
    parser = argparse.ArgumentParser(description="TAU-106K pretrening za 2D CNN+LSTM+BBox model")
    parser.add_argument("--config", default="config_tau_pretrain_2d.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.resume:
        config["resume"] = args.resume

    train(config)


if __name__ == "__main__":
    main()
