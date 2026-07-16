"""
Trening skripta za 3D CNN model – next-frame predviđanje opasnosti.

Pokretanje:
    cd "3D CNN"
    python train_3d.py --config config_3d.yaml

Razlike u odnosu na 2D CNN + LSTM pristup:
    - 3D CNN (R(2+1)D-18) obrađuje cijeli clip odjednom
    - Nema LSTM za globalne features (3D CNN modelira temporalne uzorke)
    - Nema ROI Align (3D feature mape su spatiotemporalne)
    - Objektni tok koristi samo numeričke bbox značajke + attention
    - Manji batch size (3D CNN troši više VRAM-a)
"""

import os
import sys
import json
import time
import random
import shutil
import argparse
from datetime import datetime

# v3: smanji VRAM fragmentaciju — pomaže pri prijelazu train→val (val_batch=64)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

from torch.utils.data import Subset

# Import iz parent direktorija
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataloader import create_dataloaders
from model_3d import CNN3D_Danger, count_parameters, count_all_parameters


# ========================================================================= #
#  Focal Loss                                                                 #
# ========================================================================= #

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss za binarnu klasifikaciju (radi s logitima).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

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
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ========================================================================= #
#  EMA (Exponential Moving Average)                                           #
# ========================================================================= #

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialized = False

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if not self._initialized:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        self._initialized = True

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ========================================================================= #
#  Mixup za sekvence                                                          #
# ========================================================================= #

def mixup_batch(batch, alpha=0.2):
    if alpha <= 0:
        return batch
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)
    B = batch["frames"].size(0)
    indices = torch.randperm(B)

    mixed = {}
    for key in ["frames", "bboxes", "bbox_features"]:
        if key in batch:
            mixed[key] = lam * batch[key] + (1 - lam) * batch[key][indices]
    mixed["categories"] = batch["categories"]
    mixed["obj_mask"] = batch["obj_mask"]
    mixed["target"] = lam * batch["target"] + (1 - lam) * batch["target"][indices]
    if "flow" in batch:
        mixed["flow"] = lam * batch["flow"] + (1 - lam) * batch["flow"][indices]
    if "per_frame_targets" in batch:
        mixed["per_frame_targets"] = lam * batch["per_frame_targets"] + (1 - lam) * batch["per_frame_targets"][indices]
    return mixed


# ========================================================================= #
#  Trening jedne epohe                                                       #
# ========================================================================= #

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    accum_steps=1, mixup_alpha=0.0, amp_dtype=torch.float16):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0

    pbar = tqdm(loader, desc=f"  Train ep {epoch}")
    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        if mixup_alpha > 0:
            batch = mixup_batch(batch, alpha=mixup_alpha)

        # 3D model zahtijeva raw frames (ne CNN cache)
        assert "frames" in batch, "3D CNN model zahtijeva raw frames! Ne koristite cnn_cache_dir."
        frames = batch["frames"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(
                frames, bbox_features=bbox_features,
                categories=categories, obj_mask=obj_mask, flow=flow,
            )
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
    metrics = {
        "loss": running_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    return metrics


# ========================================================================= #
#  Validacija                                                                 #
# ========================================================================= #

@torch.inference_mode()
def validate(model, loader, criterion, device, amp_dtype=torch.float16, micro_batch=16):
    """
    micro_batch: forward kroz 3D CNN ide u komadima ove veličine (VRAM safety).
    Dataloader i dalje vuče val_batch_size=64, ali GPU obrađuje 16 odjednom —
    izbjegava OOM na conv1/layer1 (1.29 GiB kontinuirani allocation pri batchu 64).
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    n_samples = 0

    for batch in tqdm(loader, desc="  Val"):
        assert "frames" in batch, "3D CNN model zahtijeva raw frames!"
        frames = batch["frames"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None

        bs = target.shape[0]
        # Mikro-batching: spliti batch u komade da izbjegnemo OOM
        logit_chunks = []
        for s in range(0, bs, micro_batch):
            e = min(s + micro_batch, bs)
            f_chunk = flow[s:e] if flow is not None else None
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logit_chunks.append(model(
                    frames[s:e],
                    bbox_features=bbox_features[s:e],
                    categories=categories[s:e],
                    obj_mask=obj_mask[s:e],
                    flow=f_chunk,
                ))
        logits = torch.cat(logit_chunks, dim=0)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            loss = criterion(logits, target)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        binary_labels = (target >= 0.5).float()

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
#  Checkpoint save / load                                                     #
# ========================================================================= #

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_auc, history):
    state_model = model._orig_mod if hasattr(model, '_orig_mod') else model
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
#  Main trening petlja                                                        #
# ========================================================================= #

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUređaj: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0)
        total_mem = getattr(mem, 'total_mem', getattr(mem, 'total_memory', 0))
        print(f"  VRAM: {total_mem / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print(f"  cudnn.benchmark = True, TF32 = True")

    # --- Checkpoint folder (auto-increment) ---
    ckpt_base = "checkpoints_3d"
    version = 1
    while os.path.isdir(os.path.join(ckpt_base, f"v{version}")):
        version += 1
    ckpt_dir = os.path.join(ckpt_base, f"v{version}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint folder: {ckpt_dir}")

    # --- Spremi snapshot izvornog koda i konfiguracije ---
    source_dir = os.path.join(ckpt_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    for _src_file in ["model_3d.py", "train_3d.py", "evaluate_3d.py", "config_3d.yaml"]:
        _src_path = os.path.join(_script_dir, _src_file)
        if os.path.isfile(_src_path):
            shutil.copy2(_src_path, os.path.join(source_dir, _src_file))
    # Kopiraj i parent dataloader za potpuni snapshot
    _parent_dir = os.path.join(_script_dir, "..")
    for _src_file in ["dataloader.py"]:
        _src_path = os.path.join(_parent_dir, _src_file)
        if os.path.isfile(_src_path):
            shutil.copy2(_src_path, os.path.join(source_dir, _src_file))
    print(f"  Izvorni kod spremljen: {source_dir}/")

    # --- Data ---
    print("\nPriprema podataka...")
    # 3D model NE koristi CNN cache – uvijek učitava raw frames
    assert config.get("cnn_cache_dir") is None, \
        "3D CNN model ne podržava CNN cache! Uklonite 'cnn_cache_dir' iz konfiguracije."
    train_loader, val_loader, _ = create_dataloaders(config)
    if len(train_loader.dataset) == 0:
        print("GREŠKA: Prazan dataset!")
        return

    # --- Model ---
    model = CNN3D_Danger(
        backbone_3d=config.get("backbone_3d", "r2plus1d_18"),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.5),
        freeze_3d=config.get("freeze_3d", False),
        freeze_early=config.get("freeze_early", True),
        use_grad_checkpoint=config.get("use_grad_checkpoint", True),
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

    raw_model = model

    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"\nModel: CNN3D_Danger (3D CNN backbone)")
    print(f"  Backbone: {config.get('backbone_3d', 'r2plus1d_18')}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total:     {total:,}")
    print(f"  Frozen:    {total - trainable:,}")
    print(f"  Hidden dim: {config.get('hidden_dim', 256)}")
    print(f"  Object branch: {config.get('use_object_branch', True)}")
    print(f"  Object interaction: {config.get('use_object_interaction', False)}")
    print(f"  Freeze 3D: {config.get('freeze_3d', False)}")
    print(f"  Freeze early: {config.get('freeze_early', True)}")
    print(f"  Grad checkpoint: {config.get('use_grad_checkpoint', True)}")
    print(f"  Batch size: {config.get('batch_size', 16)} (effective: {config.get('batch_size', 16) * config.get('gradient_accumulation_steps', 1)})")
    print(f"  Image size: {config.get('img_size', [112, 112])}")

    # --- Loss ---
    use_focal = config.get("use_focal_loss", True)
    if use_focal:
        gamma = config.get("focal_gamma", 2.0)
        alpha = config.get("focal_alpha", 0.75)
        ls = config.get("label_smoothing", 0.0)
        criterion = FocalBCEWithLogitsLoss(gamma=gamma, alpha=alpha, label_smoothing=ls)
        print(f"\n  Loss: Focal BCE (gamma={gamma}, alpha={alpha}, label_smooth={ls})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"\n  Loss: Standard BCE")

    # --- Optimizer ---
    base_lr = config.get("learning_rate", 1e-4)
    cnn_lr = base_lr * config.get("cnn_lr_factor", 0.1)

    # Param grupe: head (base_lr), 3D backbone later layers (cnn_lr)
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ["layer3", "layer4"]):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = [
        {"params": head_params, "lr": base_lr},
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cnn_lr})
        print(f"  Backbone LR: {cnn_lr:.2e} ({config.get('cnn_lr_factor', 0.1)}x base)")

    optimizer = optim.AdamW(param_groups, weight_decay=config.get("weight_decay", 0.02))

    # --- Scheduler ---
    warmup_epochs = config.get("warmup_epochs", 0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get("cosine_T0", 10),
        T_mult=config.get("cosine_T_mult", 2),
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
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=(device.type == 'cuda' and not use_bf16))
    print(f"  AMP: {'BFloat16' if use_bf16 else 'Float16'}")

    # --- Resume ---
    start_epoch = 1
    best_auc = 0.0
    history = []

    resume_path = config.get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        print(f"\nResume from: {resume_path}")
        start_epoch, best_auc, history = load_checkpoint(
            resume_path, raw_model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        print(f"  Continuing from epoch {start_epoch}, best AUC={best_auc:.4f}")

    # --- Training loop ---
    num_epochs = config.get("num_epochs", 25)
    patience = config.get("early_stopping_patience", 7)
    epochs_no_improve = 0
    save_every = config.get("save_every", 5)
    accum_steps = config.get("gradient_accumulation_steps", 1)
    mixup_alpha = config.get("mixup_alpha", 0.0)

    print(f"\nTrening: {num_epochs} epoha, patience={patience}")
    if accum_steps > 1:
        print(f"  Gradient accumulation: {accum_steps} koraka")
    if mixup_alpha > 0:
        print(f"  Mixup: alpha={mixup_alpha}")
    print("-" * 60)

    # --- EMA ---
    ema_decay = config.get("ema_decay", 0.0)
    ema_start = config.get("ema_start_epoch", 1)
    ema = None
    if ema_decay > 0:
        ema = EMA(raw_model, decay=ema_decay)
        print(f"  EMA: decay={ema_decay}, start epoch {ema_start}")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        torch.cuda.empty_cache()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            accum_steps=accum_steps,
            mixup_alpha=mixup_alpha,
            amp_dtype=amp_dtype,
        )

        if ema is not None and epoch >= ema_start:
            ema.update(raw_model)

        torch.cuda.empty_cache()

        if ema is not None and epoch >= ema_start:
            ema.apply_shadow(raw_model)

        val_metrics = validate(model, val_loader, criterion, device, amp_dtype=amp_dtype)

        if ema is not None and epoch >= ema_start:
            ema.restore(raw_model)

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

        # Best model
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            if ema is not None and epoch >= ema_start:
                ema.apply_shadow(raw_model)
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pth"),
                model, optimizer, scheduler, scaler, epoch, best_auc, history,
            )
            if ema is not None and epoch >= ema_start:
                ema.restore(raw_model)
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

    # Save final
    if ema is not None and ema._initialized:
        ema.apply_shadow(raw_model)
    save_checkpoint(
        os.path.join(ckpt_dir, "final_model.pth"),
        model, optimizer, scheduler, scaler, epoch, best_auc, history,
    )
    if ema is not None and ema._initialized:
        ema.restore(raw_model)

    with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTrening završen!")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Checkpointi: {ckpt_dir}")

    plot_training_curves(history, ckpt_dir)


# ========================================================================= #
#  Grafovi                                                                    #
# ========================================================================= #

def plot_training_curves(history, save_dir):
    epochs = [e["epoch"] for e in history]
    train_loss = [e["train"]["loss"] for e in history]
    val_loss = [e["val"]["loss"] for e in history]
    train_f1 = [e["train"]["f1"] for e in history]
    val_f1 = [e["val"]["f1"] for e in history]
    val_auc = [e["val"].get("auc", 0) for e in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, "b-o", ms=3, label="Train")
    axes[0].plot(epochs, val_loss, "r-o", ms=3, label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_f1, "b-o", ms=3, label="Train F1")
    axes[1].plot(epochs, val_f1, "r-o", ms=3, label="Val F1")
    axes[1].set_title("F1 Score")
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_auc, "g-o", ms=3, label="Val AUC")
    axes[2].set_title("Validation AUC")
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Grafovi: {save_dir}/training_curves.png")


# ========================================================================= #
#  CLI                                                                        #
# ========================================================================= #

def main():
    parser = argparse.ArgumentParser(description="Train 3D CNN danger prediction model")
    parser.add_argument("--config", default="config_3d.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.resume:
        config["resume"] = args.resume

    train(config)


if __name__ == "__main__":
    main()
