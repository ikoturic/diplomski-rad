"""
Trening skripta za CNN+LSTM+BBox model – next-frame predviđanje opasnosti.

Pokretanje:
    python train.py --config config.yaml

Značajke:
    - Focal Loss za neuravnoteženi dataset
    - Cosine Annealing LR scheduler
    - AUC-based best model selection
    - Automatic checkpoint versioning
    - Mixed precision training (AMP)
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
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Subset

from model import CNN_LSTM_BBox, count_parameters, count_all_parameters
from dataloader import create_dataloaders


# ========================================================================= #
#  EMA (Exponential Moving Average)                                           #
# ========================================================================= #

class EMA:
    """
    Exponential Moving Average za model parametre.
    Gladi težine svakog batcha: ema_param = decay * ema_param + (1 - decay) * param
    Shadow se inicijalizira lazy — tek pri prvom update() pozivu.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialized = False
        self._param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if not self._initialized:
                    # Lazy init: kopiraj trenutne (trenirane) težine kao početni shadow
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        self._initialized = True

    def apply_shadow(self, model):
        """Zamijeni model parametre s EMA vrijednostima (za evaluaciju)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Vrati originalne parametre nakon evaluacije."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ========================================================================= #
#  Mixup za sekvence                                                          #
# ========================================================================= #

def mixup_batch(batch, alpha=0.2):
    """
    Mixup augmentacija za sekvencijalne podatke.
    Miješa dva nasumična uzorka iz iste batch-a s težinom lambda.
    """
    if alpha <= 0:
        return batch

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)  # Osiguraj da je lam >= 0.5

    B = batch["frames"].size(0) if "frames" in batch else batch["feat_maps"].size(0)
    indices = torch.randperm(B)

    mixed = {}
    for key in ["bboxes", "bbox_features"]:
        mixed[key] = lam * batch[key] + (1 - lam) * batch[key][indices]
    if "frames" in batch:
        mixed["frames"] = lam * batch["frames"] + (1 - lam) * batch["frames"][indices]
    if "feat_maps" in batch:
        mixed["feat_maps"] = lam * batch["feat_maps"] + (1 - lam) * batch["feat_maps"][indices]
        mixed["global_feats"] = lam * batch["global_feats"] + (1 - lam) * batch["global_feats"][indices]
    # Kategorije i maske: uzmi primarni uzorak (ne može se interpolirati)
    mixed["categories"] = batch["categories"]
    mixed["obj_mask"] = batch["obj_mask"]
    # Target: interpoliraj
    mixed["target"] = lam * batch["target"] + (1 - lam) * batch["target"][indices]
    # Flow: interpoliraj ako postoji
    if "flow" in batch:
        mixed["flow"] = lam * batch["flow"] + (1 - lam) * batch["flow"][indices]
    # Per-frame targets: interpoliraj ako postoji (multi-timestep loss)
    if "per_frame_targets" in batch:
        mixed["per_frame_targets"] = lam * batch["per_frame_targets"] + (1 - lam) * batch["per_frame_targets"][indices]

    return mixed


# ========================================================================= #
#  Focal Loss                                                                 #
# ========================================================================= #

class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss za binarnu klasifikaciju (radi s logitima).

    Smanjuje doprinos "laganih" primjera (normalna vožnja) i pojačava
    doprinos "teških" primjera (tranzicija normalno → opasno).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Label smoothing: targets 0→ε, 1→1-ε  (sprječava overconfidence)
    """

    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing: 0 → ε, 1 → 1-ε
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
#  Trening jedne epohe                                                       #
# ========================================================================= #

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, accum_steps=1, mixup_alpha=0.0, amp_dtype=torch.float16, aux_loss_weight=0.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0
    use_multi_timestep = aux_loss_weight > 0

    pbar = tqdm(loader, desc=f"  Train ep {epoch}")
    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        # Mixup augmentacija
        if mixup_alpha > 0:
            batch = mixup_batch(batch, alpha=mixup_alpha)

        frames = batch["frames"].to(device, non_blocking=True) if "frames" in batch else None
        bboxes = batch["bboxes"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
        obj_flow = batch["obj_flow"].to(device, non_blocking=True) if "obj_flow" in batch else None
        cached_fm = batch["feat_maps"].to(device, non_blocking=True) if "feat_maps" in batch else None
        cached_gf = batch["global_feats"].to(device, non_blocking=True) if "feat_maps" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            if use_multi_timestep:
                # Multi-timestep: predikcija na svakom timestep-u
                all_logits = model(frames, bboxes, bbox_features, categories, obj_mask,
                                   flow=flow, obj_flow=obj_flow, return_all_timesteps=True,
                                   cached_feat_maps=cached_fm, cached_global_feats=cached_gf)  # (B, T)
                per_frame_targets = batch["per_frame_targets"].to(device, non_blocking=True)  # (B, T)

                # Glavni loss: zadnji timestep
                main_loss = criterion(all_logits[:, -1], target)
                # Auxiliary loss: prosjek svih timestepova
                aux_loss = criterion(all_logits.reshape(-1), per_frame_targets.reshape(-1))
                loss = (main_loss + aux_loss_weight * aux_loss) / accum_steps

                logits = all_logits[:, -1]  # za metrike koristimo zadnji timestep
            else:
                logits = model(frames, bboxes, bbox_features, categories, obj_mask, flow=flow, obj_flow=obj_flow,
                               cached_feat_maps=cached_fm, cached_global_feats=cached_gf)
                loss = criterion(logits, target) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        bs = target.shape[0]
        running_loss += loss.item() * accum_steps * bs  # Undo /accum_steps
        n_samples += bs

        preds = (torch.sigmoid(logits) > 0.5).float()
        binary_labels = (target >= 0.5).float()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(binary_labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

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
def validate(model, loader, criterion, device, amp_dtype=torch.float16):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    n_samples = 0

    for batch in tqdm(loader, desc="  Val"):
        frames = batch["frames"].to(device, non_blocking=True) if "frames" in batch else None
        bboxes = batch["bboxes"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
        obj_flow = batch["obj_flow"].to(device, non_blocking=True) if "obj_flow" in batch else None
        cached_fm = batch["feat_maps"].to(device, non_blocking=True) if "feat_maps" in batch else None
        cached_gf = batch["global_feats"].to(device, non_blocking=True) if "feat_maps" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(frames, bboxes, bbox_features, categories, obj_mask, flow=flow, obj_flow=obj_flow,
                           cached_feat_maps=cached_fm, cached_global_feats=cached_gf)
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
#  Checkpoint save / load                                                     #
# ========================================================================= #

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_auc, history):
    # Ako je model kompajliran s torch.compile, spremi originalni model
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
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Optimizacija za fiksne input dimenzije
        torch.backends.cudnn.benchmark = True
        # TF32 za brže matmul i konvolucije na Ampere+ GPU (RTX 30xx/40xx/50xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # BFloat16 je stabilniji od Float16 na novijim GPU-ovima
        torch.set_float32_matmul_precision('high')
        print(f"  cudnn.benchmark = True")
        print(f"  TF32 matmul + cudnn = True")
        print(f"  float32_matmul_precision = high")

    # --- Checkpoint folder (auto-increment) ---
    ckpt_base = config.get("checkpoint_dir", "checkpoints")
    if ckpt_base.startswith("checkpoints"):
        version = 1
        while os.path.isdir(os.path.join("checkpoints", f"v{version}")):
            version += 1
        ckpt_dir = os.path.join("checkpoints", f"v{version}")
    else:
        ckpt_dir = ckpt_base
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint folder: {ckpt_dir}")

    # --- Spremi snapshot izvornog koda i konfiguracije ---
    source_dir = os.path.join(ckpt_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    for _src_file in ["model.py", "dataloader.py", "train.py", "evaluate.py", "config.yaml", "visualize.py"]:
        _src_path = os.path.join(_script_dir, _src_file)
        if os.path.isfile(_src_path):
            shutil.copy2(_src_path, os.path.join(source_dir, _src_file))
    print(f"  Izvorni kod spremljen: {source_dir}/")

    # --- Data ---
    print("\nPriprema podataka...")
    train_loader, val_loader, _ = create_dataloaders(config)
    if len(train_loader.dataset) == 0:
        print("GREŠKA: Prazan dataset!")
        return

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

    # Referenca na originalni model za checkpoint (torch.compile mijenja state_dict ključeve)
    raw_model = model

    # torch.compile: na Windows bez Triton-a ne donosi ubrzanje, preskačemo
    # Na Linux-u otkomenirati sljedeće:
    # if hasattr(torch, 'compile') and device.type == 'cuda':
    #     print("\ntorch.compile: kompajliram model...")
    #     model = torch.compile(model)
    #     print("  Model kompajliran (Inductor backend)")

    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"\nModel: CNN_LSTM_BBox (next-frame prediction)")
    print(f"  Backbone: {config.get('backbone', 'resnet50')}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total:     {total:,}")
    print(f"  Frozen:    {total - trainable:,}")
    print(f"  Hidden dim: {config.get('hidden_dim', 256)}")
    print(f"  LSTM layers: {config.get('lstm_layers', 1)}")
    print(f"  Max objects: {config.get('max_objects', 10)}")
    print(f"  Temporal Attention: {config.get('use_attention', True)}")
    print(f"  Danger horizon: {config.get('danger_horizon', 10)}")
    print(f"  Fine-tune layer3: {config.get('unfreeze_layer3', False)}")
    print(f"  Fine-tune layer4: {config.get('unfreeze_layer4', False)}")
    print(f"  Optical flow: {config.get('use_optical_flow', False)}")
    if config.get("use_per_object_flow", False):
        print(f"  Per-object flow: pool={config.get('obj_flow_pool_size', 5)}, dim={config.get('obj_flow_feat_dim', 128)}")

    # --- Učitaj pre-trenirani backbone (TAU-106K → DoTA transfer) ---
    pretrained_backbone = config.get("pretrained_backbone", None)
    if pretrained_backbone and os.path.isfile(pretrained_backbone):
        print(f"\nUčitavam pre-trenirani backbone: {pretrained_backbone}")
        bb_ckpt = torch.load(pretrained_backbone, map_location=device, weights_only=False)
        bb_state = bb_ckpt.get("backbone_state_dict", bb_ckpt.get("model_state_dict", None))
        if bb_state is not None:
            model_state = raw_model.state_dict()
            backbone_keys = {k: v for k, v in bb_state.items()
                             if any(k.startswith(p) for p in ("cnn_early.", "cnn_layer3.", "cnn_layer4."))}
            loaded = {k: v for k, v in backbone_keys.items() if k in model_state}
            missing = [k for k in backbone_keys if k not in model_state]
            model_state.update(loaded)
            raw_model.load_state_dict(model_state, strict=False)
            print(f"  Učitano {len(loaded)} backbone parametara, preskočeno {len(missing)}")
        else:
            print("  [WARN] backbone_state_dict nije pronađen u checkpointu")
    elif pretrained_backbone:
        print(f"  [WARN] pretrained_backbone nije pronađen: {pretrained_backbone}")

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
    base_lr = config.get("learning_rate", 3e-4)
    param_groups = [
        {"params": model.lstm.parameters(), "lr": base_lr},
        {"params": model.classifier.parameters(), "lr": base_lr},
        {"params": model.roi_proj.parameters(), "lr": base_lr},
        {"params": model.bbox_encoder.parameters(), "lr": base_lr},
        {"params": model.cat_embed.parameters(), "lr": base_lr},
        {"params": model.obj_attention.parameters(), "lr": base_lr},
        {"params": model.motion_proj.parameters(), "lr": base_lr},
        {"params": [model.h0, model.c0], "lr": base_lr},
    ]
    if model.use_attention:
        param_groups.append({"params": model.temp_attn.parameters(), "lr": base_lr})
        param_groups.append({"params": model.attn_norm.parameters(), "lr": base_lr})
    if model.use_flow:
        param_groups.append({"params": model.flow_encoder.parameters(), "lr": base_lr})
    if model.use_object_interaction:
        param_groups.append({"params": model.obj_interaction.parameters(), "lr": base_lr})

    # CNN late layers (ako nisu zamrznuti)
    cnn_lr = base_lr * config.get("cnn_lr_factor", 0.1)
    if any(p.requires_grad for p in model.cnn_layer3.parameters()):
        param_groups.append({"params": model.cnn_layer3.parameters(), "lr": cnn_lr})
    if any(p.requires_grad for p in model.cnn_layer4.parameters()):
        param_groups.append({"params": model.cnn_layer4.parameters(), "lr": cnn_lr})
    # Global projection (za ResNet50: 2048→512)
    if hasattr(model, 'global_proj') and not isinstance(model.global_proj, nn.Identity):
        param_groups.append({"params": model.global_proj.parameters(), "lr": base_lr})

    optimizer = optim.AdamW(param_groups, weight_decay=config.get("weight_decay", 1e-3))

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get("cosine_T0", 10),
        T_mult=config.get("cosine_T_mult", 2),
        eta_min=base_lr * 0.01,
    )

    # --- Warmup scheduler ---
    warmup_epochs = config.get("warmup_epochs", 0)
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

    # Koristi BFloat16 ako GPU podržava (Ampere+), inače Float16
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # GradScaler nije potreban za BFloat16 (dovoljno širok dinamički raspon)
    scaler = GradScaler(enabled=(device.type == 'cuda' and not use_bf16))
    if use_bf16:
        print(f"  AMP: BFloat16 (bez GradScaler-a)")
    else:
        print(f"  AMP: Float16 (s GradScaler-om)")

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
    num_epochs = config.get("num_epochs", 40)
    patience = config.get("early_stopping_patience", 12)
    epochs_no_improve = 0
    save_every = config.get("save_every", 5)

    print(f"\nTrening: {num_epochs} epoha, patience={patience}")
    print("-" * 60)

    # --- Mixup ---
    mixup_alpha = config.get("mixup_alpha", 0.0)
    if mixup_alpha > 0:
        print(f"  Mixup: alpha={mixup_alpha}")

    # --- Multi-timestep auxiliary loss ---
    aux_loss_weight = config.get("aux_loss_weight", 0.0)
    if aux_loss_weight > 0:
        print(f"  Multi-timestep aux loss: weight={aux_loss_weight}")

    # --- SWA (Stochastic Weight Averaging) ---
    swa_start = config.get("swa_start_epoch", 0)  # 0 = disabled
    swa_model = None
    swa_scheduler = None
    if swa_start > 0:
        swa_model = AveragedModel(model)
        swa_lr = config.get("swa_lr", base_lr * 0.5)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=2)
        print(f"  SWA: start epoch {swa_start}, lr={swa_lr:.2e}")

    # --- Epoch subsampling ---
    subsample_ratio = config.get("epoch_subsample", 1.0)
    if subsample_ratio < 1.0:
        print(f"  Epoch subsampling: {subsample_ratio*100:.0f}% uzoraka po epohi")

    # --- EMA (Exponential Moving Average) ---
    ema_decay = config.get("ema_decay", 0.0)
    ema_start = config.get("ema_start_epoch", 1)
    ema = None
    if ema_decay > 0:
        ema = EMA(raw_model, decay=ema_decay)
        print(f"  EMA: decay={ema_decay}, start epoch {ema_start}")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        torch.cuda.empty_cache()  # Clear fragmented VRAM before each epoch

        # Svake epohe koristi drugi nasumični podskup trening podataka
        if subsample_ratio < 1.0:
            full_ds = train_loader.dataset
            n_total = len(full_ds)
            n_sub = max(1, int(n_total * subsample_ratio))
            indices = random.sample(range(n_total), n_sub)
            subset = Subset(full_ds, indices)
            epoch_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=config.get("num_workers", 0),
                pin_memory=True,
                persistent_workers=config.get("persistent_workers", False) and config.get("num_workers", 0) > 0,
                prefetch_factor=config.get("prefetch_factor", 2) if config.get("num_workers", 0) > 0 else None,
            )
        else:
            epoch_loader = train_loader

        train_metrics = train_one_epoch(
            model, epoch_loader, criterion, optimizer, scaler, device, epoch,
            accum_steps=config.get("gradient_accumulation_steps", 1),
            mixup_alpha=mixup_alpha,
            amp_dtype=amp_dtype,
            aux_loss_weight=config.get("aux_loss_weight", 0.0),
        )
        # EMA: ažuriraj shadow parametre nakon treninga
        if ema is not None and epoch >= ema_start:
            ema.update(raw_model)

        # Oslobodi VRAM cache prije validacije da izbjegnemo OOM u pin_memory
        torch.cuda.empty_cache()

        # EMA: koristi shadow parametre za validaciju
        if ema is not None and epoch >= ema_start:
            ema.apply_shadow(raw_model)

        val_metrics = validate(model, val_loader, criterion, device, amp_dtype=amp_dtype)

        # EMA: vrati originalne parametre za nastavak treninga
        if ema is not None and epoch >= ema_start:
            ema.restore(raw_model)

        # SWA: update averaged model i koristi SWA scheduler
        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif warmup_epochs > 0:
            scheduler.step()  # SequentialLR ne prima epoch argument
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

        print(
            f"\nEpoha {epoch}/{num_epochs}  ({elapsed:.0f}s)  lr={lr:.2e}"
        )
        print(
            f"  Train - loss: {train_metrics['loss']:.4f}  "
            f"acc: {train_metrics['accuracy']:.4f}  "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"  Val   - loss: {val_metrics['loss']:.4f}  "
            f"acc: {val_metrics['accuracy']:.4f}  "
            f"F1: {val_metrics['f1']:.4f}  "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        # Best model (po AUC-u)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            # Spremi EMA shadow parametre ako su dostupni (bolji za inference)
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

        # Spremi history nakon svake epohe (da se ne izgubi pri Ctrl+C)
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

    # Save final (s EMA ako je aktivan)
    if ema is not None and epoch >= ema_start:
        ema.apply_shadow(raw_model)
    save_checkpoint(
        os.path.join(ckpt_dir, "final_model.pth"),
        model, optimizer, scheduler, scaler, epoch, best_auc, history,
    )
    if ema is not None and epoch >= ema_start:
        ema.restore(raw_model)

    # --- SWA: update BN statistika i spremi SWA model ---
    if swa_model is not None and epoch >= swa_start:
        print("\nSWA: Ažuriram BatchNorm statistike...")
        # Custom BN update jer naš DataLoader vraća dict, ne tenzore
        swa_model.train()
        with torch.no_grad():
            # Reset BN statistika
            for module in swa_model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.running_mean.zero_()
                    module.running_var.fill_(1)
                    module.num_batches_tracked.zero_()
            # Forward pass kroz train podatke za BN statistike
            for batch in tqdm(train_loader, desc="  SWA BN update"):
                frames = batch["frames"].to(device, non_blocking=True)
                bboxes = batch["bboxes"].to(device, non_blocking=True)
                bbox_features = batch["bbox_features"].to(device, non_blocking=True)
                categories = batch["categories"].to(device, non_blocking=True)
                obj_mask = batch["obj_mask"].to(device, non_blocking=True)
                flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
                obj_flow = batch["obj_flow"].to(device, non_blocking=True) if "obj_flow" in batch else None
                swa_model(frames, bboxes, bbox_features, categories, obj_mask, flow=flow, obj_flow=obj_flow)

        # Validiraj SWA model
        torch.cuda.empty_cache()
        swa_val = validate(swa_model, val_loader, criterion, device)
        swa_auc = swa_val["auc"]
        print(f"  SWA Val AUC: {swa_auc:.4f} (vs best regular: {best_auc:.4f})")

        # Spremi SWA model (kopiraj weighte u originalni model za kompatibilnost)
        swa_state = swa_model.module.state_dict()
        swa_save_model = CNN_LSTM_BBox(
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
        )
        swa_save_model.load_state_dict(swa_state)
        torch.save({
            "epoch": epoch,
            "model_state_dict": swa_state,
            "best_auc": max(best_auc, swa_auc),
            "swa_auc": swa_auc,
            "history": history,
        }, os.path.join(ckpt_dir, "swa_model.pth"))
        print(f"  SWA model spremljen: {ckpt_dir}/swa_model.pth")

        # Ako je SWA bolji, spremi i kao best
        if swa_auc > best_auc:
            torch.save({
                "epoch": epoch,
                "model_state_dict": swa_state,
                "best_auc": swa_auc,
                "swa_auc": swa_auc,
                "history": history,
            }, os.path.join(ckpt_dir, "best_model.pth"))
            best_auc = swa_auc
            print(f"  >>> SWA je novi best model! AUC={swa_auc:.4f}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.resume:
        config["resume"] = args.resume

    train(config)


if __name__ == "__main__":
    main()
