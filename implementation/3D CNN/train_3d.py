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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataloader import create_dataloaders
from model_3d import CNN3D_Danger, count_parameters, count_all_parameters
from utils import EMA, mixup_batch, save_checkpoint, load_checkpoint, FocalBCEWithLogitsLoss


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    accum_steps=1, mixup_alpha=0.0, amp_dtype=torch.float16,
                    use_temporal_aux=False, temporal_aux_weight=0.3):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    n_samples = 0

    pbar = tqdm(loader, desc=f"  Train ep {epoch}")
    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        if mixup_alpha > 0:
            batch = mixup_batch(batch, alpha=mixup_alpha)

        assert "frames" in batch, "3D CNN model zahtijeva raw frames!"
        frames = batch["frames"].to(device, non_blocking=True)
        bbox_features = batch["bbox_features"].to(device, non_blocking=True)
        categories = batch["categories"].to(device, non_blocking=True)
        obj_mask = batch["obj_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
        per_frame_t = batch["per_frame_targets"].to(device, non_blocking=True) if "per_frame_targets" in batch else None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            result = model(
                frames, bbox_features=bbox_features,
                categories=categories, obj_mask=obj_mask, flow=flow,
                return_aux=(use_temporal_aux and per_frame_t is not None),
            )
            if isinstance(result, tuple):
                logits, temporal_logits = result
            else:
                logits, temporal_logits = result, None

            main_loss = criterion(logits, target)
            if temporal_logits is not None and per_frame_t is not None:
                T = per_frame_t.shape[1]
                T_prime = temporal_logits.shape[1]
                seg = max(1, T // T_prime)
                aux_targets = torch.stack(
                    [per_frame_t[:, i * seg:(i + 1) * seg].max(dim=1).values
                     for i in range(T_prime)], dim=1
                )
                aux_loss = criterion(temporal_logits.reshape(-1), aux_targets.reshape(-1))
                loss = (main_loss + temporal_aux_weight * aux_loss) / accum_steps
            else:
                loss = main_loss / accum_steps

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


@torch.inference_mode()
def validate(model, loader, criterion, device, amp_dtype=torch.float16):
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

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(
                frames, bbox_features=bbox_features,
                categories=categories, obj_mask=obj_mask, flow=flow,
            )
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


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUređaj: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    ckpt_base = "checkpoints_3d"
    if "_ckpt_dir_override" in config:
        ckpt_dir = config["_ckpt_dir_override"]
        version  = config.get("_version", "?")
    else:
        version = 1
        while os.path.isdir(os.path.join(ckpt_base, f"v{version}")):
            version += 1
        ckpt_dir = os.path.join(ckpt_base, f"v{version}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint folder: {ckpt_dir}")

    source_dir = os.path.join(ckpt_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    for _src_file in ["model_3d.py", "train_3d.py", "evaluate_3d.py", "config_3d.yaml"]:
        _src_path = os.path.join(_script_dir, _src_file)
        if os.path.isfile(_src_path):
            shutil.copy2(_src_path, os.path.join(source_dir, _src_file))
    _parent_dir = os.path.join(_script_dir, "..")
    for _src_file in ["dataloader.py"]:
        _src_path = os.path.join(_parent_dir, _src_file)
        if os.path.isfile(_src_path):
            shutil.copy2(_src_path, os.path.join(source_dir, _src_file))

    print("\nPriprema podataka...")
    assert config.get("cnn_cache_dir") is None, "3D CNN model ne podržava CNN cache!"
    fold_train = config.pop("_fold_train_vids", None)
    fold_val   = config.pop("_fold_val_vids",   None)
    train_loader, val_loader, _ = create_dataloaders(
        config, train_vids=fold_train, val_vids=fold_val,
    )
    if len(train_loader.dataset) == 0:
        print("GREŠKA: Prazan dataset!")
        return

    model = CNN3D_Danger(
        backbone_3d=config.get("backbone_3d", "r2plus1d_18"),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.5),
        freeze_3d=config.get("freeze_3d", False),
        freeze_early=config.get("freeze_early", True),
        freeze_layer3=config.get("freeze_layer3", False),
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
        use_temporal_aux=config.get("use_temporal_aux", False),
    ).to(device)

    raw_model = model

    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"\nModel: {config.get('backbone_3d', 'r2plus1d_18')} - Trainable: {trainable:,} / Total: {total:,}")

    use_focal = config.get("use_focal_loss", True)
    if use_focal:
        gamma = config.get("focal_gamma", 2.0)
        alpha = config.get("focal_alpha", 0.75)
        ls = config.get("label_smoothing", 0.0)
        criterion = FocalBCEWithLogitsLoss(gamma=gamma, alpha=alpha, label_smoothing=ls)
    else:
        criterion = nn.BCEWithLogitsLoss()

    base_lr = config.get("learning_rate", 1e-4)
    cnn_lr = base_lr * config.get("cnn_lr_factor", 0.1)

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

    optimizer = optim.AdamW(param_groups, weight_decay=config.get("weight_decay", 0.02))

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
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=(device.type == 'cuda' and not use_bf16))

    pretrained_backbone = config.get("pretrained_backbone", None)
    if pretrained_backbone and os.path.isfile(pretrained_backbone):
        bb_ckpt = torch.load(pretrained_backbone, map_location=device, weights_only=False)
        bb_state = bb_ckpt.get("backbone_state_dict", bb_ckpt.get("model_state_dict", None))
        if bb_state is not None:
            model_state = raw_model.state_dict()
            backbone_keys = {k: v for k, v in bb_state.items()
                             if any(k.startswith(p) for p in ("stem.", "layer1.", "layer2.", "layer3.", "layer4."))}
            missing = [k for k in backbone_keys if k not in model_state]
            loaded = {k: v for k, v in backbone_keys.items() if k in model_state}
            model_state.update(loaded)
            raw_model.load_state_dict(model_state, strict=False)
            print(f"  Učitano {len(loaded)} backbone parametara")

    start_epoch = 1
    best_auc = 0.0
    history = []

    resume_path = config.get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_auc, history = load_checkpoint(
            resume_path, raw_model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1

    num_epochs = config.get("num_epochs", 25)
    patience = config.get("early_stopping_patience", 7)
    epochs_no_improve = 0
    save_every = config.get("save_every", 5)
    accum_steps = config.get("gradient_accumulation_steps", 1)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    use_temporal_aux = config.get("use_temporal_aux", False)
    temporal_aux_weight = config.get("temporal_aux_weight", 0.3)

    print(f"\nTrening: {num_epochs} epoha, patience={patience}")
    print("-" * 60)

    ema_decay = config.get("ema_decay", 0.0)
    ema_start = config.get("ema_start_epoch", 1)
    ema = None
    if ema_decay > 0:
        ema = EMA(raw_model, decay=ema_decay)

    swa_start = config.get("swa_start_epoch", 0)
    swa_lr = config.get("swa_lr", 5e-5)
    swa_model = None
    swa_scheduler = None
    best_swa_auc = 0.0
    if swa_start > 0:
        swa_model = AveragedModel(raw_model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        torch.cuda.empty_cache()

        _in_swa_phase = swa_model is not None and epoch >= swa_start

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            accum_steps=accum_steps,
            mixup_alpha=mixup_alpha,
            amp_dtype=amp_dtype,
            use_temporal_aux=use_temporal_aux,
            temporal_aux_weight=temporal_aux_weight,
        )

        if ema is not None and epoch >= ema_start:
            ema.update(raw_model)

        torch.cuda.empty_cache()

        if ema is not None and epoch >= ema_start:
            ema.apply_shadow(raw_model)

        val_metrics = validate(model, val_loader, criterion, device, amp_dtype=amp_dtype)

        if ema is not None and epoch >= ema_start:
            ema.restore(raw_model)

        if _in_swa_phase:
            swa_model.update_parameters(raw_model)
            swa_scheduler.step()
        elif warmup_epochs > 0:
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
        print(f"  Train - loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}  F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}  F1: {val_metrics['f1']:.4f}  AUC: {val_metrics['auc']:.4f}")

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

    if ema is not None and ema._initialized:
        ema.apply_shadow(raw_model)
    save_checkpoint(
        os.path.join(ckpt_dir, "final_model.pth"),
        model, optimizer, scheduler, scaler, epoch, best_auc, history,
    )
    if ema is not None and ema._initialized:
        ema.restore(raw_model)
    if swa_model is not None and swa_model.n_averaged > 0:
        swa_model.train()
        for m in swa_model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
                m.num_batches_tracked.zero_()
                m.momentum = None

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="  SWA BN update"):
                frames = batch["frames"].to(device, non_blocking=True)
                bbox_features = batch["bbox_features"].to(device, non_blocking=True)
                categories = batch["categories"].to(device, non_blocking=True)
                obj_mask = batch["obj_mask"].to(device, non_blocking=True)
                flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
                swa_model(frames, bbox_features=bbox_features,
                          categories=categories, obj_mask=obj_mask, flow=flow)

        swa_model.eval()
        swa_val = validate(swa_model, val_loader, criterion, device, amp_dtype=amp_dtype)
        best_swa_auc = swa_val["auc"]
        print(f"  SWA AUC: {best_swa_auc:.4f}  (regular best: {best_auc:.4f})")
        torch.save(
            {"model_state_dict": swa_model.module.state_dict(),
             "swa_auc": best_swa_auc,
             "history": history},
            os.path.join(ckpt_dir, "swa_model.pth"),
        )
        if best_swa_auc > best_auc:
            torch.save(
                {"model_state_dict": swa_model.module.state_dict(),
                 "swa_auc": best_swa_auc},
                os.path.join(ckpt_dir, "best_model_swa.pth"),
            )
            print(f"  >>> SWA bolji od best_model!")
    if config.get("save_backbone_only", False):
        backbone_path = os.path.join(ckpt_dir, "tau_pretrain_backbone.pt")
        state_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        backbone_state = {k: v for k, v in state_model.state_dict().items()
                          if any(k.startswith(p) for p in ("stem.", "layer1.", "layer2.", "layer3.", "layer4."))}
        torch.save({"backbone_state_dict": backbone_state, "best_auc": best_auc}, backbone_path)

    with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    parent_dir = os.path.dirname(ckpt_dir)
    if os.path.basename(ckpt_dir).startswith("fold"):
        fold_id = os.path.basename(ckpt_dir)
        cv_summary_path = os.path.join(parent_dir, "cv_results.json")
        cv_summary = {}
        if os.path.isfile(cv_summary_path):
            with open(cv_summary_path) as f:
                cv_summary = json.load(f)
        cv_summary[fold_id] = {"best_auc": best_auc, "epochs": len(history)}
        fold_results = {k: v for k, v in cv_summary.items() if k.startswith("fold")}
        if len(fold_results) > 1:
            aucs = [v["best_auc"] for v in fold_results.values()]
            cv_summary["mean_auc"] = sum(aucs) / len(aucs)
            cv_summary["std_auc"]  = float((sum((a - cv_summary["mean_auc"])**2 for a in aucs) / len(aucs)) ** 0.5)
        with open(cv_summary_path, "w") as f:
            json.dump(cv_summary, f, indent=2)

    print(f"\nTrening završen! Best AUC: {best_auc:.4f}")
    if swa_model is not None and swa_model.n_averaged > 0:
        print(f"Best SWA AUC: {best_swa_auc:.4f}")
    plot_training_curves(history, ckpt_dir)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_3d.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fold",    type=int, default=-1)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.resume:
        config["resume"] = args.resume

    ckpt_base = "checkpoints_3d"
    if args.version is not None:
        version = args.version
    else:
        version = 1
        while os.path.isdir(os.path.join(ckpt_base, f"v{version}")):
            version += 1
    config["_version"] = version

    if args.fold < 0:
        config["_ckpt_dir_override"] = os.path.join(ckpt_base, f"v{version}")
        train(config)
    else:
        fold      = args.fold
        n_folds   = args.n_folds
        seed      = config.get("seed", 42)
        meta_dir  = config["metadata_dir"]

        split_path = os.path.join(
            os.path.dirname(os.path.abspath(args.config)),
            meta_dir, "train_split.txt",
        )
        with open(split_path, encoding="utf-8") as f:
            all_clips = [l.strip() for l in f if l.strip()]

        import numpy as np
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_clips))
        fold_size = len(all_clips) // n_folds
        val_idx   = set(indices[fold * fold_size: (fold + 1) * fold_size].tolist())
        fold_train = [all_clips[i] for i in indices if i not in val_idx]
        fold_val   = [all_clips[i] for i in indices if i in val_idx]

        print(f"\n{'='*60}")
        print(f"K-FOLD CV  fold={fold}/{n_folds}  train={len(fold_train)}  val={len(fold_val)}")
        print(f"{'='*60}")

        config["_ckpt_dir_override"] = os.path.join(ckpt_base, f"v{version}", f"fold{fold}")
        config["_fold_train_vids"]   = fold_train
        config["_fold_val_vids"]     = fold_val
        train(config)


if __name__ == "__main__":
    main()
