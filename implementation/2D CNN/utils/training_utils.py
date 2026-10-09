"""
Utility klase i funkcije za trening modela.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


# ========================================================================= #
#  Exponential Moving Average (EMA)                                          #
# ========================================================================= #

class EMA:
    """
    Exponential Moving Average za model parametre.
    
    Gladi težine svakog batcha: ema_param = decay * ema_param + (1 - decay) * param
    Shadow se inicijalizira lazy — tek pri prvom update() pozivu.
    
    Args:
        model: Model čiji parametri se prate
        decay: Decay faktor (obično 0.999)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialized = False
        self._param_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Ažurira shadow parametre s trenutnim parametrima modela.
        
        Args:
            model: Model s trenutnim parametrima
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                if not self._initialized:
                    # Lazy init: kopiraj trenutne (trenirane) težine kao početni shadow
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )
        self._initialized = True

    def apply_shadow(self, model: nn.Module) -> None:
        """
        Zamijeni model parametre s EMA vrijednostima (za evaluaciju).
        
        Args:
            model: Model čiji parametri se zamjenjuju
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """
        Vrati originalne parametre nakon evaluacije.
        
        Args:
            model: Model čiji parametri se vraćaju
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ========================================================================= #
#  Mixup Augmentacija                                                         #
# ========================================================================= #

def mixup_batch(batch: Dict[str, Any], alpha: float = 0.2) -> Dict[str, Any]:
    """
    Mixup augmentacija za sekvencijalne podatke.
    
    Miješa dva nasumična uzorka iz iste batch-a s težinom lambda.
    
    Args:
        batch: Batch podataka (dict s tensorima)
        alpha: Beta distribucija parametar za lambda
        
    Returns:
        Mixed batch
    """
    if alpha <= 0:
        return batch

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)  # Osiguraj da je lam >= 0.5

    B = batch["frames"].size(0) if "frames" in batch else batch["feat_maps"].size(0)
    indices = torch.randperm(B)

    mixed = {}
    
    # Interpoliraj kontinuirane podatke
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
        
    # Per-frame targets: interpoliraj ako postoji
    if "per_frame_targets" in batch:
        mixed["per_frame_targets"] = (
            lam * batch["per_frame_targets"] + 
            (1 - lam) * batch["per_frame_targets"][indices]
        )

    return mixed


# ========================================================================= #
#  Model Checkpoint Utilities                                                 #
# ========================================================================= #

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_auc: float,
    history: list
) -> None:
    """
    Sprema checkpoint modela i training state-a.
    
    Args:
        path: Putanja za spremanje
        model: Model (može biti torch.compile wrappan)
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: Gradient scaler
        epoch: Trenutna epoha
        best_auc: Najbolji AUC do sada
        history: History treninga
    """
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


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> tuple:
    """
    Učitava checkpoint modela i training state-a.
    
    Args:
        path: Putanja checkpointa
        model: Model u koji se učitava state
        optimizer: Optimizer u koji se učitava state
        scheduler: Scheduler u koji se učitava state
        scaler: Scaler u koji se učitava state
        device: Device za učitavanje
        
    Returns:
        (epoch, best_auc, history): Učitani metapodaci
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    
    return (
        ckpt["epoch"],
        ckpt.get("best_auc", 0.0),
        ckpt.get("history", [])
    )


def count_parameters(model: nn.Module) -> int:
    """Vraća broj trainable parametara."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Vraća ukupan broj parametara (uključujući zamrznute)."""
    return sum(p.numel() for p in model.parameters())
