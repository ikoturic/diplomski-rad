"""
Loss funkcije za trening modela.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss za binarnu klasifikaciju (radi s logitima).

    Smanjuje doprinos "laganih" primjera (normalna vožnja) i pojačava
    doprinos "teških" primjera (tranzicija normalno → opasno).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Label smoothing: targets 0→ε, 1→1-ε  (sprječava overconfidence)
    
    Args:
        gamma: Fokusni parametar (obično 2.0)
        alpha: Balansiranje pozitivnih/negativnih primjera (obično 0.75)
        label_smoothing: Smoothing parametar za labele
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Model output logiti (B,) ili (B, T)
            targets: Ground truth labele (B,) ili (B, T)
            
        Returns:
            Skalarna loss vrijednost
        """
        # Label smoothing: 0 → ε, 1 → 1-ε
        if self.label_smoothing > 0:
            targets = (
                targets * (1 - self.label_smoothing) + 
                0.5 * self.label_smoothing
            )

        # BCE loss bez redukcije
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # p_t: probability of true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Class balancing weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final loss
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCE Loss s class weight balansiranjem.
    
    Jednostavnija alternativa Focal lossu.
    
    Args:
        pos_weight: Težina pozitivne klase (obično > 1 za neuravnotežene podatke)
        label_smoothing: Label smoothing parametar
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.label_smoothing = label_smoothing
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Model output logiti
            targets: Ground truth labele
            
        Returns:
            Skalarna loss vrijednost
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = (
                targets * (1 - self.label_smoothing) + 
                0.5 * self.label_smoothing
            )
            
        # Move pos_weight to correct device
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
            
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        
        return loss
