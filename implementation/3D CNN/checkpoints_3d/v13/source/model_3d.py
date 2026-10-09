"""
3D CNN model za next-frame predviđanje opasnosti u prometu.

Arhitektura:
    Ulaz:  24 framea RGB + bounding box anotacije po frameu
    Izlaz: danger score za frame 25 (next-frame prediction)

    Dva informacijska toka:

    1. GLOBALNI spatiotemporalni tok (3D CNN):
       - Pretrenirana 3D CNN (R(2+1)D-18 / R3D-18 / MC3-18) obrađuje
         cijeli clip (24 framea) odjednom → 512-dim spatiotemporalni vektor
       - Za razliku od 2D CNN + LSTM pristupa, 3D konvolucije zajednički
         modeliraju prostorne i vremenske uzorke (motion, scene changes)

    2. OBJEKTNI tok (object-level):
       - Bbox Encoder: numeričke značajke (pozicija, veličina, brzina iz tracking-a)
       - Object Attention: agregira sve objekte u jednu reprezentaciju po frameu
       - Temporal LSTM + Attention: procesira sekvencijalno objekte kroz vrijeme

    3. Classifier:
       - Spaja 3D CNN features + object features → danger logit za frame 25

    Pretrained backbone:
       R(2+1)D-18 (Kinetics-400): factorized 3D conv = spatial 2D + temporal 1D
       Prednost nad R3D-18: bolja optimizacija, bolji accuracy, sličan compute
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Import FlowEncoder iz parent modula ako treba
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Kategorije objekata u DoTA anotacijama
CATEGORY_MAP = {
    "car": 1,
    "truck": 2,
    "bus": 3,
    "person": 4,
    "rider": 5,
    "bike": 6,
    "motor": 7,
}
NUM_CATEGORIES = len(CATEGORY_MAP)  # 7


# ========================================================================= #
#  Object Attention – agregira objekte u frameu                               #
# ========================================================================= #

class ObjectAttention(nn.Module):
    """
    Attention agregacija objekata u jednom frameu.

    Za svaki frame, spaja N detektiranih objekata u jedan kontekstni vektor.
    Model sam uči koji objekti su važni (npr. auto koji se brzo kreće
    dobiva veću težinu od parkiranih auta).

    Input:  (N, max_objects, object_dim)
    Output: (N, object_dim)
    """

    def __init__(self, object_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, mask):
        scores = self.attn(features).squeeze(-1)      # (N, max_obj)
        scores = scores.masked_fill(~mask, float("-inf"))
        has_objects = mask.any(dim=-1)                 # (N,)
        weights = F.softmax(scores, dim=-1)            # (N, max_obj)
        weights = weights.masked_fill(~has_objects.unsqueeze(-1), 0.0)
        context = torch.bmm(
            weights.unsqueeze(1), features
        ).squeeze(1)                                   # (N, D)
        return context


# ========================================================================= #
#  Temporal Attention – causal attention nad objektnim kontekstom              #
# ========================================================================= #

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention s kauzalnom maskom.
    Koristi se za temporalnu agregaciju objektnog konteksta.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape
        H = self.num_heads

        Q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ========================================================================= #
#  Flow Encoder – enkodira optički tok (reuse iz parent modula)               #
# ========================================================================= #

class FlowEncoder(nn.Module):
    """
    CNN enkoder za optički tok (2-kanalni: dx, dy).
    Input:  (N, 2, H, W) – flow mape proizvoljne veličine
    Output: (N, out_dim)  – flow feature vektor
    """

    def __init__(self, out_dim=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, flow):
        x = self.conv(flow)
        x = x.flatten(1)
        return self.fc(x)


# ========================================================================= #
#  Object Interaction – self-attention između objekata u frameu               #
# ========================================================================= #

class _InteractionLayer(nn.Module):
    """Jedan sloj multi-head self-attention + FFN za objekte."""

    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, M, D = x.shape
        H = self.n_heads

        Q = self.q(x).view(N, M, H, self.head_dim).transpose(1, 2)
        K = self.k(x).view(N, M, H, self.head_dim).transpose(1, 2)
        V = self.v(x).view(N, M, H, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_mask = ~mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)
        ctx = ctx.transpose(1, 2).contiguous().view(N, M, D)
        ctx = self.out(ctx)

        x = self.norm1(x + self.dropout(ctx))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        x = x * mask.unsqueeze(-1).float()
        return x


class ObjectInteraction(nn.Module):
    """Multi-head self-attention interakcija između objekata u frameu."""

    def __init__(self, object_dim, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            _InteractionLayer(object_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, features, mask):
        for layer in self.layers:
            features = layer(features, mask)
        return features


# ========================================================================= #
#  Glavni model: 3D CNN + Object Branch                                       #
# ========================================================================= #

class CNN3D_Danger(nn.Module):
    """
    3D CNN model za next-frame predviđanje opasnosti.

    Umjesto 2D CNN (per-frame) + LSTM, koristi 3D CNN koji
    obrađuje cijeli clip (24 framea) odjednom i zajednički
    modelira prostorne i vremenske značajke.

    Podržani backbonei:
        - r2plus1d_18: R(2+1)D-18 – factorized 3D conv (preporučeno)
        - r3d_18:      3D ResNet-18 – standardne 3D konvolucije
        - mc3_18:      MC3-18 – mixed conv (3D rani, 2D kasni slojevi)

    Svi su pretrenirani na Kinetics-400 (action recognition).
    """

    def __init__(
        self,
        backbone_3d="r2plus1d_18",
        hidden_dim=256,
        dropout=0.5,
        freeze_3d=False,
        freeze_early=True,
        use_grad_checkpoint=True,
        max_objects=15,
        bbox_feat_dim=64,
        category_embed_dim=16,
        use_object_branch=True,
        use_object_interaction=False,
        object_interaction_heads=4,
        object_interaction_layers=1,
        obj_temporal_layers=1,
        obj_temporal_dropout=0.0,
        attention_heads=4,
        use_causal_attention=True,
        use_flow=False,
        flow_feat_dim=128,
        use_temporal_aux=False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.use_object_branch = use_object_branch
        self.use_object_interaction = use_object_interaction
        self.use_flow = use_flow
        self.flow_feat_dim = flow_feat_dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.use_temporal_aux = use_temporal_aux

        # =====================================================================
        # 3D CNN Backbone
        # =====================================================================
        self.backbone_3d_type = backbone_3d

        if backbone_3d == "r3d_18":
            backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
        elif backbone_3d == "r2plus1d_18":
            backbone = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.DEFAULT)
        elif backbone_3d == "mc3_18":
            backbone = video_models.mc3_18(weights=video_models.MC3_18_Weights.DEFAULT)
        else:
            raise ValueError(f"Nepoznat 3D backbone: {backbone_3d}")

        # Svi backbonei: stem → layer1 → layer2 → layer3 → layer4 → avgpool → fc(512→400)
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool3d(1,1,1)
        self.global_feat_dim = 512       # Svi 3D ResNet varijante: 512-dim izlaz

        # =====================================================================
        # Freezing strategija
        # =====================================================================
        self._finetune_3d = False
        if freeze_3d:
            # Zamrzni cijeli backbone
            for p in self.stem.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
            for p in self.layer2.parameters():
                p.requires_grad = False
            for p in self.layer3.parameters():
                p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = False
        elif freeze_early:
            # Zamrzni stem + layer1 + layer2, fine-tune layer3 + layer4
            for p in self.stem.parameters():
                p.requires_grad = False
            for p in self.layer1.parameters():
                p.requires_grad = False
            for p in self.layer2.parameters():
                p.requires_grad = False
            self._finetune_3d = True
        else:
            # Fine-tune sve
            self._finetune_3d = True

        # =====================================================================
        # Global Feature Projection: 512 → hidden_dim
        # =====================================================================
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )

        # =====================================================================
        # Object Branch (numeričke bbox značajke + categorical)
        # =====================================================================
        classifier_input_dim = hidden_dim

        if use_object_branch:
            # Category embedding
            self.cat_embed = nn.Embedding(NUM_CATEGORIES + 1, category_embed_dim)  # 0=pad

            # Bbox encoder: 17 numeričkih + category_embed_dim → bbox_feat_dim
            bbox_input_dim = 17 + category_embed_dim
            self.bbox_encoder = nn.Sequential(
                nn.Linear(bbox_input_dim, bbox_feat_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),
            )

            object_dim = bbox_feat_dim

            # Object Interaction (opcionalno)
            if use_object_interaction:
                self.obj_interaction = ObjectInteraction(
                    object_dim,
                    n_heads=object_interaction_heads,
                    n_layers=object_interaction_layers,
                    dropout=dropout * 0.3,
                )

            # Per-frame object attention
            self.obj_attention = ObjectAttention(object_dim, hidden_dim=64)

            # Temporalna agregacija objektnog konteksta (LSTM + Attention)
            self.obj_temporal_lstm = nn.LSTM(
                input_size=object_dim,
                hidden_size=object_dim,
                num_layers=obj_temporal_layers,
                batch_first=True,
                dropout=obj_temporal_dropout if obj_temporal_layers > 1 else 0.0,
            )
            n_heads = min(attention_heads, max(1, object_dim // 16))
            # Osiguraj da object_dim bude djeljiv s brojem glava
            while object_dim % n_heads != 0:
                n_heads -= 1
            self.obj_temporal_attn = TemporalAttention(
                object_dim, num_heads=n_heads,
                dropout=dropout * 0.3, causal=use_causal_attention,
            )
            self.obj_temporal_norm = nn.LayerNorm(object_dim)

            # Projekcija objektnih značajki
            self.obj_proj = nn.Sequential(
                nn.Linear(object_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),
            )
            classifier_input_dim += hidden_dim

        # =====================================================================
        # Optical Flow Encoder (opcionalno – redundantno s 3D CNN ali konfigurabilno)
        # =====================================================================
        if use_flow:
            self.flow_encoder = FlowEncoder(
                out_dim=flow_feat_dim, dropout=dropout * 0.3,
            )
            self.flow_temporal = nn.Sequential(
                nn.Linear(flow_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),
            )
            classifier_input_dim += hidden_dim

        # =====================================================================
        # Classifier: combined features → danger logit
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # =====================================================================
        # Temporal Auxiliary Head (opcionalno)
        # Radi na per-temporal-slice featurima iz layer4 (T'=3 za T=24)
        # Daje 3× više nadzornog signala → bolja generalizacija
        # =====================================================================
        if use_temporal_aux:
            # Spatial pool zadržava temporalnu dimenziju
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.temporal_head = nn.Sequential(
                nn.Linear(self.global_feat_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1),
            )

    # ------------------------------------------------------------------ #
    #  3D CNN Feature Extraction                                           #
    # ------------------------------------------------------------------ #

    def _extract_3d_features(self, x):
        """
        Izvlači globalne spatiotemporalne značajke iz 3D CNN.

        Args:
            x: (B, 3, T, H, W) – RGB clip
        Returns:
            global_feat: (B, 512) – spatiotemporalni feature vektor
        """
        if not self.training:
            # Eval mode: čisti forward bez grad konteksta
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self._finetune_3d and self.use_grad_checkpoint:
            # Zamrznuti rani slojevi
            with torch.no_grad():
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
            # Fine-tune s gradient checkpointingom za uštedu VRAM-a
            x = grad_checkpoint(self.layer3, x, use_reentrant=False)
            x = grad_checkpoint(self.layer4, x, use_reentrant=False)
        elif self._finetune_3d:
            with torch.no_grad():
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            # Sve zamrznuto
            with torch.no_grad():
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

        # x: (B, 512, T', H', W') nakon layer4
        # Temporalni aux features – samo u training modu
        temporal_feats = None
        if self.use_temporal_aux and self.training:
            sp = self.spatial_pool(x)                              # (B, 512, T', 1, 1)
            temporal_feats = sp.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, T', 512)

        x = self.avgpool(x)           # (B, 512, 1, 1, 1)
        return x.flatten(1), temporal_feats  # (B, 512), (B, T', 512) or None

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, frames, bboxes=None, bbox_features=None, categories=None,
                obj_mask=None, flow=None, return_aux=False, **kwargs):
        """
        Args:
            frames:        (B, T, 3, H, W)       – RGB framea
            bboxes:        (B, T, max_obj, 4)     – ne koristi se (nema ROI Align u 3D)
            bbox_features: (B, T, max_obj, 17)    – numeričke značajke objekata
            categories:    (B, T, max_obj)         – kategorije (long)
            obj_mask:      (B, T, max_obj)         – maska pravih objekata
            flow:          (B, T, 2, Hf, Wf)      – optički tok (opcionalno)

        Returns:
            logit: (B,) – danger logit za sljedeći frame
        """
        B, T = frames.shape[:2]

        # =====================================================================
        # 3D CNN: (B, T, 3, H, W) → permute → (B, 3, T, H, W) → (B, 512)
        # =====================================================================
        x = frames.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, H, W)
        global_feat, temporal_feats = self._extract_3d_features(x)  # (B, 512), maybe (B, T', 512)
        global_feat = self.global_proj(global_feat)       # (B, hidden_dim)

        parts = [global_feat]

        # =====================================================================
        # Object Branch: bbox features → temporal aggregation
        # =====================================================================
        if self.use_object_branch and bbox_features is not None:
            cats_flat = categories.view(B * T, self.max_objects)
            cat_emb = self.cat_embed(cats_flat)            # (B*T, max_obj, embed_dim)
            bbox_flat = bbox_features.view(B * T, self.max_objects, -1)
            bbox_input = torch.cat([bbox_flat, cat_emb], dim=-1)
            bbox_enc = self.bbox_encoder(bbox_input)       # (B*T, max_obj, bbox_feat_dim)

            # Object Interaction (opcionalno)
            if self.use_object_interaction:
                bbox_enc = self.obj_interaction(
                    bbox_enc, obj_mask.view(B * T, self.max_objects)
                )

            # Per-frame object attention → (B*T, bbox_feat_dim)
            obj_context = self.obj_attention(
                bbox_enc, obj_mask.view(B * T, self.max_objects)
            )
            obj_context = obj_context.view(B, T, -1)       # (B, T, bbox_feat_dim)

            # Temporal LSTM + Attention
            obj_temporal, _ = self.obj_temporal_lstm(obj_context)
            attn_out = self.obj_temporal_attn(obj_temporal)
            obj_temporal = self.obj_temporal_norm(obj_temporal + attn_out)
            obj_feat = obj_temporal[:, -1, :]              # (B, bbox_feat_dim)

            obj_feat = self.obj_proj(obj_feat)             # (B, hidden_dim)
            parts.append(obj_feat)

        # =====================================================================
        # Optical Flow Branch (opcionalno)
        # =====================================================================
        if self.use_flow and flow is not None:
            flow_flat = flow.view(B * T, *flow.shape[2:])
            flow_feats = self.flow_encoder(flow_flat)      # (B*T, flow_feat_dim)
            flow_feats = flow_feats.view(B, T, -1)
            flow_feat = flow_feats.mean(dim=1)             # (B, flow_feat_dim)
            flow_feat = self.flow_temporal(flow_feat)      # (B, hidden_dim)
            parts.append(flow_feat)

        # =====================================================================
        # Combine & Classify
        # =====================================================================
        combined = torch.cat(parts, dim=-1)
        logit = self.classifier(combined).squeeze(-1)      # (B,)
        if return_aux and temporal_feats is not None:
            temporal_logits = self.temporal_head(temporal_feats).squeeze(-1)  # (B, T')
            return logit, temporal_logits
        return logit


# ========================================================================= #
#  Pomoćne funkcije                                                          #
# ========================================================================= #

def count_parameters(model):
    """Broj trainable parametara."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """Ukupan broj parametara (uklj. zamrznute)."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for bb in ["r2plus1d_18", "r3d_18", "mc3_18"]:
        print(f"\n{'='*60}")
        print(f"3D Backbone: {bb}")
        print(f"{'='*60}")
        model = CNN3D_Danger(
            backbone_3d=bb, hidden_dim=256, dropout=0.5,
            freeze_3d=False, freeze_early=True,
            max_objects=15, use_object_branch=True,
        )
        print(f"Trainable: {count_parameters(model):,}")
        print(f"Total:     {count_all_parameters(model):,}")

        B, T = 2, 24
        frames = torch.randn(B, T, 3, 112, 112)
        bboxes = torch.rand(B, T, 15, 4)
        bbox_feat = torch.randn(B, T, 15, 17)
        cats = torch.randint(0, 8, (B, T, 15))
        mask = torch.zeros(B, T, 15, dtype=torch.bool)
        mask[:, :, :3] = True

        logit = model(frames, bboxes, bbox_feat, cats, mask)
        prob = torch.sigmoid(logit)
        print(f"Input:  frames {frames.shape}")
        print(f"Output: logit {logit.shape}, prob = {[f'{p:.3f}' for p in prob.tolist()]}")
