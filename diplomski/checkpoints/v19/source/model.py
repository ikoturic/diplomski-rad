"""
CNN + LSTM + BBox model za next-frame predviđanje opasnosti u prometu.

Arhitektura:
    Ulaz:  24 framea RGB + bounding box anotacije po frameu
    Izlaz: danger score za frame 25 (next-frame prediction)

    Dva informacijska toka:

    1. GLOBALNI tok (scene-level):
       - ResNet18 izvlači globalne vizualne značajke svake scene (512-dim)
       - Motion signal: razlika značajki uzastopnih frameova (128-dim)

    2. OBJEKTNI tok (object-level):
       - ROI Align: iz CNN feature mapa izvlači vizualne značajke svakog objekta
       - Bbox Encoder: numeričke značajke (pozicija, veličina, brzina iz tracking-a)
       - Object Attention: agregira sve objekte u jedno kontekst po frameu

    3. TEMPORALNI tok:
       - LSTM s naučenim h_0: procesira 24 timestep-ova
       - Temporal Attention: causal attention - povezuje ključne ranijie frameove

    4. Next-frame Classifier:
       - Iz zadnjeg LSTM stanja → danger logit za frame 25
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_align
from torch.utils.checkpoint import checkpoint as grad_checkpoint


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

        # Multi-head self-attention
        Q = self.q(x).view(N, M, H, self.head_dim).transpose(1, 2)  # (N,H,M,d)
        K = self.k(x).view(N, M, H, self.head_dim).transpose(1, 2)
        V = self.v(x).view(N, M, H, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (N,H,M,M)
        attn_mask = ~mask.unsqueeze(1).unsqueeze(1)                  # (N,1,1,M)
        scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1).nan_to_num(0.0)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)                                  # (N,H,M,d)
        ctx = ctx.transpose(1, 2).contiguous().view(N, M, D)
        ctx = self.out(ctx)

        x = self.norm1(x + self.dropout(ctx))

        # FFN
        x = self.norm2(x + self.dropout(self.ffn(x)))
        x = x * mask.unsqueeze(-1).float()
        return x


class ObjectInteraction(nn.Module):
    """
    Multi-head self-attention interakcija između objekata u frameu.

    Svaki objekt "gleda" sve ostale objekte — model uči relacijske
    uzorke poput "auto se približava kamionu". Obogaćuje per-object
    značajke relacijskim kontekstom prije finalne agregacije.

    Input:  (N, max_objects, object_dim), mask (N, max_objects)
    Output: (N, max_objects, object_dim)  – obogaćeni per-object vektori
    """

    def __init__(self, object_dim, n_heads=4, n_layers=1, dropout=0.1, **kwargs):
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
        """
        Args:
            features: (N, max_obj, D) – per-object feature vektori
            mask:     (N, max_obj)     – True za stvarne objekte, False za padding
        Returns:
            context:  (N, D) – agregirani objektni kontekst
        """
        scores = self.attn(features).squeeze(-1)      # (N, max_obj)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Frameovi bez ijednog objekta → context = nula
        has_objects = mask.any(dim=-1)                 # (N,)
        weights = F.softmax(scores, dim=-1)            # (N, max_obj)
        weights = weights.masked_fill(~has_objects.unsqueeze(-1), 0.0)

        context = torch.bmm(
            weights.unsqueeze(1), features
        ).squeeze(1)                                   # (N, D)
        return context


# ========================================================================= #
#  Temporal Attention – causal attention nad LSTM izlazima                    #
# ========================================================================= #

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention s opcionalno kauzalnom maskom.

    Koristi se za naglašavanje ključnih vremenskih koraka
    unutar sekvence LSTM outputa.
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
#  Flow Encoder – enkodira optički tok                                        #
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
        """flow: (N, 2, H, W) → (N, out_dim)"""
        x = self.conv(flow)
        x = x.flatten(1)
        return self.fc(x)


# ========================================================================= #
#  Per-Object Flow Encoder – enkodira flow patch per objekt                   #
# ========================================================================= #

class ObjectFlowEncoder(nn.Module):
    """
    MLP enkoder za per-object optical flow patches.

    Svaki objekt dobiva flow crop rezan iz residual flow mape
    (ego-motion subtracted) i poolan na fiksnu veličinu (pool_size × pool_size).

    Input:  (N, max_obj, 2, pool_size, pool_size) – flow patches per object
    Output: (N, max_obj, out_dim) – per-object flow feature vektori
    """

    def __init__(self, pool_size=5, out_dim=128, dropout=0.3):
        super().__init__()
        in_dim = 2 * pool_size * pool_size  # 2 * 5 * 5 = 50
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, obj_flow, mask):
        """
        Args:
            obj_flow: (N, max_obj, 2, ps, ps) – per-object flow patches
            mask:     (N, max_obj) – True za stvarne objekte
        Returns:
            (N, max_obj, out_dim) – per-object flow features (zero-masked za padding)
        """
        N, M = obj_flow.shape[:2]
        x = obj_flow.flatten(2)                     # (N, M, 2*ps*ps)
        x = self.encoder(x)                         # (N, M, out_dim)
        x = x * mask.unsqueeze(-1).float()          # Zero-out padding
        return x


# ========================================================================= #
#  Glavni model: CNN + LSTM + BBox                                            #
# ========================================================================= #

class CNN_LSTM_BBox(nn.Module):
    """
    CNN + LSTM + BBox za next-frame predviđanje opasnosti.

    Vidi 24 framea + bounding box anotacije → predviđa danger za frame 25.

    Parametri:
        hidden_dim:         LSTM skrivena dimenzija
        lstm_layers:        broj LSTM slojeva
        dropout:            dropout stopa
        backbone:           'resnet18' ili 'resnet50'
        freeze_cnn:         zamrzni CNN backbone (True preporučeno)
        unfreeze_layer3:    odmrzni layer3 za fine-tuning
        unfreeze_layer4:    odmrzni layer4 za fine-tuning
        max_objects:        max objekata po frameu
        roi_output_size:    ROI Align output veličina (3x3)
        roi_feat_dim:       dimenzija ROI feature vektora
        bbox_feat_dim:      dimenzija bbox numeričkih značajki
        category_embed_dim: dimenzija category embedding-a
        use_attention:      koristiti temporal attention
        attention_heads:    broj attention glava
        use_flow:           koristiti optical flow enkoder
        flow_feat_dim:      dimenzija flow feature vektora
    """

    def __init__(
        self,
        hidden_dim=256,
        lstm_layers=1,
        dropout=0.5,
        lstm_dropout=0.0,
        backbone='resnet50',
        freeze_cnn=True,
        unfreeze_layer3=False,
        unfreeze_layer4=False,
        max_objects=10,
        roi_output_size=3,
        roi_feat_dim=128,
        bbox_feat_dim=64,
        category_embed_dim=16,
        use_attention=True,
        attention_heads=4,
        use_flow=False,
        flow_feat_dim=128,
        use_object_interaction=False,
        object_interaction_heads=4,
        object_interaction_layers=1,
        use_per_object_flow=False,
        obj_flow_pool_size=5,
        obj_flow_feat_dim=128,
        bidirectional=False,
        use_causal_attention=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.max_objects = max_objects
        self.use_attention = use_attention
        self.use_flow = use_flow
        self.flow_feat_dim = flow_feat_dim
        self.use_object_interaction = use_object_interaction
        self.use_per_object_flow = use_per_object_flow
        self.obj_flow_feat_dim = obj_flow_feat_dim
        self.roi_output_size = roi_output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # =====================================================================
        # CNN Backbone: ResNet18 ili ResNet50 pretrained
        # =====================================================================
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.layer3_channels = 1024   # ResNet50 layer3: 1024ch
            self.raw_global_dim = 2048    # ResNet50 layer4+pool: 2048-dim
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.layer3_channels = 256    # ResNet18 layer3: 256ch
            self.raw_global_dim = 512     # ResNet18 layer4+pool: 512-dim

        # Rani slojevi (uvijek zamrznuti)
        self.cnn_early = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,   # R18: 64ch / R50: 256ch
            resnet.layer2,   # R18: 128ch / R50: 512ch
        )
        # Layer3: feature mape za ROI Align
        self.cnn_layer3 = resnet.layer3  # R18: 256ch / R50: 1024ch
        # Layer4 + pool: globalne scene značajke
        self.cnn_layer4 = resnet.layer4  # R18: 512ch / R50: 2048ch
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_cnn:
            for p in self.cnn_early.parameters():
                p.requires_grad = False
            if not unfreeze_layer3:
                for p in self.cnn_layer3.parameters():
                    p.requires_grad = False
            if not unfreeze_layer4:
                for p in self.cnn_layer4.parameters():
                    p.requires_grad = False

        # Trebamo li grad za CNN (za gradient checkpointing)
        self._finetune_cnn = (
            any(p.requires_grad for p in self.cnn_layer3.parameters()) or
            any(p.requires_grad for p in self.cnn_layer4.parameters())
        )

        # =====================================================================
        # Global Feature Projection: raw_dim → 512
        # =====================================================================
        self.global_out_dim = 512  # Standardna dimenzija neovisna o backboneu
        if self.raw_global_dim != self.global_out_dim:
            self.global_proj = nn.Sequential(
                nn.Linear(self.raw_global_dim, self.global_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),
            )
        else:
            self.global_proj = nn.Identity()

        # =====================================================================
        # ROI Feature Extraction: vizualne značajke svakog objekta
        # =====================================================================
        # ROI Align → (layer3_ch, roi_size, roi_size) → flatten → project
        roi_flat_dim = self.layer3_channels * roi_output_size * roi_output_size
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_flat_dim, roi_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )

        # =====================================================================
        # Bbox Feature Encoder: numeričke značajke + kategorija
        # =====================================================================
        # Numeričke: [cx, cy, w, h, area, aspect, dx, dy, dw, dh, conf, approach, expansion, ttc, flow_mag, flow_angle, flow_std] = 17
        self.cat_embed = nn.Embedding(NUM_CATEGORIES + 1, category_embed_dim)  # 0=pad
        bbox_input_dim = 17 + category_embed_dim  # 33
        self.bbox_encoder = nn.Sequential(
            nn.Linear(bbox_input_dim, bbox_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )

        # =====================================================================
        # Per-Object Flow Encoder (opcionalno)
        # =====================================================================
        obj_flow_dim = 0
        if use_per_object_flow:
            self.obj_flow_encoder = ObjectFlowEncoder(
                pool_size=obj_flow_pool_size,
                out_dim=obj_flow_feat_dim,
                dropout=dropout * 0.3,
            )
            obj_flow_dim = obj_flow_feat_dim

        # =====================================================================
        # Object Interaction: self-attention između objekata (opcionalno)
        # =====================================================================
        object_dim = roi_feat_dim + bbox_feat_dim + obj_flow_dim  # 192 or 320
        if use_object_interaction:
            self.obj_interaction = ObjectInteraction(
                object_dim,
                n_heads=object_interaction_heads,
                n_layers=object_interaction_layers,
                dropout=dropout * 0.3,
            )

        # =====================================================================
        # Object Attention: agregira objekte u kontekst po frameu
        # =====================================================================
        self.obj_attention = ObjectAttention(object_dim, hidden_dim=64)

        # =====================================================================
        # Motion Signal: razlika globalnih značajki (iz projiciranih 512-dim)
        # =====================================================================
        self.motion_proj = nn.Sequential(
            nn.Linear(self.global_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )
        motion_dim = 128

        # =====================================================================
        # Optical Flow Encoder (opcionalno)
        # =====================================================================
        flow_dim = 0
        if use_flow:
            self.flow_encoder = FlowEncoder(
                out_dim=flow_feat_dim,
                dropout=dropout * 0.3,
            )
            flow_dim = flow_feat_dim

        # =====================================================================
        # LSTM s naučenim početnim stanjem
        # =====================================================================
        lstm_input_dim = self.global_out_dim + object_dim + motion_dim + flow_dim
        self.feature_dropout = nn.Dropout(dropout * 0.3)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # Recurrent dropout: primjenjuje se na LSTM output po timestep-u
        lstm_out_dim = hidden_dim * self.num_directions
        self.lstm_drop = nn.Dropout(lstm_dropout) if lstm_dropout > 0 else None

        # Learnable h_0 / c_0: model uči "neutralno" stanje normalne vožnje
        self.h0 = nn.Parameter(torch.zeros(lstm_layers * self.num_directions, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(lstm_layers * self.num_directions, 1, hidden_dim))
        nn.init.xavier_uniform_(self.h0)
        nn.init.xavier_uniform_(self.c0)

        # =====================================================================
        # Temporal Attention
        # =====================================================================
        if use_attention:
            self.temp_attn = TemporalAttention(
                lstm_out_dim, num_heads=attention_heads, dropout=dropout * 0.3,
                causal=use_causal_attention,
            )
            self.attn_norm = nn.LayerNorm(lstm_out_dim)

        # =====================================================================
        # Next-frame Classifier
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------ #
    #  CNN Feature Extraction                                              #
    # ------------------------------------------------------------------ #

    def _extract_features(self, x):
        """
        Izvlači feature mape (za ROI) i globalne značajke.

        Args:
            x: (N, 3, H, W)
        Returns:
            feat_maps:   (N, layer3_ch, H', W')  – za ROI Align
            global_feat: (N, 512)                 – globalna scena (projicirano)
        """
        if self._finetune_cnn:
            # Fine-tuning: gradient checkpointing za uštedu VRAM-a
            with torch.no_grad():
                early = self.cnn_early(x)       # Uvijek zamrznuti
            feat_maps = grad_checkpoint(
                self.cnn_layer3, early, use_reentrant=False,
            )
            late = grad_checkpoint(
                self.cnn_layer4, feat_maps, use_reentrant=False,
            )
        else:
            # Sve zamrznuto: bez gradijenta
            with torch.no_grad():
                early = self.cnn_early(x)
                feat_maps = self.cnn_layer3(early)
                late = self.cnn_layer4(feat_maps)
        global_feat = self.pool(late).flatten(1)     # (N, raw_global_dim)
        global_feat = self.global_proj(global_feat)  # (N, 512)
        return feat_maps, global_feat

    # ------------------------------------------------------------------ #
    #  ROI Feature Extraction                                              #
    # ------------------------------------------------------------------ #

    def _extract_roi_features(self, feat_maps, bboxes, mask):
        """
        ROI Align: izvlači vizualne značajke za svaki bounding box.

        Args:
            feat_maps: (N, C, H', W') – CNN feature mape (layer3 output)
            bboxes:    (N, max_obj, 4) – normalizirani [x1,y1,x2,y2] u [0,1]
            mask:      (N, max_obj)    – True za stvarne objekte
        Returns:
            roi_feats: (N, max_obj, roi_feat_dim)
        """
        N = feat_maps.shape[0]
        feat_h = feat_maps.shape[-1]  # Veličina feature mape (14 za 224x224)
        max_obj = bboxes.shape[1]

        # Batch index za svaki ROI: [batch_idx, x1, y1, x2, y2]
        # Efikasnije: repeat_interleave umjesto expand+reshape
        batch_ids = torch.arange(N, device=feat_maps.device, dtype=torch.float32)
        batch_ids = batch_ids.repeat_interleave(max_obj).unsqueeze(1)  # (N*max_obj, 1)

        flat_boxes = bboxes.reshape(N * max_obj, 4)
        rois = torch.cat([batch_ids, flat_boxes], dim=1)  # (N*max_obj, 5)

        # ROI Align: spatial_scale dinamički iz feature map veličine
        # Bboxovi su u [0,1] → [0,1] * feat_h = feature map koordinate
        roi_feats = roi_align(
            feat_maps, rois,
            output_size=self.roi_output_size,
            spatial_scale=float(feat_h),
        )  # (N*max_obj, layer3_ch, roi_size, roi_size)

        roi_feats = roi_feats.flatten(1)               # (N*max_obj, layer3_ch*roi_size^2)
        roi_feats = self.roi_proj(roi_feats)            # (N*max_obj, roi_feat_dim)

        # Zero-out padding objekti
        flat_mask = mask.reshape(N * max_obj).float().unsqueeze(1)
        roi_feats = roi_feats * flat_mask

        return roi_feats.view(N, max_obj, -1)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, frames, bboxes, bbox_features, categories, obj_mask, flow=None, obj_flow=None, return_all_timesteps=False):
        """
        Args:
            frames:        (B, T, 3, H, W)       – RGB framea
            bboxes:        (B, T, max_obj, 4)     – normalizirani [x1,y1,x2,y2]
            bbox_features: (B, T, max_obj, 17)    – numeričke značajke
            categories:    (B, T, max_obj)         – kategorije (long)
            obj_mask:      (B, T, max_obj)         – maska pravih objekata
            flow:          (B, T, 2, Hf, Wf)      – optički tok (opcionalno)
            obj_flow:      (B, T, max_obj, 2, ps, ps) – per-object flow patches (opcionalno)

        Returns:
            logit: (B,) – danger logit za sljedeći frame
        """
        B, T = frames.shape[:2]

        # === CNN Feature Extraction ===
        frames_flat = frames.view(B * T, *frames.shape[2:])
        feat_maps, global_feats = self._extract_features(frames_flat)
        global_feats = global_feats.view(B, T, -1)    # (B, T, 512)

        # === ROI Features ===
        roi_feats = self._extract_roi_features(
            feat_maps,
            bboxes.view(B * T, self.max_objects, 4),
            obj_mask.view(B * T, self.max_objects),
        )  # (B*T, max_obj, roi_feat_dim)

        # === Bbox Features + Category Embedding ===
        cats_flat = categories.view(B * T, self.max_objects)
        cat_emb = self.cat_embed(cats_flat)            # (B*T, max_obj, embed_dim)
        bbox_flat = bbox_features.view(B * T, self.max_objects, -1)
        bbox_input = torch.cat([bbox_flat, cat_emb], dim=-1)
        bbox_enc = self.bbox_encoder(bbox_input)       # (B*T, max_obj, bbox_feat_dim)

        # === Per-Object: ROI + Bbox + Flow ===
        obj_parts = [roi_feats, bbox_enc]
        if self.use_per_object_flow:
            if obj_flow is not None:
                of_flat = obj_flow.view(B * T, self.max_objects, *obj_flow.shape[3:])
                of_feats = self.obj_flow_encoder(
                    of_flat, obj_mask.view(B * T, self.max_objects),
                )  # (B*T, max_obj, obj_flow_feat_dim)
            else:
                of_feats = torch.zeros(
                    B * T, self.max_objects, self.obj_flow_feat_dim,
                    device=frames.device,
                )
            obj_parts.append(of_feats)
        obj_feats = torch.cat(obj_parts, dim=-1)  # (B*T, max_obj, obj_dim)

        # === Object Interaction: self-attention između objekata ===
        if self.use_object_interaction:
            obj_feats = self.obj_interaction(
                obj_feats, obj_mask.view(B * T, self.max_objects)
            )  # (B*T, max_obj, obj_dim) — obogaćen relacijskim kontekstom

        # === Object Attention: agregiraj objekte po frameu ===
        obj_context = self.obj_attention(
            obj_feats, obj_mask.view(B * T, self.max_objects)
        )  # (B*T, obj_dim)
        obj_context = obj_context.view(B, T, -1)      # (B, T, obj_dim)

        # === Motion Signal (iz projiciranih 512-dim značajki) ===
        motion = global_feats[:, 1:] - global_feats[:, :-1]
        zero_pad = torch.zeros(B, 1, global_feats.shape[-1], device=frames.device)
        motion = torch.cat([zero_pad, motion], dim=1)
        motion = self.motion_proj(motion)              # (B, T, 128)

        # === Optical Flow Features ===
        if self.use_flow:
            if flow is not None:
                flow_flat = flow.view(B * T, *flow.shape[2:])
                flow_feats = self.flow_encoder(flow_flat)
                flow_feats = flow_feats.view(B, T, -1)  # (B, T, flow_dim)
            else:
                flow_feats = torch.zeros(
                    B, T, self.flow_feat_dim, device=frames.device,
                )
            combined = torch.cat(
                [global_feats, obj_context, motion, flow_feats], dim=-1,
            )
        else:
            combined = torch.cat(
                [global_feats, obj_context, motion], dim=-1,
            )

        combined = self.feature_dropout(combined)

        # === LSTM s naučenim h_0 ===
        h0 = self.h0.expand(-1, B, -1).contiguous()
        c0 = self.c0.expand(-1, B, -1).contiguous()
        lstm_out, _ = self.lstm(combined, (h0, c0))    # (B, T, hidden_dim)

        # Recurrent dropout na LSTM outputu
        if self.lstm_drop is not None and self.training:
            lstm_out = self.lstm_drop(lstm_out)

        # === Temporal Attention (residual) ===
        if self.use_attention:
            attn_out = self.temp_attn(lstm_out)
            lstm_out = self.attn_norm(lstm_out + attn_out)

        # === Predikcija ===
        if return_all_timesteps:
            # Multi-timestep: predikcija na svakom timestep-u
            all_logits = self.classifier(lstm_out).squeeze(-1)  # (B, T)
            return all_logits
        else:
            # Standardno: samo zadnji timestep
            last_hidden = lstm_out[:, -1, :]               # (B, hidden_dim)
            logit = self.classifier(last_hidden).squeeze(-1)  # (B,)
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
    for bb in ['resnet18', 'resnet50']:
        print(f"\n{'='*60}")
        print(f"Backbone: {bb}")
        print(f"{'='*60}")
        model = CNN_LSTM_BBox(
            hidden_dim=256, lstm_layers=1, dropout=0.5,
            backbone=bb, freeze_cnn=True, max_objects=10, use_attention=True,
        )
        print(f"Trainable: {count_parameters(model):,}")
        print(f"Total:     {count_all_parameters(model):,}")

        B, T = 2, 16
        frames = torch.randn(B, T, 3, 224, 224)
        bboxes = torch.rand(B, T, 10, 4)
        bbox_feat = torch.randn(B, T, 10, 10)
        cats = torch.randint(0, 8, (B, T, 10))
        mask = torch.zeros(B, T, 10, dtype=torch.bool)
        mask[:, :, :3] = True

        logit = model(frames, bboxes, bbox_feat, cats, mask)
        prob = torch.sigmoid(logit)
        print(f"Input:  frames {frames.shape}")
        print(f"Output: logit {logit.shape}, prob = {[f'{p:.3f}' for p in prob.tolist()]}")

    # Test fine-tuning mode
    print(f"\n{'='*60}")
    print(f"ResNet50 fine-tune (layer3+layer4)")
    print(f"{'='*60}")
    model_ft = CNN_LSTM_BBox(
        hidden_dim=256, lstm_layers=1, dropout=0.5,
        backbone='resnet50', freeze_cnn=True,
        unfreeze_layer3=True, unfreeze_layer4=True,
        max_objects=10, use_attention=True,
    )
    print(f"Trainable: {count_parameters(model_ft):,}")
    print(f"Total:     {count_all_parameters(model_ft):,}")
