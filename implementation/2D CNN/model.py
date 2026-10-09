import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_align
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from utils import NUM_CATEGORIES


class _InteractionLayer(nn.Module):
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


class ObjectAttention(nn.Module):
    def __init__(self, object_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, mask):
        scores = self.attn(features).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))

        has_objects = mask.any(dim=-1)
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~has_objects.unsqueeze(-1), 0.0)

        context = torch.bmm(
            weights.unsqueeze(1), features
        ).squeeze(1)
        return context


class TemporalAttention(nn.Module):
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


class FlowEncoder(nn.Module):
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


class CNN_LSTM_BBox(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        lstm_layers=1,
        dropout=0.5,
        lstm_dropout=0.0,
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
        use_causal_attention=True,
        **kwargs
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
        self.roi_output_size = roi_output_size

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer3_channels = 256
        self.raw_global_dim = 512

        self.cnn_early = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,
        )
        self.cnn_layer3 = resnet.layer3
        self.cnn_layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for p in self.cnn_early.parameters():
            p.requires_grad = False
        for p in self.cnn_layer3.parameters():
            p.requires_grad = False
        for p in self.cnn_layer4.parameters():
            p.requires_grad = False

        self.global_out_dim = 512
        self.global_proj = nn.Identity()

        roi_flat_dim = self.layer3_channels * roi_output_size * roi_output_size
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_flat_dim, roi_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )

        self.cat_embed = nn.Embedding(NUM_CATEGORIES + 1, category_embed_dim)
        bbox_input_dim = 17 + category_embed_dim
        self.bbox_encoder = nn.Sequential(
            nn.Linear(bbox_input_dim, bbox_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )

        object_dim = roi_feat_dim + bbox_feat_dim
        if use_object_interaction:
            self.obj_interaction = ObjectInteraction(
                object_dim,
                n_heads=object_interaction_heads,
                n_layers=object_interaction_layers,
                dropout=dropout * 0.3,
            )

        self.obj_attention = ObjectAttention(object_dim, hidden_dim=64)

        self.motion_proj = nn.Sequential(
            nn.Linear(self.global_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )
        motion_dim = 128

        flow_dim = 0
        if use_flow:
            self.flow_encoder = FlowEncoder(
                out_dim=flow_feat_dim,
                dropout=dropout * 0.3,
            )
            flow_dim = flow_feat_dim

        lstm_input_dim = self.global_out_dim + object_dim + motion_dim + flow_dim
        self.feature_dropout = nn.Dropout(dropout * 0.3)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        lstm_out_dim = hidden_dim
        self.lstm_drop = nn.Dropout(lstm_dropout) if lstm_dropout > 0 else None

        self.h0 = nn.Parameter(torch.zeros(lstm_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(lstm_layers, 1, hidden_dim))
        nn.init.xavier_uniform_(self.h0)
        nn.init.xavier_uniform_(self.c0)

        if use_attention:
            self.temp_attn = TemporalAttention(
                lstm_out_dim, num_heads=attention_heads, dropout=dropout * 0.3,
                causal=use_causal_attention,
            )
            self.attn_norm = nn.LayerNorm(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _extract_features(self, x):
        with torch.no_grad():
            early = self.cnn_early(x)
            feat_maps = self.cnn_layer3(early)
            late = self.cnn_layer4(feat_maps)
        global_feat = self.pool(late).flatten(1)
        global_feat = self.global_proj(global_feat)
        return feat_maps, global_feat

    def _extract_roi_features(self, feat_maps, bboxes, mask):
        N = feat_maps.shape[0]
        feat_h = feat_maps.shape[-1]
        max_obj = bboxes.shape[1]

        batch_ids = torch.arange(N, device=feat_maps.device, dtype=torch.float32)
        batch_ids = batch_ids.repeat_interleave(max_obj).unsqueeze(1)

        flat_boxes = bboxes.reshape(N * max_obj, 4)
        rois = torch.cat([batch_ids, flat_boxes], dim=1)

        roi_feats = roi_align(
            feat_maps, rois,
            output_size=self.roi_output_size,
            spatial_scale=float(feat_h),
        )

        roi_feats = roi_feats.flatten(1)
        roi_feats = self.roi_proj(roi_feats)

        flat_mask = mask.reshape(N * max_obj).float().unsqueeze(1)
        roi_feats = roi_feats * flat_mask

        return roi_feats.view(N, max_obj, -1)

    def forward(self, frames, bboxes, bbox_features, categories, obj_mask, flow=None, obj_flow=None, return_all_timesteps=False, cached_feat_maps=None, cached_global_feats=None):
        if cached_feat_maps is not None:
            B, T = cached_feat_maps.shape[:2]
            feat_maps = cached_feat_maps.reshape(B * T, *cached_feat_maps.shape[2:])
            global_feats = self.global_proj(cached_global_feats.reshape(B * T, -1))
            global_feats = global_feats.view(B, T, -1)
        else:
            B, T = frames.shape[:2]

            frames_flat = frames.view(B * T, *frames.shape[2:])
            feat_maps, global_feats = self._extract_features(frames_flat)
            global_feats = global_feats.view(B, T, -1)

        roi_feats = self._extract_roi_features(
            feat_maps,
            bboxes.view(B * T, self.max_objects, 4),
            obj_mask.view(B * T, self.max_objects),
        )

        cats_flat = categories.view(B * T, self.max_objects)
        cat_emb = self.cat_embed(cats_flat)
        bbox_flat = bbox_features.view(B * T, self.max_objects, -1)
        bbox_input = torch.cat([bbox_flat, cat_emb], dim=-1)
        bbox_enc = self.bbox_encoder(bbox_input)

        obj_feats = torch.cat([roi_feats, bbox_enc], dim=-1)

        if self.use_object_interaction:
            obj_feats = self.obj_interaction(
                obj_feats, obj_mask.view(B * T, self.max_objects)
            )

        obj_context = self.obj_attention(
            obj_feats, obj_mask.view(B * T, self.max_objects)
        )
        obj_context = obj_context.view(B, T, -1)

        motion = global_feats[:, 1:] - global_feats[:, :-1]
        zero_pad = torch.zeros(B, 1, global_feats.shape[-1], device=global_feats.device)
        motion = torch.cat([zero_pad, motion], dim=1)
        motion = self.motion_proj(motion)

        if self.use_flow:
            if flow is not None:
                flow_flat = flow.view(B * T, *flow.shape[2:])
                flow_feats = self.flow_encoder(flow_flat)
                flow_feats = flow_feats.view(B, T, -1)
            else:
                flow_feats = torch.zeros(
                    B, T, self.flow_feat_dim, device=global_feats.device,
                )
            combined = torch.cat(
                [global_feats, obj_context, motion, flow_feats], dim=-1,
            )
        else:
            combined = torch.cat(
                [global_feats, obj_context, motion], dim=-1,
            )

        combined = self.feature_dropout(combined)

        h0 = self.h0.expand(-1, B, -1).contiguous()
        c0 = self.c0.expand(-1, B, -1).contiguous()
        lstm_out, _ = self.lstm(combined, (h0, c0))

        if self.lstm_drop is not None and self.training:
            lstm_out = self.lstm_drop(lstm_out)

        if self.use_attention:
            attn_out = self.temp_attn(lstm_out)
            lstm_out = self.attn_norm(lstm_out + attn_out)

        if return_all_timesteps:
            all_logits = self.classifier(lstm_out).squeeze(-1)
            return all_logits
        else:
            last_hidden = lstm_out[:, -1, :]
            logit = self.classifier(last_hidden).squeeze(-1)
            return logit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"ResNet18 Backbone")
    print(f"{'='*60}")
    model = CNN_LSTM_BBox(
        hidden_dim=256, lstm_layers=1, dropout=0.5,
        max_objects=10, use_attention=True,
    )
    print(f"Trainable: {count_parameters(model):,}")
    print(f"Total:     {count_all_parameters(model):,}")

    B, T = 2, 16
    frames = torch.randn(B, T, 3, 224, 224)
    bboxes = torch.rand(B, T, 10, 4)
    bbox_feat = torch.randn(B, T, 10, 17)
    cats = torch.randint(0, 8, (B, T, 10))
    mask = torch.zeros(B, T, 10, dtype=torch.bool)
    mask[:, :, :3] = True

    logit = model(frames, bboxes, bbox_feat, cats, mask)
    prob = torch.sigmoid(logit)
    print(f"Input:  frames {frames.shape}")
    print(f"Output: logit {logit.shape}, prob = {[f'{p:.3f}' for p in prob.tolist()]}")
