import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models

from utils import NUM_CATEGORIES


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
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
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


class CNN3D_Danger(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        dropout=0.5,
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
        use_temporal_aux=False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.use_object_branch = use_object_branch
        self.use_object_interaction = use_object_interaction
        self.use_temporal_aux = use_temporal_aux

        backbone = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.DEFAULT)

        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.global_feat_dim = 512

        for p in self.stem.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = False

        self.global_proj = nn.Sequential(
            nn.Linear(self.global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
        )

        classifier_input_dim = hidden_dim

        if use_object_branch:
            self.cat_embed = nn.Embedding(NUM_CATEGORIES + 1, category_embed_dim)

            bbox_input_dim = 17 + category_embed_dim
            self.bbox_encoder = nn.Sequential(
                nn.Linear(bbox_input_dim, bbox_feat_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),
            )

            object_dim = bbox_feat_dim

            if use_object_interaction:
                self.obj_interaction = ObjectInteraction(
                    object_dim,
                    n_heads=object_interaction_heads,
                    n_layers=object_interaction_layers,
                    dropout=dropout * 0.3,
                )

            self.obj_attention = ObjectAttention(object_dim, hidden_dim=64)

            self.obj_temporal_lstm = nn.LSTM(
                input_size=object_dim,
                hidden_size=object_dim,
                num_layers=obj_temporal_layers,
                batch_first=True,
                dropout=obj_temporal_dropout if obj_temporal_layers > 1 else 0.0,
            )
            n_heads = min(attention_heads, max(1, object_dim // 16))
            while object_dim % n_heads != 0:
                n_heads -= 1
            self.obj_temporal_attn = TemporalAttention(
                object_dim, num_heads=n_heads,
                dropout=dropout * 0.3, causal=use_causal_attention,
            )
            self.obj_temporal_norm = nn.LayerNorm(object_dim)

            self.obj_proj = nn.Sequential(
                nn.Linear(object_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3),
            )
            classifier_input_dim += hidden_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        if use_temporal_aux:
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.temporal_head = nn.Sequential(
                nn.Linear(self.global_feat_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1),
            )

    def _extract_3d_features(self, x):
        if not self.training:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            with torch.no_grad():
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        temporal_feats = None
        if self.use_temporal_aux and self.training:
            sp = self.spatial_pool(x)
            temporal_feats = sp.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        x = self.avgpool(x)
        return x.flatten(1), temporal_feats

    def forward(self, frames, bboxes=None, bbox_features=None, categories=None,
                obj_mask=None, flow=None, return_aux=False, **kwargs):
        B, T = frames.shape[:2]

        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        global_feat, temporal_feats = self._extract_3d_features(x)
        global_feat = self.global_proj(global_feat)

        parts = [global_feat]

        if self.use_object_branch and bbox_features is not None:
            cats_flat = categories.view(B * T, self.max_objects)
            cat_emb = self.cat_embed(cats_flat)
            bbox_flat = bbox_features.view(B * T, self.max_objects, -1)
            bbox_input = torch.cat([bbox_flat, cat_emb], dim=-1)
            bbox_enc = self.bbox_encoder(bbox_input)

            if self.use_object_interaction:
                bbox_enc = self.obj_interaction(
                    bbox_enc, obj_mask.view(B * T, self.max_objects)
                )

            obj_context = self.obj_attention(
                bbox_enc, obj_mask.view(B * T, self.max_objects)
            )
            obj_context = obj_context.view(B, T, -1)

            obj_temporal, _ = self.obj_temporal_lstm(obj_context)
            attn_out = self.obj_temporal_attn(obj_temporal)
            obj_temporal = self.obj_temporal_norm(obj_temporal + attn_out)
            obj_feat = obj_temporal[:, -1, :]

            obj_feat = self.obj_proj(obj_feat)
            parts.append(obj_feat)

        combined = torch.cat(parts, dim=-1)
        logit = self.classifier(combined).squeeze(-1)
        if return_aux and temporal_feats is not None:
            temporal_logits = self.temporal_head(temporal_feats).squeeze(-1)
            return logit, temporal_logits
        return logit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"3D Backbone: r2plus1d_18")
    print(f"{'='*60}")
    model = CNN3D_Danger(
        hidden_dim=256, dropout=0.5,
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
