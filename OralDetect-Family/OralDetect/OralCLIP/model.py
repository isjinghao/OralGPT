"""OralCLIP architecture — the definition the released `oralclip.pt` was trained with.

ConvNeXt-Base vision tower and a DentalBERT text tower projected into a shared 512-d space,
plus a learnable modality-query token that cross-attends over the vision feature map and
predicts which of 9 imaging modalities the image is.

The modality head is part of the released weights (`vision.mod_query`, `vision.mod_attn.*`,
`vision.mod_norm.*`, `vision.mod_head.*`). A definition without it loads under
`strict=False` and silently drops those tensors, so `load_oralclip` below uses `strict=True`.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# The 9-way modality vocabulary, in the order the head's logits are laid out.
MODALITIES = ["intraoral", "periapical_xray", "panoramic_xray", "CT",
              "histopathology", "cytology", "cephalometric_xray", "speech_mri", "others"]

# Preprocessing the released weights were evaluated with.
IMAGE_SIZE = 224
MAX_TEXT_LEN = 128
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]


class VisionEncoder(nn.Module):
    """ConvNeXt-Base + contrastive projection + learnable modality-query head."""

    def __init__(self, embed_dim: int = 512, n_modalities: int = 9, attn_heads: int = 8):
        super().__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.num_features = self.backbone.num_features          # 1024
        self.proj = nn.Linear(self.num_features, embed_dim)

        self.mod_query = nn.Parameter(torch.randn(1, 1, self.num_features) * 0.02)
        self.mod_attn = nn.MultiheadAttention(self.num_features, attn_heads, batch_first=True)
        self.mod_norm = nn.LayerNorm(self.num_features)
        self.mod_head = nn.Linear(self.num_features, n_modalities)

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)                    # (B, C, H, W)
        pooled = self.backbone.forward_head(feat_map, pre_logits=True)  # (B, C)
        img_emb = self.proj(pooled)                                     # (B, embed_dim)

        B = feat_map.shape[0]
        tokens = feat_map.flatten(2).transpose(1, 2)                    # (B, H*W, C)
        q = self.mod_query.expand(B, -1, -1)                            # (B, 1, C)
        attn_out, _ = self.mod_attn(q, tokens, tokens)                  # (B, 1, C)
        mod_logits = self.mod_head(self.mod_norm(attn_out.squeeze(1)))  # (B, n_modalities)
        return img_emb, mod_logits


class TextEncoder(nn.Module):
    """DentalBERT + a 2-layer projection. `proj_type` must stay 'mlp' for the released weights."""

    def __init__(self, model_path: str, embed_dim: int = 512, proj_type: str = "mlp"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        # The pooler is unused (the CLS hidden state is projected directly) and is absent from
        # the checkpoint, so dropping it is what lets the state_dict load strictly.
        if getattr(self.bert, "pooler", None) is not None:
            self.bert.pooler = None
        hidden = self.bert.config.hidden_size
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(hidden, 640, bias=False),
                nn.GELU(),
                nn.Linear(640, embed_dim, bias=False),
            )
        else:
            self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0])


class OralCLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, init_temp: float = 0.07):
        super().__init__()
        self.vision = vision_encoder
        self.text = text_encoder
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def forward(self, images, input_ids, attention_mask):
        img_emb, mod_logits = self.vision(images)
        txt_emb = self.text(input_ids, attention_mask)
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = scale * img_emb @ txt_emb.t()
        return logits_per_image, logits_per_image.t(), mod_logits


def load_oralclip(checkpoint: str, text_tower: str, device: str = "cpu",
                  embed_dim: int = 512, strict: bool = True):
    """Build the model and load `oralclip.pt` into it.

    checkpoint  OralCLIP/oralclip.pt from the model hub
    text_tower  OralCLIP/oralbert   from the model hub (a directory)

    Raises on any key or shape mismatch. That is deliberate: torch drops mismatched keys
    silently under strict=False, which yields a partly-random model that still runs and
    still produces plausible-looking similarities.
    """
    model = OralCLIP(VisionEncoder(embed_dim=embed_dim),
                     TextEncoder(text_tower, embed_dim=embed_dim, proj_type="mlp"))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("state_dict", ck))
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval()
