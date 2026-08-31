"""Modality-conditioned semantic calibration of the class prototypes.

THE PROBLEM this exists for
    Oral diagnosis is cross-modality, and the same class NAME denotes different visual evidence in
    each modality: "dental caries" is a discoloured, cavitated enamel surface in an intraoral
    photograph but a radiolucent shadow on a periapical radiograph. In an open-vocabulary detector
    the text embedding IS the classifier -- the head scores a region by its cosine similarity to the
    prototype of each class name. The text tower emits ONE prototype per name, shared across every
    modality, so that single vector is pulled toward an average of visually incompatible appearances
    and matches none of them well. Training on a pooled multi-modal set therefore conflates the text
    semantics of a finding with its modality-specific visual semantics.

WHAT THIS DOES
    Conditions the prototypes on the modality of the image actually being scored:

      1. MODALITY TOKENS. `num_modality_tokens` learnable queries attention-pool the vision tower's
         deepest feature map -> a per-image modality descriptor. ConvNeXt is a CNN and has no token
         stream, so the "special token" is realised as a learnable query + attention pooling rather
         than a CLS slot. The descriptor is INFERRED from the image; the ground-truth modality label
         is never used, so this works at test time on an unlabelled image.
      2. CALIBRATION. The class prototypes cross-attend to those tokens (Q = text, K = V = modality
         tokens) -> a per-class, per-modality delta.
      3. RESIDUAL + RENORMALISE. prototype' = normalise(prototype + gamma * delta).

WHY THE GATE IS ZERO-INITIALISED
    `gamma` starts at 0, so at step 0 the module is EXACTLY the identity: prototype' == prototype
    (the text tower already L2-normalises its output, so re-normalising a unit vector is a no-op).
    The pretrained text tower is therefore untouched at initialisation and the module can only earn
    its way in. Without this, inserting a randomly-initialised cross-attention would corrupt the
    prototypes on the first forward pass -- exactly the failure the staged recipe exists to avoid.

COST NOTE
    This makes the prototypes image-dependent, so they can no longer be precomputed once and cached
    for the whole dataset -- which is WeDetect's "detection as retrieval" efficiency argument. The
    expensive part (the BERT forward) can still be cached via `reparameterize()`; only this
    cross-attention (n_cls x num_modality_tokens, tiny) runs per image.
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class ModalityCalibration(BaseModule):
    """Vision modality tokens -> cross-attention -> modality-calibrated text prototypes.

    Args:
        embed_dims: text prototype dim (768 for the WeDetect text tower).
        vision_dims: channels of the vision feature map used (1024 = ConvNeXt-B's deepest).
        num_modality_tokens: how many learnable queries pool the vision map.
        num_heads: attention heads for both the pooling and the calibration attention.
        dropout: attention dropout. Default 0.0 -- the text tower's stock 0.1 was a measured
            regression (MACRO -0.0104) and everything here follows that finding.
        vision_index: which backbone feature map to pool. -1 = deepest (most semantic, coarsest),
            which is what carries modality identity.
    """

    def __init__(self,
                 embed_dims: int = 768,
                 vision_dims: int = 1024,
                 num_modality_tokens: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 vision_index: int = -1,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.vision_index = vision_index
        self.num_modality_tokens = num_modality_tokens

        # (1) the modality "special token": learnable queries that pool the vision map
        self.mod_queries = nn.Parameter(torch.randn(num_modality_tokens, embed_dims) * 0.02)
        self.vis_proj = nn.Sequential(
            nn.Linear(vision_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )
        self.pool_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)

        # (2) calibration: text prototypes attend to the modality tokens
        self.q_norm = nn.LayerNorm(embed_dims)
        self.calib_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)

        # (3) zero-init gate -> identity at step 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def modality_tokens(self, img_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        """-> [B, num_modality_tokens, D]. Inferred from the image; no modality label used."""
        v = img_feats[self.vision_index]              # [B, Cv, H, W]
        b = v.shape[0]
        v = v.flatten(2).transpose(1, 2)             # [B, H*W, Cv]
        v = self.vis_proj(v)                         # [B, H*W, D]
        q = self.mod_queries.unsqueeze(0).expand(b, -1, -1)   # [B, M, D]
        tok, _ = self.pool_attn(q, v, v, need_weights=False)  # [B, M, D]
        return tok

    def forward(self, img_feats: Sequence[torch.Tensor],
                txt_feats: torch.Tensor) -> torch.Tensor:
        """img_feats: backbone maps. txt_feats: [B, n_cls, D] (or [n_cls, D] when precomputed)."""
        tok = self.modality_tokens(img_feats)
        squeeze_back = False
        if txt_feats.dim() == 2:                     # precomputed prototypes, shared across batch
            txt_feats = txt_feats.unsqueeze(0).expand(tok.shape[0], -1, -1)
            squeeze_back = True
        elif txt_feats.shape[0] == 1 and tok.shape[0] > 1:
            txt_feats = txt_feats.expand(tok.shape[0], -1, -1)

        delta, _ = self.calib_attn(self.q_norm(txt_feats), tok, tok, need_weights=False)
        out = F.normalize(txt_feats + self.gamma * delta, dim=-1)
        if squeeze_back and out.shape[0] == 1:
            out = out.squeeze(0)
        return out
