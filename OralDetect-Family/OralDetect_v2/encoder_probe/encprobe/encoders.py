"""Frozen encoders behind one interface: `model(x) -> (B, C, h, w)`, the last feature map.

x is an RGB batch in [0, 1]; each wrapper applies its own normalisation.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, body: nn.Module, kind: str, norm: dict):
        super().__init__()
        self.body, self.kind = body, kind
        self.register_buffer("mean", torch.tensor(norm["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(norm["std"]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.kind == "convnext":
            return self.body.forward_features(x)                              # (B, 1024, H/32, W/32)
        if self.kind == "vit":
            t = self.body.forward_features(x)[:, self.body.num_prefix_tokens:]  # drop cls / register tokens
            h = x.shape[-2] // 16
            return t.transpose(1, 2).reshape(t.shape[0], -1, h, t.shape[1] // h)
        if self.kind == "sam":                                                # hidden state before the neck
            hs = self.body(pixel_values=x, output_hidden_states=True).hidden_states[-1]
            return hs.permute(0, 3, 1, 2)                                     # (B, h, w, C) -> (B, C, h, w)
        raise ValueError(self.kind)


def _fb_to_timm(k: str) -> str | None:
    """Facebook / mmpretrain ConvNeXt key -> timm key (inverse of OralDetect's build_oraldetect_init)."""
    m = re.match(r"downsample_layers\.0\.(.*)", k)
    if m:
        return "stem." + m.group(1)
    m = re.match(r"downsample_layers\.(\d+)\.(.*)", k)
    if m:
        return f"stages.{m.group(1)}.downsample.{m.group(2)}"
    m = re.match(r"stages\.(\d+)\.(\d+)\.(.*)", k)
    if m:
        i, j, rest = m.groups()
        rest = rest.replace("dwconv", "conv_dw").replace("pwconv1", "mlp.fc1").replace("pwconv2", "mlp.fc2")
        return f"stages.{i}.blocks.{j}.{rest}"
    return None


def _load_checked(model: nn.Module, sd: dict, what: str) -> None:
    """Load `sd`, and fail unless every backbone tensor outside the classifier head was supplied."""
    own = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in sd and not k.startswith("head.")]
    if missing:
        raise RuntimeError(f"{what}: {len(missing)} backbone tensors not loaded, e.g. {missing[:5]}")
    model.load_state_dict(sd, strict=False)
    print(f"{what}: loaded {len(sd)}/{len(own)} tensors")


def _load_lenient(path: str):
    """torch.load for checkpoints that pickle classes of packages not installed here (mmengine
    metadata in WeDetect's): unknown classes become empty dict stubs, tensors load as usual."""
    import pickle
    import types

    class _Meta(type):                       # any attribute looked up on the stub class is a no-op callable
        def __getattr__(cls, name):
            return lambda *a, **k: None

    class _Stub(dict, metaclass=_Meta):
        def __init__(self, *a, **k):
            super().__init__()

        def __setstate__(self, state):
            pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return _Stub

    mod = types.ModuleType("lenient_pickle")
    mod.__dict__.update({k: v for k, v in vars(pickle).items() if not k.startswith("__")})
    mod.Unpickler = _Unpickler
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=mod)


def build(cfg: dict, img_size: int) -> Encoder:
    import timm

    loader = cfg["loader"]
    if loader == "timm":
        body = timm.create_model(cfg["source"], pretrained=True, num_classes=0)
        kind = "convnext" if cfg["backbone"].startswith("convnext") else "vit"
    elif loader in ("wedetect", "oralclip"):
        body, kind = timm.create_model("convnext_base", pretrained=False, num_classes=0), "convnext"
        ck = _load_lenient(cfg["weights"])
        if loader == "oralclip":
            pre = "vision.backbone."
            sd = {k[len(pre):]: v for k, v in ck["model"].items() if k.startswith(pre)}
        else:
            pre = "backbone.image_model.model."
            sd = {_fb_to_timm(k[len(pre):]): v for k, v in ck["state_dict"].items() if k.startswith(pre)}
            sd.pop(None, None)
        _load_checked(body, sd, cfg["name"])
    elif loader == "open_clip":
        import open_clip
        body, kind = open_clip.create_model_from_pretrained(cfg["source"])[0].visual.trunk, "vit"
    elif loader == "hf_sam":
        from transformers import SamModel
        body, kind = SamModel.from_pretrained(cfg["source"]).vision_encoder, "sam"
    else:
        raise ValueError(loader)
    if kind == "vit" and hasattr(body, "set_input_size") and getattr(body, "pos_embed", None) is not None:
        body.set_input_size(img_size=(img_size, img_size))          # resample a learned position embedding
    if kind == "sam" and img_size != 1024:
        raise ValueError("the SAM encoder is run at its native 1024 only")
    return Encoder(body, kind, cfg["norm"]).eval().requires_grad_(False)


def build_finetuned(cfg: dict, ckpt: str, trained_size: int, img_size: int) -> Encoder:
    """An encoder with weights saved by finetune(): built at the size it was trained at (a learned
    position embedding has that shape), then switched to `img_size`."""
    model = build(cfg, trained_size)
    model.body.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    if img_size != trained_size:
        if model.kind == "sam":
            raise ValueError("the SAM encoder is run at its native 1024 only")
        if model.kind == "vit" and getattr(model.body, "pos_embed", None) is not None:
            model.body.set_input_size(img_size=(img_size, img_size))
    return model
