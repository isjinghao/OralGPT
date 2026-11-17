#!/usr/bin/env python
"""Encode dental images and text with OralCLIP, and score one against the other.

    # zero-shot: which label fits each image
    python infer.py --checkpoint <oralclip.pt> --text-tower <oralbert/> \
        --images a.jpg b.jpg --labels "dental caries" "healthy tooth" "dental calculus"

    # embeddings only
    python infer.py --checkpoint ... --text-tower ... --images *.jpg --save emb.npz

As a library:

    from model import load_oralclip
    from infer import encode_images, encode_texts
    m = load_oralclip(ckpt, tower, device="cuda")
    img = encode_images(m, ["a.jpg"], device="cuda")      # (N, 512), L2-normalised
    txt = encode_texts(m, tower, ["dental caries"], device="cuda")
    sim = img @ txt.T
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer

from model import IMAGE_SIZE, MAX_TEXT_LEN, MODALITIES, PIXEL_MEAN, PIXEL_STD, load_oralclip

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Must match what the released weights were evaluated with: resize the short side to
# 224 * 1.14 then centre-crop 224.
preprocess = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
])


@torch.no_grad()
def encode_images(model, paths, device="cpu", batch_size=32, return_modality=False):
    """L2-normalised image embeddings, (N, 512). With return_modality, also (N, 9) logits."""
    embs, mods = [], []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                             for p in paths[i:i + batch_size]]).to(device)
        emb, mod = model.vision(batch)
        embs.append(F.normalize(emb, dim=-1).cpu())
        mods.append(mod.cpu())
    embs = torch.cat(embs)
    return (embs, torch.cat(mods)) if return_modality else embs


@torch.no_grad()
def encode_texts(model, text_tower, texts, device="cpu", batch_size=64):
    """L2-normalised text embeddings, (N, 512)."""
    tok = AutoTokenizer.from_pretrained(text_tower)
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=MAX_TEXT_LEN, return_tensors="pt").to(device)
        emb = model.text(enc["input_ids"], enc["attention_mask"])
        out.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description="Encode and score with OralCLIP.")
    ap.add_argument("--checkpoint", required=True, help="OralCLIP/oralclip.pt")
    ap.add_argument("--text-tower", required=True, help="OralCLIP/oralbert (a directory)")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=[],
                    help="candidate texts; omit to only write embeddings")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save", help="write embeddings to this .npz")
    a = ap.parse_args()

    model = load_oralclip(a.checkpoint, a.text_tower, device=a.device)
    print(f"loaded {a.checkpoint} on {a.device}")

    img, mod = encode_images(model, a.images, a.device, a.batch_size, return_modality=True)
    txt = encode_texts(model, a.text_tower, a.labels, a.device) if a.labels else None

    if txt is not None:
        sim = (img @ txt.T) * model.logit_scale.exp().clamp(max=100.0).cpu()
        prob = sim.softmax(dim=-1)
        w = max(len(s) for s in a.labels) + 2
        print(f"\n{'image':<34}{'modality':<20}" + "".join(f"{s:>{w}}" for s in a.labels))
        print("-" * (54 + w * len(a.labels)))
        for i, p in enumerate(a.images):
            name = p.split("/")[-1]
            m = MODALITIES[mod[i].argmax().item()]
            print(f"{name[:32]:<34}{m:<20}"
                  + "".join(f"{v:>{w}.3f}" for v in prob[i].tolist()))
        print("\nColumns are softmax over the labels given, so they say which of THESE labels")
        print("fits best -- not whether any of them fits at all.")
    else:
        for i, p in enumerate(a.images):
            print(f"  {p.split('/')[-1][:40]:<42}{MODALITIES[mod[i].argmax().item()]}")

    if a.save:
        import numpy as np
        d = {"image_paths": np.array(a.images), "image_emb": img.numpy(),
             "modality_logits": mod.numpy()}
        if txt is not None:
            d["labels"] = np.array(a.labels)
            d["text_emb"] = txt.numpy()
        np.savez(a.save, **d)
        print(f"\nembeddings -> {a.save}")


if __name__ == "__main__":
    raise SystemExit(main())
