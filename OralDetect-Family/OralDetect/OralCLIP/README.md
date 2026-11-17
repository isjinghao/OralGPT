# OralCLIP

Oral vision–language model. ConvNeXt-Base + DentalBERT in a shared 512-d space, plus a head
that predicts which of 9 imaging modalities an image is. It is the vision tower OralDetect is
built on, and it works on its own for zero-shot classification, retrieval and embeddings.

## Weights

```bash
hf download OralGPT/OralDetect-Family --local-dir weights
# weights/OralCLIP/oralclip.pt   the model
# weights/OralCLIP/oralbert/     the text tower, required at run time
```

## Use

```bash
pip install torch torchvision timm transformers pillow

python infer.py --checkpoint weights/OralCLIP/oralclip.pt --text-tower weights/OralCLIP/oralbert \
    --images tooth.jpg --labels "dental caries" "dental calculus" "healthy tooth"
```

Weight loading example:

```python
from model import load_oralclip
from infer import encode_images, encode_texts

m = load_oralclip("weights/OralCLIP/oralclip.pt", "weights/OralCLIP/oralbert", device="cuda")
img = encode_images(m, ["tooth.jpg"], device="cuda")                    # (N, 512), L2-normalised
txt = encode_texts(m, "weights/OralCLIP/oralbert", ["dental caries"], device="cuda")
sim = img @ txt.T
```

`encode_images(..., return_modality=True)` also returns the 9-way modality logits, ordered as
`model.MODALITIES`.

## Files

| | |
|---|---|
| `model.py` | the architecture, and `load_oralclip()` |
| `infer.py` | preprocessing, `encode_images` / `encode_texts`, and the CLI |

## Notes

`load_oralclip` loads with `strict=True` and raises on any mismatch. The released weights carry a
modality-query head (`vision.mod_query`, `vision.mod_attn.*`, `vision.mod_norm.*`,
`vision.mod_head.*`).
