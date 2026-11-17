"""Build the OralDetect init checkpoint: take the WeDetect base, inject OralCLIP's
dental-CLIP-trained ConvNeXt vision backbone, then DROP the XLM-R text encoder so the init
pairs cleanly with the DentalBertLanguageBackbone (which loads its own DentalBERT weights
from HF at build time).

This is how `oraldetect_init.pth` on the model hub was produced.

The vision keys are remapped from timm's convnext layout to Facebook/mmpretrain's, and
copied only where the shapes match; unmatched WeDetect tensors keep the base init. After
injection every `backbone.text_model.model.*` (XLM-R encoder) key is deleted, while
`backbone.text_model.head.*`, the neck and the bbox_head are kept."""
import argparse, re, torch

ap = argparse.ArgumentParser(
    description="Inject OralCLIP's ConvNeXt vision backbone into a WeDetect init checkpoint.")
ap.add_argument("--wedetect", required=True, help="WeDetect base checkpoint (wedetect_base.pth)")
ap.add_argument("--oralclip", required=True, help="OralCLIP checkpoint holding vision.backbone.*")
ap.add_argument("--out", required=True, help="where to write the init checkpoint")
args = ap.parse_args()
WD, OC, OUT = args.wedetect, args.oralclip, args.out

WD_PREF = "backbone.image_model.model."
TEXT_MODEL_PREF = "backbone.text_model.model."


def timm_to_fb(k):
    """timm convnext key (no prefix) -> Facebook convnext key (no prefix)."""
    if k.startswith("stem."):                       # stem.0=conv stem.1=norm
        return "downsample_layers.0." + k[len("stem."):]
    m = re.match(r"stages\.(\d+)\.downsample\.(.*)", k)
    if m:                                            # stage i downsample -> downsample_layers.i
        return f"downsample_layers.{m.group(1)}.{m.group(2)}"
    m = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.(.*)", k)
    if m:
        i, j, rest = m.groups()
        rest = (rest.replace("conv_dw", "dwconv")
                    .replace("mlp.fc1", "pwconv1")
                    .replace("mlp.fc2", "pwconv2"))
        return f"stages.{i}.{j}.{rest}"
    if k.startswith("head.norm."):                   # timm final norm -> FB final norm
        return "norm." + k[len("head.norm."):]
    return None


wd = torch.load(WD, map_location="cpu", weights_only=False)
sd = wd["state_dict"]
oc = torch.load(OC, map_location="cpu", weights_only=False)["model"]
oc_vb = {k[len("vision.backbone."):]: v for k, v in oc.items() if k.startswith("vision.backbone.")}

wd_img = {k[len(WD_PREF):] for k in sd if k.startswith(WD_PREF)}
copied, shape_mismatch, no_target, unmapped = 0, [], [], []
for k, v in oc_vb.items():
    fb = timm_to_fb(k)
    if fb is None:
        unmapped.append(k); continue
    if fb not in wd_img:
        no_target.append((k, fb)); continue
    tgt = WD_PREF + fb
    if tuple(sd[tgt].shape) != tuple(v.shape):
        shape_mismatch.append((k, fb, tuple(v.shape), tuple(sd[tgt].shape))); continue
    sd[tgt] = v.clone()
    copied += 1

covered = {WD_PREF + timm_to_fb(k) for k in oc_vb if timm_to_fb(k)}
wd_uncovered = [k for k in sd if k.startswith(WD_PREF) and k not in covered]

print(f"OralCLIP vision.backbone tensors : {len(oc_vb)}")
print(f"WeDetect image_model tensors     : {len(wd_img)}")
print(f"COPIED (OralCLIP -> WeDetect)    : {copied}")
print(f"unmapped OralCLIP keys           : {len(unmapped)}  {unmapped[:6]}")
print(f"mapped but no WeDetect target    : {len(no_target)}  {no_target[:6]}")
print(f"shape mismatch                   : {len(shape_mismatch)}  {shape_mismatch[:6]}")
print(f"WeDetect image_model NOT replaced: {len(wd_uncovered)}  {[k[len(WD_PREF):] for k in wd_uncovered][:8]}")

# Drop the XLM-R text encoder so load_from does not conflict with the DentalBert backbone,
# which loads its own weights via AutoModel.from_pretrained. Keep text_model.head.* etc.
text_drop = [k for k in sd if k.startswith(TEXT_MODEL_PREF)]
for k in text_drop:
    del sd[k]
text_head_kept = [k for k in sd if k.startswith("backbone.text_model.") and not k.startswith(TEXT_MODEL_PREF)]
print(f"\nDROPPED backbone.text_model.model.* keys : {len(text_drop)}")
print(f"KEPT backbone.text_model.* (non-model)   : {len(text_head_kept)}  {text_head_kept[:6]}")

torch.save({"meta": wd.get("meta", {}), "state_dict": sd}, OUT)
print(f"\nsaved -> {OUT}")
