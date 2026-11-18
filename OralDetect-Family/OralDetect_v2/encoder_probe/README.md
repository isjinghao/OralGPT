# encoder_probe

Which vision encoder, and which pretraining, gives the most useful features for dental images?
Encoders are frozen; only a small readout is trained, so differences come from the pretrained weights.

## Design

Two backbones, pretraining varied inside each (same architecture and size, different weights):

| backbone | encoder config | pretraining |
|---|---|---|
| ConvNeXt-B (88M) | `convnext_b_in22k` | supervised, ImageNet-22k (official) |
| | `convnext_b_wedetect` | WeDetect base detector: the weights OralCLIP started from |
| | `convnext_b_oralclip` | OralCLIP v15: WeDetect init, then 710K dental image-text pairs |
| ViT-B/16 (86M) | `vit_b_dinov3` | self-supervised, general images |
| | `vit_b_biomedclip` | biomedical image-text contrastive |
| | `vit_b_medsam` | medical promptable segmentation |

Probe `tooth_fdi`: every ground-truth tooth box of IOP-Compass is RoI-pooled from the encoder's last
feature map and classified into its FDI code. Readouts: `linear`, `mlp`, `mlp+geom` (feature plus box
position/size), and a `geom`-only floor. Reported: FDI accuracy and its parts (position, quadrant, arch, side).

Compare within a backbone. Across backbones the feature maps differ (ConvNeXt stride 32 and 1024-d,
ViT stride 16 and 768-d), so cross-family gaps are not controlled.

## Run

```bash
bash launch/submit.sh 01_fetch_weights                       # CPU: download + forward-pass check
for f in configs/encoders/*.yaml; do ENCODER=$(basename $f .yaml) bash launch/submit.sh 02_extract; done
bash launch/submit.sh 03_probe                               # -> <exp_root>/exps_outputs_v2/encoder_probe/tooth_fdi/report.md
```

Add an encoder by adding a file under `configs/encoders/`; add a loader in `encprobe/encoders.py` only
if its weights come in a new format.
