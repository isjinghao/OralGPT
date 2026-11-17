"""CocoMetric + per-modality / per-MODALITY-macro mAP for the unified 86-class OralDetect COCO.

WHAT `coco/bbox_mAP` ACTUALLY IS — do not call it "micro".
    COCO's mAP is *already* a macro average, but over CLASSES: `COCOeval.summarize()` computes
    `np.mean(s[s > -1])` over a `[T x R x K x A x M]` array whose K axis is the category. Every
    class contributes equally; box counts do not enter.

    That is not the problem. The problem is that a CLASS-macro is not a MODALITY-macro, because our
    five modalities contribute wildly unequal numbers of classes:

        panoramic   48 / 90 class slots = 53.3%   (of which 32 are FDI tooth indices + 4 quadrants)
        intraoral   21 / 90             = 23.3%
        periapical  11 / 90             = 12.2%
        cytology     9 / 90             = 10.0%
        histology     1 / 90            =  1.1%    <- vs 20% under a modality-macro. 18x.

    So ~40% of the standard COCO mAP on OralDetect is tooth/quadrant *indexing* on panoramic
    radiographs, simply because every tooth is individually indexed on every radiograph.

    Verified empirically (1024 3-stage, test bench, observed bbox_mAP = 0.632):
        predict by weighting per-modality mAPs by CLASS count -> 0.624   (off by 0.008)  <- this
        predict by weighting per-modality mAPs by BOX   count -> 0.717   (off by 0.085)
    It tracks class counts, not box counts, exactly as the above says it must.

This metric therefore adds a macro over MODALITIES, so checkpoint selection can be driven by it.

Scoring matches tools/per_modality_eval.py exactly: COCOeval restricted to each
modality's images and to the categories present in that modality's GT (the open-vocab sigmoid head
gives vocab-size-independent per-class scores, so a single 86-class prediction dump can be scored
per modality).

Emitted keys (prefix `coco/`):
    bbox_mAP        CocoMetric's value, unchanged — a CLASS-macro over all 86 classes. Kept for the
                    record only; we do not report or select on it (see CLAUDE.md §3).
    <modality>_mAP  one per modality (itself a class-macro within that modality)
    macro_mAP       MODALITY-macro: mean of all 5 modalities            <- what we report
    macro4_mAP      MODALITY-macro excluding histology                  <- what we select on
                    (histology val = 29 imgs / 1 class; swings +-0.13 between adjacent epochs)
"""
import contextlib
import io
import json
import os.path as osp
import tempfile

from mmengine.logging import MMLogger
from pycocotools.cocoeval import COCOeval

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS

MODALITIES = ('intraoral', 'panoramic', 'periapical', 'cytology', 'histology')


@METRICS.register_module()
class PerModalityCocoMetric(CocoMetric):
    """CocoMetric that also reports per-modality mAP and the MACRO aggregates."""

    def __init__(self, *args, macro_exclude=('histology', ), **kwargs):
        super().__init__(*args, **kwargs)
        # only affects the `macro4_mAP` key; `macro_mAP` always covers all 5
        self.macro_exclude = tuple(macro_exclude)

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        metrics = super().compute_metrics(results)

        coco = self._coco_api
        if coco is None or not coco.dataset.get('images'):
            return metrics
        # Checking only images[0] is not enough: the v2 rebuild left `modality` off every
        # periapical record while the older records kept it, so the metric silently scored 4
        # modalities and selected checkpoints on a macro that never saw periapical. Fail loudly
        # instead -- a missing modality is a corrupt GT file, not a reason to score less.
        missing = [im['file_name'] for im in coco.dataset['images'] if not im.get('modality')]
        if missing:
            raise ValueError(
                f'{len(missing)} of {len(coco.dataset["images"])} GT images carry no `modality` '
                f'field, so their modality would be dropped from the macro silently. '
                f'First few: {missing[:3]}. Run tools/fix_modality_field.py --apply.')

        _, preds = zip(*results)
        with tempfile.TemporaryDirectory() as tmp:
            files = self.results2json(preds, osp.join(tmp, 'permod'))
            with open(files['bbox']) as f:
                dets = json.load(f)

        mod_of = {im['id']: im.get('modality') for im in coco.dataset['images']}
        cats_of = {}
        for ann in coco.dataset['annotations']:
            cats_of.setdefault(mod_of.get(ann['image_id']), set()).add(ann['category_id'])

        per = {}
        for m in MODALITIES:
            imgs = {i for i, mm in mod_of.items() if mm == m}
            cats = cats_of.get(m, set())
            if not imgs or not cats:
                continue
            sub = [d for d in dets if d['image_id'] in imgs]
            if not sub:
                per[m] = 0.0
                continue
            with contextlib.redirect_stdout(io.StringIO()):   # COCOeval is very chatty
                dt = coco.loadRes(sub)
                e = COCOeval(coco, dt, 'bbox')
                e.params.imgIds = sorted(imgs)
                e.params.catIds = sorted(cats)
                e.evaluate()
                e.accumulate()
                e.summarize()
            per[m] = float(e.stats[0])

        if not per:
            return metrics
        for m, v in per.items():
            metrics[f'{m}_mAP'] = round(v, 4)
        metrics['macro_mAP'] = round(sum(per.values()) / len(per), 4)
        keep = [v for m, v in per.items() if m not in self.macro_exclude]
        if keep:
            metrics['macro4_mAP'] = round(sum(keep) / len(keep), 4)

        logger.info(
            'per-modality mAP: ' + '  '.join(f'{m}={v:.4f}' for m, v in per.items())
            + f"  ||  MACRO={metrics['macro_mAP']}"
            + f"  MACRO4={metrics.get('macro4_mAP')}"
            + f"  || class-macro(coco/bbox_mAP, not reported)={metrics.get('bbox_mAP')}")
        return metrics
