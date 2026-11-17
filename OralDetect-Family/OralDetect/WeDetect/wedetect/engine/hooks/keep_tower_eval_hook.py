"""Keep a tower in eval() mode while still training its weights.

`MultiModalYOLOBackbone._freeze_modules()` calls `.eval()` on every submodule of a tower listed in
`frozen_modules`, and does nothing when `frozen_modules=[]`. So the moment a staged recipe unfreezes
a tower, that tower also leaves eval() and enters train() mode — which turns DentalBERT's dropout
(hidden 0.1 / attention 0.1) back on.

That mode switch, not the learning rate, is what causes the "unfreeze shock". Measured with a
forward-only probe (identical weights, identical batches, no optimizer, zero weight updates), simply
flipping the towers train()<->eval() moves:

    loss_cls   3.521 -> 16.591   (4.71x)
    loss_bbox  2.939 ->  3.274   (1.11x)
    loss_dfl   4.264 ->  4.429   (1.04x)

i.e. the entire shock lands on the region-text alignment term, exactly as a stochastic text
embedding would predict. It matches the real run (loss_cls 14.6 -> 59.9, 4.10x). Because no weight
was ever updated in the probe, **no learning-rate schedule can fix this** — and indeed giving the
towers a 10x lower lr left the loss curve untouched (92.15/90.83 at iter 50, 78.89/79.17 at 100).

This hook re-asserts eval() on the named towers before every train iteration. Their parameters keep
receiving gradients and keep being updated; only the stochastic forward behaviour (dropout,
stochastic depth) stays off, so the class-text embeddings remain deterministic across steps.
"""
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class KeepTowerEvalHook(Hook):
    """Force the named backbone towers to stay in eval() mode during training.

    Args:
        towers: attribute names under ``model.backbone`` to hold in eval mode.
            Defaults to the text tower only — it is the one with dropout, and the probe shows the
            vision tower contributes almost nothing to the shock (loss_bbox 1.11x, loss_dfl 1.04x).
    """

    priority = 'NORMAL'

    def __init__(self, towers=('text_model', )):
        self.towers = tuple(towers)

    def _apply(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        for name in self.towers:
            tower = getattr(model.backbone, name, None)
            if tower is not None:
                tower.eval()

    def before_train_epoch(self, runner):
        self._apply(runner)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        # the train loop calls model.train() once per epoch, but PipelineSwitchHook and other hooks
        # can also flip modes, so re-assert every iteration — it is a no-op when already in eval()
        self._apply(runner)
