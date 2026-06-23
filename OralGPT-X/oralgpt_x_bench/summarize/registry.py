"""Benchmark registry for summarize step."""

from __future__ import annotations

BENCHMARK_REGISTRY = {
    "cbct": {
        "metrics": ["ssim", "psnr", "nmi", "mae", "lpips"],
        "primary": ["ssim", "psnr", "nmi"],
        "groups": ["task_type", "volume_id"],
    },
    "ortho": {
        "metrics": ["ssim", "psnr", "nmi", "mae", "lpips"],
        "primary": ["ssim", "lpips"],
        "groups": ["modality", "batch"],
    },
    "mri": {
        "metrics": ["ssim", "psnr", "nmi", "mae", "lpips"],
        "primary": ["ssim", "nmi"],
        "groups": ["task_type", "cohort"],
    },
    "t2i": {
        "metrics": ["consistency", "realism", "clinical_plausibility"],
        "primary": ["consistency", "clinical_plausibility"],
        "groups": ["category"],
    },
}
