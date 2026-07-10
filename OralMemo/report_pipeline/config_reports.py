from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import ROOT, BENCH_ROOT, get_settings, Settings  # noqa: F401


@dataclass(frozen=True)
class ReportSettings:
    bench_root: Path
    reports_dir: Path            # 原始 PDF 目录
    report_output_root: Path     # 每篇报告的流水线产物
    report_dataset_json: Path    # 汇总的 SFT 数据集


def get_report_settings() -> ReportSettings:
    return ReportSettings(
        bench_root=BENCH_ROOT,
        reports_dir=BENCH_ROOT / "reports",
        report_output_root=BENCH_ROOT / "outputs" / "report",
        report_dataset_json=BENCH_ROOT / "report_dataset.json",
    )


def name_paths(settings: ReportSettings, name: str) -> dict:
    out_dir = settings.report_output_root / name
    return {
        "out_dir": out_dir,
        "raw_dir": out_dir / "raw",
        "images_dir": out_dir / "images",
    }
