"""对数据集中的全部患者批量执行 Step1 + Step2。
用法:
    python -m bench.step2_evidence.run_all           # 跑全部患者, 默认断点续跑
    python -m bench.step2_evidence.run_all --force   # 忽略已有结果, 强制重跑全部
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bench.config import get_settings
from bench.step2_evidence.pipeline import (
    build_client,
    patient_output_root,
    process_patient,
)
from bench.step1_patient_trajectory.dataset import index_by_patient_id, load_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def already_done(output_root_for: Path) -> bool:
    return (output_root_for / "evidence" / "evidence.json").exists()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    index = index_by_patient_id(load_dataset(settings.dataset_json))
    patient_ids = list(index.keys())
    client = build_client(settings)
    failed: list[str] = []

    for patient_id in patient_ids:
        if not args.force and already_done(patient_output_root(settings, patient_id)):
            continue
        try:
            process_patient(index[patient_id], settings, client)
        except KeyboardInterrupt:
            break
        except Exception:
            failed.append(patient_id)
            continue

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
