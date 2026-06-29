"""对数据集中的全部患者批量执行 Step1 + Step2 流水线。

设计目标:
* 健壮   -- 单个患者失败时会被跳过; 批处理继续推进,
            一条坏记录绝不会阻塞其余患者(失败与否通过退出码反映)。
* 可恢复 -- ``--skip-existing`` 跳过已处理的患者; 任何失败的患者
            都可以用以下命令单独重跑::

                python -m bench.step2_evidence.run_one <patient_id>

用法:
    python -m bench.step2_evidence.run_all
    python -m bench.step2_evidence.run_all --limit 5            # 仅对前 5 个做冒烟测试
    python -m bench.step2_evidence.run_all --skip-existing      # 续跑, 跳过已完成的患者
    python -m bench.step2_evidence.run_all --start-at group3__X # 跳过 X 之前的所有患者
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
    parser = argparse.ArgumentParser(description="处理全部患者(Step1 + Step2)。")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个患者。")
    parser.add_argument("--start-at", default=None, help="跳过患者, 直到到达此标识。")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过 evidence.json 已存在的患者。",
    )
    return parser.parse_args(argv)


def select_patient_ids(all_ids: list[str], args: argparse.Namespace) -> list[str]:
    """应用 --start-at 和 --limit, 推导出实际处理的患者标识列表。"""
    ids = list(all_ids)
    if args.start_at is not None:
        if args.start_at not in ids:
            raise SystemExit(f"--start-at id not found in dataset: {args.start_at}")
        ids = ids[ids.index(args.start_at):]
    if args.limit is not None:
        ids = ids[: args.limit]
    return ids


def already_done(output_root_for: Path) -> bool:
    """若患者的 evidence.json 已存在, 即视为已完成。"""
    return (output_root_for / "evidence" / "evidence.json").exists()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    index = index_by_patient_id(load_dataset(settings.dataset_json))
    patient_ids = select_patient_ids(list(index.keys()), args)

    client = build_client(settings)

    failed: list[str] = []

    for patient_id in patient_ids:
        if args.skip_existing and already_done(patient_output_root(settings, patient_id)):
            continue

        try:
            process_patient(index[patient_id], settings, client)
        except KeyboardInterrupt:
            break
        except Exception:  # noqa: BLE001 - 单患者出错不中断整批
            failed.append(patient_id)
            continue

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
