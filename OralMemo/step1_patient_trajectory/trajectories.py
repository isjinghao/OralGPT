from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path


def _stage_input(stage: dict) -> dict:
    """抽取阶段的轨迹精简视图, 把每条QA标记为role=evidence
    输入: stage 单个阶段对象。
    输出: dict - 精简后的阶段(含 stage_id/order/modality/image_paths/qa_pairs)。
    """
    return {
        "stage_id": stage["stage_id"],
        "order": stage["order"],
        "stage_type": stage["stage_type"],
        "modality": stage["modality"],
        "image_paths": stage["image_paths"],
        "qa_pairs": [
            {
                "source_turn_id": turn["source_turn_id"],
                "human": turn["human"],
                "assistant": turn["assistant"],
                "image_paths": turn["image_paths"],
                "role": "evidence",
            }
            for turn in stage["qa_pairs"]
        ],
    }


def build_standard_trajectory(patient_stages: dict) -> dict:
    # 按阶段顺序生成 standard_full 轨迹
    return {
        "trajectory_id": f"{patient_stages['patient_id']}__standard_full",
        "patient_id": patient_stages["patient_id"],
        "trajectory_type": "standard_full",
        "stages": [_stage_input(stage) for stage in patient_stages["stages"]],
    }


def build_missing_modality_variants(standard: dict) -> list[dict]:
    # 分别删除 FP/DP/XR/CT/TMJ 各一个阶段, 并重排 order, 得到5个变体
    variants = []
    drops = {
        "no_fp": {"S1_FP"},
        "no_dp": {"S2_DP"},
        "no_xr": {"S3_XR_XLA"},
        "no_ct": {"S4_CT"},
        "no_tmj": {"S5_TMJ"},
    }
    for name, removed in drops.items():
        stages = [copy.deepcopy(s) for s in standard["stages"] if s["stage_id"] not in removed]
        for new_order, stage in enumerate(stages):
            stage["order"] = new_order
        variants.append(
            {
                "trajectory_id": f"{standard['patient_id']}__{name}",
                "patient_id": standard["patient_id"],
                "trajectory_type": name,
                "removed_stage_ids": sorted(removed),
                "stages": stages,
            }
        )
    return variants


def build_long_noisy_variant(
    standard: dict,
    noise_count: int = 3,
    seed: int = 42,
) -> dict:
    # 生成噪声变体 (分类噪声池、采样)
    stages = copy.deepcopy(standard["stages"])
    pool = json.loads(Path(__file__).with_name("noise_pool.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)

    k = max(0, min(noise_count, len(pool)))
    chosen = rng.sample(pool, k) if k else []

    populated = [i for i, s in enumerate(stages) if s["qa_pairs"]]
    target_pool = populated or list(range(len(stages)))

    fallback_order = list(target_pool)
    rng.shuffle(fallback_order)
    fb_ptr = 0

    for noise in chosen:
        target = None
        # 有模态的噪声尽量放到对应阶段
        modality = noise.get("modality")
        if modality:
            modal_targets = [i for i in target_pool if modality in stages[i]["modality"]]
            if modal_targets:
                target = modal_targets[rng.randrange(len(modal_targets))]
        # 没有模态或没有对应阶段的噪声随机放到一个阶段
        if target is None and fallback_order:
            target = fallback_order[fb_ptr % len(fallback_order)]
            fb_ptr += 1
        if target is None:
            continue
        stages[target]["qa_pairs"].append(
            {
                "source_turn_id": noise["id"],
                "human": noise["human"],
                "assistant": noise["assistant"],
                "image_paths": [],
                "role": "noise",
                "noise_category": noise["category"],
            }
        )

    return {
        "trajectory_id": f"{standard['patient_id']}__long_noisy",
        "patient_id": standard["patient_id"],
        "trajectory_type": "long_noisy",
        "noise_count": k,
        "stages": stages,
    }
