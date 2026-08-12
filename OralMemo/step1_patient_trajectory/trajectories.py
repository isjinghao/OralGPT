from __future__ import annotations

import copy
import json
import random
from itertools import combinations
from pathlib import Path


def _stage_input(stage: dict) -> dict:
    """抽取阶段的轨迹精简视图，并保持通用字段顺序一致。"""
    result = {
        "stage_id": stage["stage_id"],
        "order": stage["order"],
        "stage_type": stage["stage_type"],
        "modality": stage["modality"],
    }
    if "timepoint" in stage:
        result["timepoint"] = stage["timepoint"]
    result["image_paths"] = stage["image_paths"]
    result["qa_pairs"] = [dict(turn) for turn in stage["qa_pairs"]]
    return result


def build_standard_trajectory(patient_stages: dict) -> dict:
    # 按阶段顺序生成 standard_full 轨迹
    trajectory = {
        "trajectory_id": f"{patient_stages['patient_id']}__standard_full",
        "patient_id": patient_stages["patient_id"],
    }
    for key in ("patient_name", "group"):
        if key in patient_stages:
            trajectory[key] = patient_stages[key]
    trajectory.update(
        {
            "trajectory_type": "standard_full",
            "stages": [_stage_input(stage) for stage in patient_stages["stages"]],
        }
    )
    return trajectory


def build_missing_modality_variants(standard: dict) -> list[dict]:
    """构造缺失模态变体轨迹 (单模态缺失 + 双模态缺失)
       先分别删除单个阶段(4 个单缺失变体), 再两两组合删除(6 个双缺失变体)
    """
    # FP 为必备模态
    removable = {
        "dp": "S2_DP",
        "xr": "S3_XR_XLA",
        "ct": "S4_CT",
        "tmj": "S5_TMJ",
    }

    drops: dict[str, set[str]] = {}
    # 单模态缺失
    for name, stage_id in removable.items():
        drops[f"no_{name}"] = {stage_id}
    # 双模态缺失(可删模态两两组合)
    for (a_name, a_stage), (b_name, b_stage) in combinations(removable.items(), 2):
        drops[f"no_{a_name}_{b_name}"] = {a_stage, b_stage}

    variants = []
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
    noise_count: int = 8,
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
                "role": "observation",
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
