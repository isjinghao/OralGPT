from __future__ import annotations

from step1_patient_trajectory.trajectories import build_standard_trajectory


def build_report_standard_trajectory(patient_stages: dict) -> dict:
    """复用 step1 的标准轨迹构造, 并把时间点元信息(timepoint)回填到各阶段
    step1 的 build_standard_trajectory 只保留通用阶段字段, 这里补回 timepoint, 使长程轨迹在下游仍可访问时间信息
    """
    standard = build_standard_trajectory(patient_stages)
    tp_by_stage = {s["stage_id"]: s.get("timepoint") for s in patient_stages["stages"]}
    for stage in standard["stages"]:
        stage["timepoint"] = tp_by_stage.get(stage["stage_id"])
    return standard
