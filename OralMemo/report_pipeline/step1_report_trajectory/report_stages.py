from __future__ import annotations


def _qa_from_turn(
    turn: dict,
    *,
    ask_after_stage: str | None = None,
    release_after_stage: str | None = None,
) -> dict:
    qa = {
        "source_turn_id": turn["source_turn_id"],
        "human": turn["human"],
        "assistant": turn["assistant"],
        "image_paths": turn["image_paths"],
        "role": turn["role"],
    }
    if turn["role"] == "evaluation":
        qa.update(
            {
                "ask_after_stage": ask_after_stage,
                "release_after_stage": release_after_stage,
            }
        )
    return qa


def build_report_stages(
    normed_timepoints: list[dict],
    rendered_turns: list[dict],
    patient: dict,
) -> dict:
    """构造完整阶段视图；所有 QA 留在原时间点，role 控制可见性。"""
    turns_by_stage: dict[str, list[dict]] = {}
    for turn in rendered_turns:
        turns_by_stage.setdefault(turn["stage_id"], []).append(turn)

    perception_ids = [
        timepoint["stage_id"]
        for timepoint in normed_timepoints
        if timepoint["stage_type"] == "perception"
    ]
    treatment_ids = [
        timepoint["stage_id"]
        for timepoint in normed_timepoints
        if timepoint["stage_type"] == "treatment"
    ]
    if not perception_ids:
        raise ValueError("At least one perception stage is required")
    perception_cutoff = perception_ids[-1]
    treatment_release = treatment_ids[-1] if treatment_ids else perception_cutoff

    stages: list[dict] = []
    for timepoint in normed_timepoints:
        qa_pairs = []
        for turn in turns_by_stage[timepoint["stage_id"]]:
            if turn["role"] == "evaluation" and timepoint["stage_type"] == "treatment":
                qa_pairs.append(
                    _qa_from_turn(
                        turn,
                        ask_after_stage=perception_cutoff,
                        release_after_stage=treatment_release,
                    )
                )
            elif turn["role"] == "evaluation":
                qa_pairs.append(
                    _qa_from_turn(
                        turn,
                        ask_after_stage=timepoint["stage_id"],
                        release_after_stage=timepoint["stage_id"],
                    )
                )
            else:
                qa_pairs.append(_qa_from_turn(turn))

        observations = [qa for qa in qa_pairs if qa["role"] == "observation"]
        stages.append(
            {
                "stage_id": timepoint["stage_id"],
                "order": timepoint["order"],
                "stage_type": timepoint["stage_type"],
                "modality": timepoint["modality"],
                "timepoint": {
                    "date_text": timepoint["date_text"],
                    "t_months": timepoint["t_months"],
                },
                "image_paths": [path for qa in observations for path in qa["image_paths"]],
                "qa_pairs": qa_pairs,
            }
        )

    return {
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "group": patient["group"],
        "stages": stages,
    }
