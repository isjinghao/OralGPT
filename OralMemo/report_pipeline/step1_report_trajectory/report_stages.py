from __future__ import annotations


def build_report_stages(
    normed_timepoints: list[dict],
    rendered_turns: list[dict],
    patient: dict,
) -> dict:
    """按时间点而非模态切分阶段
    阶段信息来自规整后的时间点, 问答来自轮次(按 stage_id 归集)
    诊断/治疗/预后轮 -> heldout_turns。
    """
    turns_by_stage: dict[str, list[dict]] = {}
    heldout: list[dict] = []
    for turn in rendered_turns:
        if turn["role"] == "evidence":
            turns_by_stage.setdefault(turn["stage_id"], []).append(turn)
        else:
            heldout.append({
                "source_turn_id": turn["source_turn_id"],
                "human": turn["human"],
                "assistant": turn["assistant"],
                "image_paths": turn["image_paths"],
                "role": turn["role"],
            })

    stages = []
    for tp in normed_timepoints:
        collected = turns_by_stage.get(tp["stage_id"], [])
        qa_pairs, image_paths, source_turn_ids = [], [], []
        for turn in collected:
            qa_pairs.append({
                "source_turn_id": turn["source_turn_id"],
                "human": turn["human"],
                "assistant": turn["assistant"],
                "image_paths": turn["image_paths"],
            })
            image_paths.extend(turn["image_paths"])
            source_turn_ids.append(turn["source_turn_id"])

        stages.append({
            "stage_id": tp["stage_id"],
            "order": tp["order"],
            "stage_type": tp.get("stage_type", "followup_visit"),
            "modality": tp.get("modality", ["TEXT_QA"]),
            "timepoint": {
                "label": tp.get("label"),
                "date_text": tp.get("date_text"),
                "t_months": tp.get("t_months"),
            },
            "source_turn_ids": source_turn_ids,
            "image_paths": image_paths,
            "qa_pairs": qa_pairs,
        })

    return {
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "group": patient["group"],
        "stages": stages,
        "heldout_turns": heldout,
    }
