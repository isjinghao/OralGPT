from __future__ import annotations

from collections import defaultdict




STAGE_DEFS = [
    {
        "stage_id": "S0_PROFILE",
        "order": 0,
        "stage_type": "perception",
        "modality": ["TEXT_QA"],
        "labels": ["profile"],
    },
    {
        "stage_id": "S1_FP",
        "order": 1,
        "stage_type": "perception",
        "modality": ["FP"],
        "labels": ["FP"],
    },
    {
        "stage_id": "S2_DP",
        "order": 2,
        "stage_type": "perception",
        "modality": ["DP"],
        "labels": ["DP"],
    },
    {
        "stage_id": "S3_XR_XLA",
        "order": 3,
        "stage_type": "perception",
        "modality": ["XR", "XLData"],
        "labels": ["ceph", "panoramic"],
    },
    {
        "stage_id": "S4_CT",
        "order": 4,
        "stage_type": "perception",
        "modality": ["CT"],
        "labels": ["CT"],
    },
    {
        "stage_id": "S5_TMJ",
        "order": 5,
        "stage_type": "perception",
        "modality": ["TMJ"],
        "labels": ["TMJ", "ECT"],
    },
]


def classify_turn(human: str) -> str:
    t = human.lower()

    if "single-photon emission" in t or "(ect)" in t:
        return "ECT"

    if "3d ct" in t or "craniofacial reconstruction" in t or "dentofacial deformity based on the provided" in t:
        return "CT"

    if "cephalometric radiograph" in t or ("cephalometric" in t and "radiograph" in t):
        return "ceph"

    if "panoramic" in t:
        return "panoramic"

    if "temporomandibular joint" in t or "(tmj)" in t:
        return "TMJ"

    if "facial photograph" in t:
        return "FP"

    if "intraoral" in t or "occlusal and dental" in t:
        return "DP"

    if (
        "basic information" in t
        or "primary concern" in t
        or "past medical" in t
        or "medical or surgical history" in t
        or "chief complaint" in t
    ):
        return "profile"

    return "treatment"


def build_patient_stages(source_turns: list[dict]) -> dict:

    label_to_turns: dict[str, list[dict]] = defaultdict(list)
    for src in source_turns:
        label = classify_turn(src["human"])
        label_to_turns[label].append(src)

    stages = []
    for stage_def in STAGE_DEFS:
        collected = []
        for label in stage_def["labels"]:
            collected.extend(label_to_turns.get(label, []))

        collected.sort(key=lambda s: s["source_turn_id"])

        turns = []
        image_paths = []
        for src in collected:
            turns.append(
                {
                    "source_turn_id": src["source_turn_id"],
                    "human": src["human"],
                    "assistant": src["assistant"],
                    "image_paths": src["image_paths"],
                    "role": "observation",
                }
            )
            image_paths.extend(src["image_paths"])

        stages.append(
            {
                "stage_id": stage_def["stage_id"],
                "order": stage_def["order"],
                "stage_type": stage_def["stage_type"],
                "modality": stage_def["modality"],
                "image_paths": image_paths,
                "qa_pairs": turns,
            }
        )

    evaluation_turns = [
        {
            "source_turn_id": src["source_turn_id"],
            "human": src["human"],
            "assistant": src["assistant"],
            "image_paths": src["image_paths"],
            "role": "evaluation",
            "ask_after_stage": "S5_TMJ",
            "release_after_stage": "S6_TREATMENT",
        }
        for src in label_to_turns.get("treatment", [])
    ]
    if evaluation_turns:
        stages.append(
            {
                "stage_id": "S6_TREATMENT",
                "order": len(stages),
                "stage_type": "treatment",
                "modality": ["TEXT_QA"],
                "image_paths": [],
                "qa_pairs": evaluation_turns,
            }
        )

    return {
        "patient_id": source_turns[0]["patient_id"],
        "patient_name": source_turns[0]["patient_name"],
        "group": source_turns[0]["group"],
        "stages": stages,
    }
