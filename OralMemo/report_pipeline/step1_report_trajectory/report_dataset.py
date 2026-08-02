from __future__ import annotations


def build_report_dataset_entry(
    standard: dict,
    patient: dict,
    source_pdf: str,
) -> dict:
    conversations: list[dict] = []
    images: list[str] = []
    qa_count = 0
    for stage in standard["stages"]:
        for turn in stage["qa_pairs"]:
            conversations.append({"from": "human", "value": turn["human"]})
            conversations.append({"from": "gpt", "value": turn["assistant"]})
            images.extend("/" + rel.lstrip("/") for rel in turn["image_paths"])
            qa_count += 1

    return {
        "id": patient["patient_id"],
        "patient_uid": patient["patient_id"],
        "patient_name": patient["name"],
        "group": patient["group"],
        "source_pdf": source_pdf,
        "stages": [stage["stage_id"] for stage in standard["stages"]],
        "images": images,
        "num_images": len(images),
        "num_qa_pairs": qa_count,
        "conversations": conversations,
    }
