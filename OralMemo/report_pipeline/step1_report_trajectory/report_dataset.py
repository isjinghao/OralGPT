from __future__ import annotations


def build_report_dataset_entry(
    normed_timepoints: list[dict],
    rendered_turns: list[dict],
    patient: dict,
    source_pdf: str | None = None,
) -> dict:
    conversations: list[dict] = []
    images: list[str] = []
    for turn in rendered_turns:
        conversations.append({"from": "human", "value": turn["human"]})
        conversations.append({"from": "gpt", "value": turn["assistant"]})
        for rel in turn["image_paths"]:
            images.append("/" + rel.lstrip("/"))

    return {
        "id": patient["patient_id"],
        "patient_uid": patient["patient_id"],
        "patient_name": patient["name"],
        "group": patient["group"],
        "source_pdf": source_pdf,
        "stages": [tp["stage_id"] for tp in normed_timepoints],
        "images": images,
        "num_images": len(images),
        "num_qa_pairs": len(rendered_turns),
        "conversations": conversations,
    }
