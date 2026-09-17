from __future__ import annotations

import json
from pathlib import Path


def load_dataset(dataset_json: Path) -> list[dict]:

    return json.loads(dataset_json.read_text(encoding="utf-8"))


def build_source_turns(item: dict) -> list[dict]:

    conversations = item["conversations"]
    all_images = item["images"]
    image_cursor = 0
    turns = []

    for idx in range(0, len(conversations), 2):
        source_turn_id = idx // 2 + 1
        human = conversations[idx]["value"]
        assistant = conversations[idx + 1]["value"]
        image_count = human.count("<image>")
        image_paths = [p.lstrip("/").replace("\\", "/") for p in all_images[image_cursor:image_cursor + image_count]]
        image_cursor += image_count

        turns.append(
            {
                "patient_id": item["id"],
                "patient_name": item["patient_name"],
                "group": item["group"],
                "source_turn_id": source_turn_id,
                "human": human,
                "assistant": assistant,
                "image_paths": image_paths,
            }
        )

    return turns
