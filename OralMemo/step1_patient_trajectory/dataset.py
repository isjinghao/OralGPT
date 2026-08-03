from __future__ import annotations

import json
from pathlib import Path


def load_dataset(dataset_json: Path) -> list[dict]:
    # 读取并返回整个数据集 JSON(患者条目列表)
    return json.loads(dataset_json.read_text(encoding="utf-8"))


def index_by_patient_id(data: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for item in data:
        index[item["id"]] = item
    return index


def load_patient_item(dataset_json: Path, patient_id: str) -> dict | None:
    # 读取数据集 JSON, 按 id 查找目标患者条目
    return index_by_patient_id(load_dataset(dataset_json)).get(patient_id)


def local_image_path(image_path: str) -> str:
    return image_path.lstrip("/").replace("\\", "/")


def build_source_turns(item: dict) -> list[dict]:
    # 将 conversations 两两配对成问答轮次, 按问题中 <image> 数量从 images 顺序切分图片
    conversations = item["conversations"]
    all_images = item["images"]
    image_cursor = 0
    turns = []

    for idx in range(0, len(conversations), 2):
        source_turn_id = idx // 2 + 1
        human = conversations[idx]["value"]
        assistant = conversations[idx + 1]["value"]
        image_count = human.count("<image>")
        image_paths = [local_image_path(p) for p in all_images[image_cursor:image_cursor + image_count]]
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
