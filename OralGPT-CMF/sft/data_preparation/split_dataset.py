#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = ROOT.parent / "OralFlow"
OUTPUT = ROOT / "sft/data/raw_splits"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SOURCES = {
    "supplemental": (ROOT / "VQA140_final.json", IMAGE_ROOT / "validation"),
    "main": (ROOT / "VQA387_final.json", IMAGE_ROOT / "SH9HCMFdata"),
}

QUESTION_MODALITIES = (
    ("FP", ("facial photograph",)),
    ("DP", ("occlusal and dental findings", "intraoral examination")),
    ("TMJ", ("temporomandibular joint",)),
    ("CEPH", ("cephalometric radiograph",)),
    ("CT", ("3d ct craniofacial reconstruction",)),
    ("PANO", ("panoramic x-ray", "panoramic radiograph")),
)
MODALITY_ORDER = {name: index for index, (name, _) in enumerate(QUESTION_MODALITIES)}


def drop_panoramic_without_image(item: dict, available: dict[str, list[Path]]) -> None:
    if available.get("PANO"):
        return
    messages = item["conversations"]
    kept = []
    for index in range(0, len(messages), 2):
        if question_modality(messages[index]["value"]) != "PANO":
            kept += messages[index : index + 2]
    if len(kept) != len(messages):
        print(f"Dropped panoramic question without a panoramic image: {item['id']}")
    item["conversations"] = kept


def strip_markdown(text: str) -> str:
    text = text.replace("**", "")
    text = re.sub(r"(?<!\S)#{1,6} +", "", text)
    return re.sub(r"(?<!\S)\* ", "- ", text)


def question_modality(text: str) -> str | None:
    text = re.sub(r"(?:<image>\s*)+", "", text).lower()
    for modality, patterns in QUESTION_MODALITIES:
        if any(pattern in text for pattern in patterns):
            return modality
    return None


def file_modality(path: Path) -> str | None:
    value = path.as_posix().lower()
    name = path.name.lower()
    if value.startswith("fp_mosaic/") or value.startswith("fp/"):
        return "FP"
    if value.startswith("dp/"):
        return "DP"
    if value.startswith("tmj/"):
        return "TMJ"
    if value.startswith("ct/"):
        return "CT"
    if name.startswith("xladata") or value.startswith("xr/xf") or value.startswith("xr/xla"):
        return "CEPH"
    if value.startswith("xr/xp"):
        return "PANO"
    return None


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.as_posix().lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def patient_folder(item: dict, image_root: Path) -> Path:
    folder = image_root / item["group"] / item["patient_name"]
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing patient image directory for {item['id']}: {folder}")
    return folder


def collect_images(folder: Path) -> tuple[dict[str, list[Path]], list[str]]:
    images: dict[str, list[Path]] = defaultdict(list)
    ignored: list[str] = []
    has_mosaic = (folder / "FP_mosaic").is_dir()
    for path in sorted(folder.rglob("*"), key=natural_key):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(folder)
        if has_mosaic and relative.as_posix().lower().startswith("fp/"):
            ignored.append(relative.as_posix())
            continue
        modality = file_modality(relative)
        if modality is None:
            ignored.append(relative.as_posix())
            continue
        images[modality].append(path)
    return images, ignored


def repair_item(raw_item: dict, source: str, image_root: Path, media_root: Path, output: Path) -> tuple[dict, dict]:
    item = deepcopy(raw_item)
    folder = patient_folder(item, image_root)
    available, ignored = collect_images(folder)
    drop_panoramic_without_image(item, available)
    for message in item["conversations"]:
        if message["from"] == "gpt":
            message["value"] = strip_markdown(message["value"])
    prompt_index: dict[str, int] = {}
    for index, message in enumerate(item.get("conversations", [])):
        if message.get("from") != "human":
            continue
        modality = question_modality(message.get("value", ""))
        if modality:
            if modality in prompt_index:
                raise ValueError(f"Duplicate dedicated {modality} prompt in {item['id']}")
            prompt_index[modality] = index

    selected: dict[str, list[Path]] = {}
    skipped_without_prompt: dict[str, list[str]] = {}
    for modality, paths in available.items():
        if modality in prompt_index:
            selected[modality] = paths
        else:
            skipped_without_prompt[modality] = [path.relative_to(folder).as_posix() for path in paths]

    new_images: list[str] = []
    assignment: dict[int, list[Path]] = {}
    for modality in sorted(selected, key=MODALITY_ORDER.get):
        assignment[prompt_index[modality]] = selected[modality]

    patient_media = media_root / item["group"] / item["patient_name"]
    patient_media.mkdir(parents=True, exist_ok=True)
    for index, message in enumerate(item.get("conversations", [])):
        if message.get("from") != "human":
            continue
        clean_question = re.sub(r"^(?:\s*<image>\s*)+", "", message.get("value", "")).lstrip()
        paths = assignment.get(index, [])
        if paths:
            message["value"] = "<image>\n" * len(paths) + clean_question
            for source_path in paths:
                relative = source_path.relative_to(folder)
                link_path = patient_media / relative
                link_path.parent.mkdir(parents=True, exist_ok=True)
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(source_path)
                new_images.append(link_path.relative_to(output).as_posix())
        else:
            message["value"] = clean_question

    placeholder_count = sum(
        message.get("value", "").count("<image>")
        for message in item.get("conversations", [])
        if message.get("from") == "human"
    )
    if placeholder_count != len(new_images):
        raise ValueError(f"Image mismatch after repair for {item['id']}: {len(new_images)} vs {placeholder_count}")
    if not all((output / path).is_file() for path in new_images):
        raise FileNotFoundError(f"Broken media symlink generated for {item['id']}")

    item["images"] = new_images
    item["num_images"] = len(new_images)
    item["num_qa_pairs"] = len(item.get("conversations", [])) // 2
    item["sft_source"] = source
    audit = {
        "id": item["id"],
        "source": source,
        "num_images": len(new_images),
        "skipped_without_prompt": skipped_without_prompt,
        "ignored_or_duplicate_files": ignored,
        "modalities": {modality: len(paths) for modality, paths in selected.items()},
    }
    return item, audit


def validate_dataset(name: str, data: list[dict], output: Path) -> dict:
    ids = [item["id"] for item in data]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {name}")
    images = 0
    placeholders = 0
    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) % 2:
            raise ValueError(f"Odd conversation count in {name}/{item['id']}")
        for index, message in enumerate(conversations):
            expected = "human" if index % 2 == 0 else "gpt"
            if message.get("from") != expected:
                raise ValueError(f"Role order error in {name}/{item['id']} at {index}")
        item_placeholders = sum(
            message.get("value", "").count("<image>")
            for message in conversations
            if message.get("from") == "human"
        )
        if item_placeholders != len(item.get("images", [])):
            raise ValueError(f"Image count error in {name}/{item['id']}")
        for path in item.get("images", []):
            if Path(path).is_absolute():
                raise ValueError(f"Absolute path in {name}/{item['id']}: {path}")
            if not (output / path).is_file():
                raise FileNotFoundError(f"Missing media in {name}/{item['id']}: {path}")
        images += len(item.get("images", []))
        placeholders += item_placeholders
    return {"items": len(data), "images": images, "placeholders": placeholders}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def build_dataset(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output / "media", ignore_errors=True)
    train_media = output / "media" / "train"
    eval_media = output / "media" / "evaluation"
    raw_supplemental = json.loads(SOURCES["supplemental"][0].read_text(encoding="utf-8"))
    raw_main = json.loads(SOURCES["main"][0].read_text(encoding="utf-8"))
    if len(raw_supplemental) != 140 or len(raw_main) != 387:
        raise ValueError("Expected source sizes 140 and 387")

    audit: list[dict] = []
    main_data = []
    for item in raw_main:
        repaired, row = repair_item(item, "main_387", SOURCES["main"][1], train_media, output)
        main_data.append(repaired)
        audit.append(row)

    supplemental_data = []
    for index, item in enumerate(raw_supplemental):
        media_root = train_media if index < 13 else eval_media
        repaired, row = repair_item(item, "supplemental_140", SOURCES["supplemental"][1], media_root, output)
        supplemental_data.append(repaired)
        audit.append(row)

    train = main_data + supplemental_data[:13]
    validation = supplemental_data[13:63]
    test = supplemental_data[63:113]
    human_evaluation = supplemental_data[113:140]
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
        "human_evaluation": human_evaluation,
    }
    expected = {"train": 400, "validation": 50, "test": 50, "human_evaluation": 27}
    stats = {name: validate_dataset(name, data, output) for name, data in splits.items()}
    if {name: value["items"] for name, value in stats.items()} != expected:
        raise ValueError(f"Unexpected split sizes: {stats}")

    for name, data in splits.items():
        write_json(output / f"oralcmf_sft_{name}.json", data)
    dataset_info = {
        f"oralcmf_sft_{name}": {
            "file_name": f"oralcmf_sft_{name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
        for name in splits
    }
    write_json(output / "dataset_info.json", dataset_info)

    skipped = [row for row in audit if row["skipped_without_prompt"]]
    manifest = {
        "source_files": {name: str(path) for name, (path, _) in SOURCES.items()},
        "split_policy": {
            "train": "All 387 main patients followed by supplemental patients 0:13",
            "validation": "Supplemental patients 13:63 in source order",
            "test": "Supplemental patients 63:113 in source order",
            "human_evaluation": "Supplemental patients 113:140 in source order",
        },
        "stats": stats,
        "ids": {name: [item["id"] for item in data] for name, data in splits.items()},
        "skipped_without_prompt_summary": {
            "patients": len(skipped),
            "modality_sets": sum(len(row["skipped_without_prompt"]) for row in skipped),
            "images": sum(
                len(paths)
                for row in skipped
                for paths in row["skipped_without_prompt"].values()
            ),
        },
        "skipped_without_prompt": skipped,
        "audit": audit,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))
    summary = manifest["skipped_without_prompt_summary"]
    print(
        "Skipped images without dedicated prompts: "
        f"{summary['patients']} patients, {summary['modality_sets']} modality sets, {summary['images']} images"
    )


if __name__ == "__main__":
    build_dataset(OUTPUT)
