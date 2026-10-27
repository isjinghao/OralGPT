"""Answer files and resumable local generation state."""
import json
from pathlib import Path
from typing import Any

from common.conversation import clean_question, conversation_pairs


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def patient_paths(output_root: Path, model: str, case_id: str) -> tuple[Path, Path]:
    group, name = case_id.split("__", 1)
    patient_dir = output_root / group / name
    return patient_dir / "cache" / model / "rollout_state.json", patient_dir / "evaluation" / model / "answers.json"


def write_answers(path: Path, state: dict[str, Any]) -> None:
    write_json_atomic(
        path,
        [
            {
                "task_id": f"{state['case_id']}__turn_{turn['turn_index']:02d}",
                "turn_index": turn["turn_index"],
                "question": clean_question(turn["question"]),
                "images": turn["images"],
                "gold_answer": turn["reference"],
                "model_answer": turn["prediction"],
                "finish_reason": turn["finish_reason"],
            }
            for turn in state["turns"]
        ],
    )


def load_state(path: Path, case: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    if path.is_file() and not overwrite:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("case_id") != case["id"]:
            raise ValueError(f"State does not match requested case: {path}")
        return state
    return {
        "case_id": case["id"],
        "patient_uid": case.get("patient_uid"),
        "patient_name": case.get("patient_name"),
        "context_turn_labels": case.get("context_turn_labels", []),
        "status": "in_progress",
        "completed_turns": 0,
        "turns": [],
    }


def rebuild_history(case: dict[str, Any], state: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = conversation_pairs(case)
    completed = state["turns"]
    if len(completed) > len(pairs):
        raise ValueError(f"{case['id']}: state has too many completed turns")

    history = []
    image_cursor = 0
    for turn_index, (question, _) in enumerate(pairs):
        image_count = question["value"].count("<image>")
        turn_images = case.get("images", [])[image_cursor : image_cursor + image_count]
        if len(turn_images) != image_count:
            raise ValueError(f"{case['id']}: image references do not match turn {turn_index}")
        image_cursor += image_count
        if turn_index >= len(completed):
            break
        turn = completed[turn_index]
        if turn["question"] != question["value"] or turn["images"] != turn_images:
            raise ValueError(f"{case['id']}: state does not match source at turn {turn_index}")
        history.append((question["value"], turn["prediction"]))
    return history
