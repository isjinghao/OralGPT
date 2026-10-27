"""Conversation structure and history shared by training data and answer generation."""
import re
from typing import Any


def clean_question(question: str) -> str:
    return re.sub(r"\s*<image>\s*", " ", question).strip()


def rollout_system_context(
    patient_context: str,
    history: list[tuple[str, str]],
    history_heading: str = "Prior clinical reasoning record. Treat the following as established context.",
) -> str:
    if not history:
        return patient_context
    lines = [patient_context, "", history_heading]
    for index, (question, answer) in enumerate(history, start=1):
        lines.extend(("", f"[PRIOR TURN {index}]", f"Question: {clean_question(question)}", f"Recorded answer: {answer.strip()}"))
    return "\n".join(lines)


def system_context(case: dict[str, Any]) -> str:
    messages = case["conversations"]
    if not messages or messages[0].get("from") != "system" or not messages[0].get("value", "").strip():
        raise ValueError(f"{case['id']}: missing patient system context")
    return messages[0]["value"]


def conversation_pairs(case: dict[str, Any]) -> list[tuple[dict[str, str], dict[str, str]]]:
    messages = case["conversations"][1:]
    if len(messages) % 2:
        raise ValueError(f"{case['id']}: odd non-system conversation length")
    pairs = []
    for index in range(0, len(messages), 2):
        question, reference = messages[index : index + 2]
        if question["from"] != "human" or reference["from"] != "gpt":
            raise ValueError(f"{case['id']}: invalid roles at turns {index}/{index + 1}")
        pairs.append((question, reference))
    return pairs
