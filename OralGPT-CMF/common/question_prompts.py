"""Question-scoped rules; previous rules are never part of the conversation history."""
from pathlib import Path

import yaml


def load_question_prompts(path: Path) -> dict:
    bundle = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("common", "history_heading"):
        if not isinstance(bundle[key], str) or not bundle[key].strip():
            raise ValueError(f"Empty {key} in {path}")
    for task, spec in bundle["tasks"].items():
        if not spec["match"] or not spec["blocks"] or not spec["instruction"].strip():
            raise ValueError(f"Incomplete prompt task: {task}")
        for pattern in spec["match"]:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(f"Invalid question pattern: {task}")
        for block in spec["blocks"]:
            if not bundle["blocks"][block].strip():
                raise ValueError(f"Empty prompt block: {block}")
    return bundle


def prompt_for_question(bundle: dict, question: str) -> tuple[str, str]:
    text = " ".join(question.replace("<image>", " ").casefold().split())
    matches = [task for task, spec in bundle["tasks"].items()
               if any(" ".join(pattern.casefold().split()) in text for pattern in spec["match"])]
    if len(matches) != 1:
        raise ValueError(f"Expected one question type, got {matches}: {text}")
    task = matches[0]
    spec = bundle["tasks"][task]
    parts = [bundle["common"], f"Current question type: {task}", spec["instruction"]]
    parts.extend(bundle["blocks"][name] for name in dict.fromkeys(spec["blocks"]))
    return task, "\n\n".join(parts)


def question_system(bundle: dict, question: str, context: str) -> str:
    _, rules = prompt_for_question(bundle, question)
    return rules + "\n\nPatient record and conversation history:\n" + context


def bind_prompt_state(state: dict, bundle: dict | None) -> None:
    if (state.get("turns") or state.get("status") == "completed") and state.get("question_prompts") != bundle:
        raise ValueError("Question prompts differ from the saved run; use a new output-model name.")
    if bundle is not None:
        state["question_prompts"] = bundle
