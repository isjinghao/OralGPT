from __future__ import annotations

STAGE_TYPES = ("perception", "treatment", "followup")
STAGE_RANK = {name: rank for rank, name in enumerate(STAGE_TYPES)}
QA_ROLES = {"observation", "evaluation"}


def normalize_timepoints(extracted: dict) -> list[dict]:
    """补齐 stage_id/order，并校验 perception→treatment→followup 的连续顺序。"""
    timepoints = extracted["timepoints"]
    if not isinstance(timepoints, list) or not timepoints:
        raise ValueError("Timeline must contain non-empty timepoints")
    normalized: list[dict] = []
    previous_rank = 0
    type_counts = {stage_type: 0 for stage_type in STAGE_TYPES}
    for order, raw_timepoint in enumerate(timepoints):
        timepoint = dict(raw_timepoint)
        stage_type = str(timepoint.get("stage_type") or "").strip()
        if stage_type not in STAGE_RANK:
            raise ValueError(f"Unsupported stage_type at timepoint {order}: {stage_type}")
        rank = STAGE_RANK[stage_type]
        if rank < previous_rank:
            raise ValueError(f"Clinical period order moves backward at timepoint {order}: {stage_type}")
        previous_rank = rank

        date_text = timepoint.get("date_text")
        if not isinstance(date_text, str) or not date_text.strip():
            raise ValueError(f"date_text must be non-empty at timepoint {order}")
        qa_pairs = timepoint.get("qa_pairs")
        if not isinstance(qa_pairs, list) or not qa_pairs:
            raise ValueError(f"timepoint {order} must contain non-empty qa_pairs")
        seen_evaluation = False
        roles: list[str] = []
        for qa_index, qa in enumerate(qa_pairs):
            role = str(qa.get("role") or "").strip()
            if role not in QA_ROLES:
                raise ValueError(f"Unsupported QA role at timepoint {order}, qa {qa_index}: {role}")
            roles.append(role)
            if stage_type == "perception" and role != "observation":
                raise ValueError("perception timepoints may contain observation QAs only")
            if role == "evaluation":
                seen_evaluation = True
            elif seen_evaluation:
                raise ValueError(
                    f"Observation appears after evaluation at timepoint {order}, qa {qa_index}"
                )
            if role == "evaluation" and qa.get("figure_ref"):
                raise ValueError("Evaluation QAs cannot attach unreleased answer images")
        if stage_type == "treatment" and any(role != "evaluation" for role in roles):
            raise ValueError("Treatment timepoints must contain evaluation QAs only")
        if stage_type == "followup" and "observation" not in roles:
            raise ValueError(f"Followup timepoint {order} must contain at least one observation QA")

        type_index = type_counts[stage_type]
        type_counts[stage_type] += 1
        t_months = timepoint.get("t_months")
        if t_months is not None and (
            isinstance(t_months, bool) or not isinstance(t_months, int) or t_months < 0
        ):
            raise ValueError(
                f"t_months must be a non-negative integer or null at timepoint {order}: {t_months!r}"
            )
        modality = timepoint.get("modality")
        if not isinstance(modality, list) or not modality:
            raise ValueError(f"modality must be a non-empty list at timepoint {order}")

        timepoint["order"] = order
        timepoint["stage_id"] = f"T{order}_{stage_type}_{type_index:02d}"
        timepoint["stage_type"] = stage_type
        normalized.append(timepoint)

    if normalized[0]["stage_type"] != "perception":
        raise ValueError("The trajectory must begin with at least one perception timepoint")
    return normalized


def resolve_image(figure_ref, images_map: dict) -> list[str]:
    if not figure_ref:
        return []
    entry = images_map.get(figure_ref)
    if isinstance(entry, dict) and entry.get("images"):
        return list(entry["images"])
    return []


def prefix_images(question: str, count: int) -> str:
    return ("<image>\n" * count) + question if count > 0 else question


def _required_text(value: object, location: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required text at {location}")
    return text


def _canonical(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _validate_image_qas(timepoint: dict, images_map: dict) -> None:
    seen_figures: set[str] = set()
    earlier_text_answers: list[str] = []
    for qa_index, qa in enumerate(timepoint["qa_pairs"]):
        figure_ref = str(qa.get("figure_ref") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not figure_ref:
            earlier_text_answers.append(answer)
            continue
        if figure_ref in seen_figures:
            raise ValueError(
                f"Duplicate image QA for {figure_ref} at {timepoint['stage_id']}, qa {qa_index}"
            )
        seen_figures.add(figure_ref)
        entry = images_map.get(figure_ref) or {}
        caption = entry.get("caption", "") if isinstance(entry, dict) else ""
        if caption and _canonical(answer) == _canonical(caption):
            raise ValueError(
                f"Caption-only image QA for {figure_ref} at {timepoint['stage_id']}, qa {qa_index}"
            )
        if any(figure_ref.casefold() in text.casefold() for text in earlier_text_answers):
            raise ValueError(
                f"Text answer leaks {figure_ref} before its image QA at {timepoint['stage_id']}"
            )


def render_turns(normed_timepoints: list[dict], images_map: dict) -> list[dict]:
    """统一渲染三个时期的 observation/evaluation QA，不生成任何固定问题。"""
    turns: list[dict] = []
    source_turn_id = 0
    for timepoint in normed_timepoints:
        _validate_image_qas(timepoint, images_map)
        for qa_index, qa in enumerate(timepoint["qa_pairs"]):
            role = qa["role"]
            question = _required_text(
                qa.get("question"), f"{timepoint['stage_id']}.qa_pairs[{qa_index}].question"
            )
            answer = _required_text(
                qa.get("answer"), f"{timepoint['stage_id']}.qa_pairs[{qa_index}].answer"
            )
            images = resolve_image(qa.get("figure_ref"), images_map)
            source_turn_id += 1
            turns.append(
                {
                    "source_turn_id": source_turn_id,
                    "stage_id": timepoint["stage_id"],
                    "role": role,
                    "human": prefix_images(question, len(images)),
                    "assistant": answer,
                    "image_paths": images,
                }
            )
    return turns
