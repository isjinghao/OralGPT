from __future__ import annotations

import re


def _safe_label(label: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "-", str(label)).strip("-")
    return s or "tp"


def _coerce_int(v, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def normalize_timepoints(extracted: dict) -> list[dict]:
    """给每个时间点补上确定性的 stage_id 与 order, 并规整字段。"""
    tps = extracted.get("timepoints", []) or []
    normed = []
    for order, tp in enumerate(tps):
        tp = dict(tp)
        tp["order"] = order
        tp["stage_id"] = f"T{order}_{_safe_label(tp.get('label', order))}"
        tp["t_months"] = _coerce_int(tp.get("t_months"), default=order)
        tp.setdefault("stage_type", "followup_visit")
        tp.setdefault("modality", ["TEXT_QA"])
        tp.setdefault("findings", [])
        normed.append(tp)
    return normed


def resolve_image(figure_ref, images_map: dict) -> list[str]:
    if not figure_ref:
        return []
    entry = images_map.get(figure_ref)
    if isinstance(entry, dict) and entry.get("image"):
        return [entry["image"]]
    return []


def prefix_images(human: str, n: int) -> str:
    return ("<image>\n" * n) + human if n > 0 else human


def render_turns(normed_timepoints: list[dict], held_out: dict, images_map: dict) -> list[dict]:
    """把规整后的时间点(findings/decision)与 held_out 通用地渲染为有序问答轮次
    - 每条 finding -> 一轮问答(问题模板化, 答案=statement), figure_ref 命中则附图。
    - decision -> 一轮问答。
    - held_out(diagnosis/treatment/prognosis) -> 各一轮, role 为其自身。
    """
    turns: list[dict] = []
    tid = 0

    for tp in normed_timepoints:
        date_text = tp.get("date_text") or tp.get("label") or ""
        for finding in tp.get("findings", []):
            topic = (finding.get("topic") or "the clinical findings").strip()
            statement = (finding.get("statement") or "").strip()
            if not statement:
                continue
            images = resolve_image(finding.get("figure_ref"), images_map)
            tid += 1
            turns.append({
                "source_turn_id": tid,
                "stage_id": tp["stage_id"],
                "role": "evidence",
                "human": prefix_images(f"At the {date_text} timepoint, what are the findings regarding {topic}?", len(images)),
                "assistant": statement,
                "image_paths": images,
            })

        decision = (tp.get("decision") or "").strip()
        if decision:
            rationale = (tp.get("rationale") or "").strip()
            answer = decision if not rationale else f"{decision} The stated rationale was: {rationale}"
            tid += 1
            turns.append({
                "source_turn_id": tid,
                "stage_id": tp["stage_id"],
                "role": "evidence",
                "human": f"At the {date_text} timepoint, what was the key clinical decision and its rationale?",
                "assistant": answer,
                "image_paths": [],
            })

    held_out = held_out or {}
    ho_specs = [
        ("heldout_diagnosis", "diagnosis",
         "Based on the longitudinal clinical, radiographic, and other findings, what is the unifying diagnosis?"),
        ("heldout_treatment", "treatment",
         "Summarize the overall long-term treatment strategy delivered across the follow-up period."),
        ("heldout_prognosis", "prognosis",
         "What were the long-term outcomes over the whole follow-up period?"),
    ]
    for role, key, question in ho_specs:
        answer = (held_out.get(key) or "").strip()
        if not answer:
            continue
        tid += 1
        turns.append({
            "source_turn_id": tid,
            "stage_id": None,
            "role": role,
            "human": question,
            "assistant": answer,
            "image_paths": [],
        })

    return turns
