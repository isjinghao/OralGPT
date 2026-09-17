from __future__ import annotations

from dataclasses import dataclass

from step2_evidence.graph import stage_order


@dataclass(frozen=True)
class EvidenceIndex:
    evidence: list[dict]
    graph: dict

    def __post_init__(self) -> None:
        ids = [item["evidence_id"] for item in self.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("Evidence catalog contains duplicate evidence_id values")

    def resolve(self, evidence_ids: list[str]) -> list[dict]:
        lookup = {item["evidence_id"]: item for item in self.evidence}
        unique_ids = list(dict.fromkeys(evidence_ids))
        unknown = [evidence_id for evidence_id in unique_ids if evidence_id not in lookup]
        if unknown:
            raise ValueError(f"Unknown evidence IDs: {unknown}")
        return [lookup[evidence_id] for evidence_id in unique_ids]

    def available_at(self, stage: str, stage_orders: dict[str, int] | None = None) -> list[dict]:

        order_of = stage_orders or {
            stage_id: stage_order(stage_id)
            for stage_id in {item["introduced_stage"] for item in self.evidence}
        }
        if stage not in order_of:
            raise ValueError(f"Unknown stage: {stage}")
        unknown = sorted({
            item["introduced_stage"] for item in self.evidence
            if item["introduced_stage"] not in order_of
        })
        if unknown:
            raise ValueError(f"Evidence contains unknown stages: {unknown}")
        limit = order_of[stage]
        return [item for item in self.evidence if order_of[item["introduced_stage"]] <= limit]

    def related_evidence(self, selected_ids: list[str], available: list[dict]) -> list[dict]:
        selected = set(selected_ids)
        available_ids = {item["evidence_id"] for item in available}
        related: set[str] = set()
        for edge in self.graph["edges"]:
            source, target = edge["source"], edge["target"]
            if source in selected and target in available_ids:
                related.add(target)
            if target in selected and source in available_ids:
                related.add(source)

        anchors = self.resolve(selected_ids)
        for item in available:
            normalized = item.get("normalized", {})
            for anchor in anchors:
                anchor_normalized = anchor.get("normalized", {})
                same_field = normalized.get("field") and normalized.get("field") == anchor_normalized.get("field")
                same_tooth = normalized.get("tooth") and normalized.get("tooth") == anchor_normalized.get("tooth")
                same_side = normalized.get("side") and normalized.get("side") == anchor_normalized.get("side")
                same_dimension = item.get("clinical_dimension") == anchor.get("clinical_dimension")
                if same_field or same_tooth or (same_side and same_dimension):
                    related.add(item["evidence_id"])
                    break

        return [item for item in available if item["evidence_id"] in related - selected]

def evidence_ref(item: dict) -> dict:

    normalized = item.get("normalized", {})
    return {
        "evidence_id": item["evidence_id"],
        "stage": item["introduced_stage"],
        "modality": item.get("modality", []),
        "fact_text": item["fact_text"],
        "field": normalized.get("field"),
        "value": normalized.get("value"),
        "unit": normalized.get("unit"),
        "tooth": normalized.get("tooth"),
        "side": normalized.get("side"),
    }


STAGE_LABELS = {
    "S0_PROFILE": "patient profile and history",
    "S1_FP": "facial photographs",
    "S2_DP": "intraoral dental photographs",
    "S3_XR_XLA": "radiographic assessment",
    "S4_CT": "three-dimensional CT",
    "S5_TMJ": "temporomandibular joint clinical examination",
}


def human_stage_label(stage_id: str) -> str:
    if stage_id in STAGE_LABELS:
        return STAGE_LABELS[stage_id]
    parts = stage_id.split("_")
    if stage_id.startswith("T") and len(parts) >= 2:
        return f"the relevant {parts[1]} timepoint"
    return "the relevant clinical findings"


def compact_evidence_text(evidence: list[dict]) -> str:

    rows = []
    for item in evidence:
        rows.append(
            f"- {item['evidence_id']} | stage={item['introduced_stage']} | "
            f"modality={','.join(item.get('modality', []))} | fact={item['fact_text']}"
        )
    return "\n".join(rows)


def question_evidence_text(evidence: list[dict]) -> str:

    rows = []
    for item in evidence:
        rows.append(
            f"- {item['evidence_id']} | clinical source={human_stage_label(item['introduced_stage'])} | "
            f"modality={','.join(item.get('modality', []))} | fact={item['fact_text']}"
        )
    return "\n".join(rows)


def evidence_catalog(index: EvidenceIndex) -> str:

    ordered = sorted(index.evidence, key=lambda item: stage_order(item["introduced_stage"]))
    return compact_evidence_text(ordered)


def edges_text(index: EvidenceIndex) -> str:

    rows = []
    for edge in index.graph["edges"]:
        if edge["type"] == "context_consistency":
            continue
        rows.append(f"- {edge['source']} -> {edge['target']} | {edge.get('type', '')} | {edge.get('reason', '')}")
    return "\n".join(rows)


def stages_summary(patient_stages: dict) -> str:

    rows = []
    for stage in sorted(patient_stages["stages"], key=lambda s: s["order"]):
        rows.append(
            f"- {stage['stage_id']} (order {stage['order']}) | source={human_stage_label(stage['stage_id'])} | "
            f"modality={','.join(stage['modality'])}"
        )
    return "\n".join(rows)


def assemble_normal_task(patient_id: str, suffix: str, planned: dict, index: EvidenceIndex) -> dict:

    evidence = index.resolve(planned["required_evidence_ids"])
    return {
        "task_id": f"{patient_id.replace('__', '_')}_{suffix}",
        "patient_id": patient_id,
        "task_type": planned["task_type"],
        "ask_after_stage": planned["ask_after_stage"],
        "selected_evidence": [evidence_ref(item) for item in evidence],
        "gold_answer": planned["gold_answer"]["natural_answer"],
    }


def assemble_evaluation_task(
    *,
    patient_id: str,
    task_id: str,
    task_type: str,
    turn: dict,
    evidence_ids: list[str],
    index: EvidenceIndex,
) -> dict:
    evidence = index.resolve(evidence_ids)
    return {
        "task_id": task_id,
        "patient_id": patient_id,
        "task_type": task_type,
        "ask_after_stage": turn["ask_after_stage"],
        "release_after_stage": turn["release_after_stage"],
        "selected_evidence": [evidence_ref(item) for item in evidence],
        "question": turn["human"],
        "gold_answer": turn["assistant"],
    }
