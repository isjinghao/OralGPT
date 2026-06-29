from __future__ import annotations

from dataclasses import dataclass

from bench.step2_evidence.graph import stage_order


@dataclass(frozen=True)
class EvidenceIndex:
    evidence: list[dict]
    graph: dict

    def resolve(self, evidence_ids: list[str]) -> list[dict]:
        # 把 evidence_id 列表解析为证据对象列表
        lookup = {item["evidence_id"]: item for item in self.evidence}  # 返回 evidence_id 到证据的映射
        seen: set[str] = set()
        resolved = []
        for evidence_id in evidence_ids:
            if evidence_id in lookup and evidence_id not in seen:
                seen.add(evidence_id)
                resolved.append(lookup[evidence_id])
        return resolved

    def edges_between(self, evidence_ids: list[str]) -> list[dict]:
        # 取证据子集间的图边
        selected = set(evidence_ids)
        return [
            edge for edge in self.graph["edges"]
            if edge["source"] in selected and edge["target"] in selected
        ]


def evidence_ref(item: dict) -> dict:
    # 生成证据引用视图
    normalized = item.get("normalized", {})
    return {
        "evidence_id": item["evidence_id"],
        "stage": item["introduced_stage"],
        "modality": item.get("modality", []),
        "fact_text": item["fact_text"],
        "field": normalized.get("field"),
        "value": normalized.get("value"),
        "unit": normalized.get("unit"),
    }


def compact_evidence_text(evidence: list[dict]) -> str:
    # 每条证据生成一行 (id|stage|modality|fact), 供 prompt 使用
    rows = []
    for item in evidence:
        rows.append(
            f"- {item['evidence_id']} | stage={item['introduced_stage']} | "
            f"modality={','.join(item.get('modality', []))} | fact={item['fact_text']}"
        )
    return "\n".join(rows)


def evidence_catalog(index: EvidenceIndex) -> str:
    # 生成全部证据压缩文本
    ordered = sorted(index.evidence, key=lambda item: stage_order(item["introduced_stage"]))
    return compact_evidence_text(ordered)


def edges_text(index: EvidenceIndex) -> str:
    # 把证据图的边压成 (source -> target | type | reason) 多行文本
    rows = []
    for edge in index.graph["edges"]:
        rows.append(f"- {edge['source']} -> {edge['target']} | {edge.get('type', '')} | {edge.get('reason', '')}")
    return "\n".join(rows)


def stages_summary(patient_stages: dict) -> str:
    # 按 order 列出各阶段id/类型/模态, 供LLM判断ask_after_stage与回忆/更新关系
    rows = []
    for stage in sorted(patient_stages["stages"], key=lambda s: s["order"]):
        rows.append(
            f"- {stage['stage_id']} (order {stage['order']}) | type={stage['stage_type']} | "
            f"modality={','.join(stage['modality'])}"
        )
    return "\n".join(rows)


def assemble_normal_task(patient_id: str, suffix: str, planned: dict, index: EvidenceIndex) -> dict:
    # 把 LLM 规划结果与解析后的证据组装为一条普通任务，待 finalize_task 生成问题并校验
    evidence = index.resolve(planned["required_evidence_ids"])
    evidence_ids = [item["evidence_id"] for item in evidence]
    gold = planned["gold_answer"]
    return {
        "task_id": f"{patient_id.replace('__', '_')}_{suffix}",
        "patient_id": patient_id,
        "task_type": planned["task_type"],
        "ask_after_stage": planned["ask_after_stage"],
        "selected_evidence": [evidence_ref(item) for item in evidence],
        "evidence_graph_edges": index.edges_between(evidence_ids),
        "gold_answer": gold["natural_answer"],
    }


def assemble_heldout_task(
    *,
    patient_id: str,
    task_id: str,
    task_type: str,
    ask_after_stage: str,
    turn: dict,
    evidence_ids: list[str],
    index: EvidenceIndex,
) -> dict:
    # 组装 held-out QA 任务
    evidence = index.resolve(evidence_ids)
    resolved_ids = [item["evidence_id"] for item in evidence]
    return {
        "task_id": task_id,
        "patient_id": patient_id,
        "task_type": task_type,
        "ask_after_stage": ask_after_stage,
        "selected_evidence": [evidence_ref(item) for item in evidence],
        "evidence_graph_edges": index.edges_between(resolved_ids),
        "heldout_source_turn_ids": [turn["source_turn_id"]],
        "question": turn["human"],
        "gold_answer": turn["assistant"],
    }
