from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from string import Template

import yaml

from bench.llm_client import ChatClient


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "graph_edges.yaml"

# 跨阶段语义边的关系
ALLOWED_RELATIONS = {
    "supports",
    "refines",
    "confirms",
    "quantifies",
    "explains",
    "updates",
}

def stage_order(stage: str) -> int:
    if stage[:1] == "S" and stage[1:2].isdigit():
        return int(stage[1:2])


def build_evidence_graph(
    evidence_json: Path,
    client: ChatClient | None = None,
    cache_dir: Path | None = None,
    max_edges: int = 25,
) -> dict:
    """构建证据图 (结构规则 + LLM 语义边 + 规则校验)
    以 evidence.json 的证据条目为节点
    (1) 用确定性规则连接阶段内、同维度跨阶段递进边
    (2) 用LLM生成跨维度/跨模态的跨阶段依赖边, 经规则校验后并入
    (3) 去重编号返回完整图
    """
    evidence_data = json.loads(evidence_json.read_text(encoding="utf-8"))
    patient_id = evidence_data["patient_id"]
    nodes = evidence_data["evidence"]
    node_by_id = {node["evidence_id"]: node for node in nodes}

    edges = []
    edges.extend(build_intra_stage_edges(nodes))
    edges.extend(build_same_dimension_progression_edges(nodes))
    edges.extend(propose_cross_stage_edges(nodes, patient_id, client, cache_dir, max_edges))
    edges = dedupe_edges(edges, node_by_id)
    return {
        "patient_id": patient_id,
        "nodes": nodes,
        "edges": edges,
    }


def build_intra_stage_edges(nodes: list[dict]) -> list[dict]:
    # 构建阶段内边(确定性规则), 把同一阶段、同一维度的证据节点顺次相连 (intra_stage_link)
    edges = []
    groups = defaultdict(list)
    for node in nodes:
        if node.get("clinical_dimension", "other") == "other":
            continue
        groups[(node["introduced_stage"], node["clinical_dimension"])].append(node)

    for (stage, dimension), items in groups.items():
        ordered = sorted(items, key=lambda x: (x["source_turn_id"], x["evidence_id"]))
        for a, b in zip(ordered, ordered[1:]):
            edges.append(
                {
                    "source": a["evidence_id"],
                    "target": b["evidence_id"],
                    "type": "intra_stage_link",
                    "relation": "refines",
                    "reason": f"Same stage {stage} and clinical dimension {dimension}.",
                }
            )
    return edges


def build_same_dimension_progression_edges(nodes: list[dict]) -> list[dict]:
    # 构建同维度跨阶段递进边, 同一临床维度按阶段顺序排列, 每个节点连向下一个阶段的首个同维度节点
    # 表示同一临床属性在后续阶段被复测/细化/更新 (cross_stage_dependency)

    edges = []
    by_dimension = defaultdict(list)
    for node in nodes:
        if node.get("clinical_dimension", "other") == "other":
            continue
        by_dimension[node["clinical_dimension"]].append(node)

    for dimension, items in by_dimension.items():
        ordered = sorted(items, key=lambda x: (stage_order(x["introduced_stage"]), x["source_turn_id"], x["evidence_id"]))
        for early in ordered:
            later_candidates = [node for node in ordered if stage_order(node["introduced_stage"]) > stage_order(early["introduced_stage"])]
            if not later_candidates:
                continue
            later = later_candidates[0]
            edges.append(
                {
                    "source": early["evidence_id"],
                    "target": later["evidence_id"],
                    "type": "cross_stage_dependency",
                    "relation": "refines",
                    "reason": f"Later {dimension} evidence updates or supports earlier {dimension} evidence.",
                }
            )
    return edges


def build_edge_prompt(nodes: list[dict], patient_id: str, max_edges: int) -> str:
    # 组装跨阶段边提议 prompt, 把阶段顺序与全部节点压成紧凑文本, 填入prompt模板
    template = Template(yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))["template"])
    stages = sorted({node["introduced_stage"] for node in nodes}, key=stage_order)
    stage_lines = "\n".join(
        f"  - {stage} (stage_order {stage_order(stage)})"
        for stage in stages
    )
    node_lines = "\n".join(
        f"  - {node['evidence_id']} | stage={node['introduced_stage']} (order {stage_order(node['introduced_stage'])}) | "
        f"dimension={node.get('clinical_dimension', 'other')} | field={node.get('normalized', {}).get('field')} | "
        f"fact={node['fact_text']}"
        for node in sorted(nodes, key=lambda n: (stage_order(n["introduced_stage"]), n["source_turn_id"], n["evidence_id"]))
    )
    return template.substitute(patient_id=patient_id, stage_order=stage_lines, nodes=node_lines, max_edges=max_edges)


def request_llm_edges(
    nodes: list[dict],
    patient_id: str,
    client: ChatClient | None,
    cache_dir: Path | None,
    max_edges: int,
) -> list[dict]:
    # 获取 LLM 提议的原始边(带缓存)
    cache_path = cache_dir / "graph_edges.json" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached.get("edges", [])

    prompt = build_edge_prompt(nodes, patient_id, max_edges)
    result = client.complete_json(prompt, max_tokens=16000)
    edges = result.get("edges", [])
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"edges": edges}, ensure_ascii=False, indent=2), encoding="utf-8")
    return edges


def validate_proposed_edges(proposed: list[dict], nodes: list[dict]) -> list[dict]:
    # 校验 LLM 提议的跨阶段边 (规则gate)
    # 逐条校验端点存在、阶段单调(source 早于 target)、关系在白名单
    # 通过者归一化为标准边结构(type=cross_stage_dependency)

    node_by_id = {node["evidence_id"]: node for node in nodes}
    valid = []
    for edge in proposed:
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_by_id or target not in node_by_id:
            continue
        if stage_order(node_by_id[source]["introduced_stage"]) >= stage_order(node_by_id[target]["introduced_stage"]):
            continue
        relation = str(edge.get("relation", "")).lower()
        if relation not in ALLOWED_RELATIONS:
            continue
        valid.append(
            {
                "source": source,
                "target": target,
                "type": "cross_stage_dependency",
                "relation": relation,
                "reason": str(edge.get("reason", "")).strip() or f"{relation} dependency.",
            }
        )
    return valid

def propose_cross_stage_edges(
    nodes: list[dict],
    patient_id: str,
    client: ChatClient | None,
    cache_dir: Path | None,
    max_edges: int,
) -> list[dict]:
    # LLM 基于全部节点提议跨维度/跨模态的临床依赖边, 经规则校验过滤
    proposed = request_llm_edges(nodes, patient_id, client, cache_dir, max_edges)
    return validate_proposed_edges(proposed, nodes)


def dedupe_edges(edges: list[dict], node_by_id: dict[str, dict]) -> list[dict]:
    # 去除重复边、自环、悬空边(端点不存在), 并为每条边赋 edge_id。
    seen = set()
    out = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key in seen:
            continue
        if edge["source"] not in node_by_id or edge["target"] not in node_by_id:
            continue
        if edge["source"] == edge["target"]:
            continue
        seen.add(key)
        edge = dict(edge)
        edge["edge_id"] = f"edge_{len(out) + 1:04d}"
        out.append(edge)
    return out
