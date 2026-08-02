from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from string import Template

import yaml

from llm_client import ChatClient


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
EDGE_PROMPT_PATH = PROMPT_DIR / "graph_edges.yaml"
REVIEW_PROMPT_PATH = PROMPT_DIR / "graph_edge_review.yaml"


def stage_order(stage: str) -> int:
    """Extract a sortable stage number from a stage identifier."""
    match = re.search(r"\d+", stage or "")
    return int(match.group()) if match else 0


def _attribute_key(node: dict) -> tuple[str, str, str, str]:
    normalized = node.get("normalized", {})
    return (
        str(node.get("clinical_dimension", "other")),
        str(normalized.get("field") or ""),
        str(normalized.get("tooth") or ""),
        str(normalized.get("side") or ""),
    )


def _canonical_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold().strip())


def _normalized_value(node: dict) -> tuple[str, str]:
    normalized = node.get("normalized", {})
    return _canonical_text(normalized.get("value")), _canonical_text(normalized.get("unit"))


def _node_lines(nodes: list[dict]) -> str:
    return "\n".join(
        f"- {node['evidence_id']} | stage={node['introduced_stage']} | "
        f"dimension={node.get('clinical_dimension', 'other')} | "
        f"field={node.get('normalized', {}).get('field')} | "
        f"tooth={node.get('normalized', {}).get('tooth')} | "
        f"side={node.get('normalized', {}).get('side')} | "
        f"value={node.get('normalized', {}).get('value')} | "
        f"unit={node.get('normalized', {}).get('unit')} | "
        f"fact={node['fact_text']}"
        for node in sorted(nodes, key=lambda item: (stage_order(item["introduced_stage"]), item["source_turn_id"], item["evidence_id"]))
    )


def build_strong_edges(nodes: list[dict]) -> list[dict]:
    """Create deterministic links only for identical repeated measurements."""
    edges = []
    by_attribute = defaultdict(list)
    for node in nodes:
        key = _attribute_key(node)
        if key[0] == "other" or not key[1] or not _normalized_value(node)[0]:
            continue
        by_attribute[key].append(node)

    for items in by_attribute.values():
        by_stage = defaultdict(list)
        for node in items:
            by_stage[node["introduced_stage"]].append(node)
        ordered_stages = sorted(by_stage, key=stage_order)
        for early_stage, later_stage in zip(ordered_stages, ordered_stages[1:]):
            early = sorted(by_stage[early_stage], key=lambda item: (item["source_turn_id"], item["evidence_id"]))[-1]
            later = sorted(by_stage[later_stage], key=lambda item: (item["source_turn_id"], item["evidence_id"]))[0]
            if _normalized_value(early) != _normalized_value(later):
                continue
            edges.append(
                {
                    "source": early["evidence_id"],
                    "target": later["evidence_id"],
                    "type": "measurement_link",
                    "relation": "confirms",
                    "reason": "Repeated measurement of the same clinical attribute.",
                }
            )
    return edges


def _template(path: Path) -> Template:
    return Template(yaml.safe_load(path.read_text(encoding="utf-8"))["template"])


def _valid_pairs(items: list[dict], nodes: dict[str, dict], limit: int) -> list[dict]:
    valid = []
    seen = set()
    for item in items:
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        reason = str(item.get("reason", "")).strip()
        key = (source, target)
        if (
            key in seen
            or source not in nodes
            or target not in nodes
            or source == target
            or not reason
            or len(reason.split()) > 15
            or stage_order(nodes[source]["introduced_stage"]) >= stage_order(nodes[target]["introduced_stage"])
        ):
            continue
        seen.add(key)
        valid.append({"source": source, "target": target, "reason": reason})
        if len(valid) == limit:
            break
    return valid


def _review_edges(
    client: ChatClient,
    candidates: list[dict],
    nodes: list[dict],
    max_edges: int,
) -> list[dict]:
    if not candidates:
        return []
    prompt = _template(REVIEW_PROMPT_PATH).substitute(
        candidates=json.dumps(candidates, ensure_ascii=False),
        nodes=_node_lines(nodes),
        max_edges=max_edges,
    )
    reviewed = client.complete_json(prompt, max_tokens=12000)
    node_by_id = {node["evidence_id"]: node for node in nodes}
    candidate_keys = {(item["source"], item["target"]) for item in candidates}
    clinical = _valid_pairs(reviewed.get("clinical_support", []), node_by_id, max_edges)
    context = _valid_pairs(reviewed.get("context_consistency", []), node_by_id, max_edges)
    clinical = [item for item in clinical if (item["source"], item["target"]) in candidate_keys]
    clinical_keys = {(item["source"], item["target"]) for item in clinical}
    context = [
        item for item in context
        if (item["source"], item["target"]) in candidate_keys
        and (item["source"], item["target"]) not in clinical_keys
    ]
    return [
        {**item, "type": "clinical_support", "relation": "supports"}
        for item in clinical
    ] + [
        {**item, "type": "context_consistency", "relation": "compatible"}
        for item in context
    ]


def propose_cross_stage_edges(
    nodes: list[dict],
    patient_id: str,
    client: ChatClient | None,
    cache_dir: Path | None,
    max_edges: int,
) -> list[dict]:
    if client is None:
        raise ValueError("A ChatClient is required to generate clinical evidence edges")

    stage_lines = "\n".join(
        f"- {stage} (order {stage_order(stage)})"
        for stage in sorted({node["introduced_stage"] for node in nodes}, key=stage_order)
    )
    candidate_prompt = _template(EDGE_PROMPT_PATH).substitute(
        patient_id=patient_id,
        stage_order=stage_lines,
        nodes=_node_lines(nodes),
        max_edges=max_edges,
    )
    reviewer_template = REVIEW_PROMPT_PATH.read_text(encoding="utf-8")
    cache_key = hashlib.sha256(f"{client.model}\0{candidate_prompt}\0{reviewer_template}".encode("utf-8")).hexdigest()
    cache_path = cache_dir / "graph_edges.json" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("_cache_key") == cache_key:
            return cached.get("edges", [])

    candidates = client.complete_json(candidate_prompt, max_tokens=12000).get("candidates", [])
    node_by_id = {node["evidence_id"]: node for node in nodes}
    candidates = _valid_pairs(candidates, node_by_id, max_edges)
    edges = _review_edges(client, candidates, nodes, max_edges)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"edges": edges, "_cache_key": cache_key}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return edges


def dedupe_edges(edges: list[dict], nodes: dict[str, dict]) -> list[dict]:
    seen = set()
    out = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key in seen or edge["source"] not in nodes or edge["target"] not in nodes:
            continue
        seen.add(key)
        out.append({**edge, "edge_id": f"edge_{len(out) + 1:04d}"})
    return out


def build_evidence_graph(
    evidence_json: Path,
    client: ChatClient | None = None,
    cache_dir: Path | None = None,
    max_edges: int = 25,
) -> dict:
    """Build strong measurement links, reviewed clinical support, and visual-only context edges."""
    evidence_data = json.loads(evidence_json.read_text(encoding="utf-8"))
    nodes = evidence_data["evidence"]
    node_ids = [node["evidence_id"] for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Evidence catalog contains duplicate evidence_id values")
    node_by_id = {node["evidence_id"]: node for node in nodes}
    edges = build_strong_edges(nodes)
    edges.extend(propose_cross_stage_edges(nodes, evidence_data["patient_id"], client, cache_dir, max_edges))
    return {
        "patient_id": evidence_data["patient_id"],
        "nodes": nodes,
        "edges": dedupe_edges(edges, node_by_id),
    }
