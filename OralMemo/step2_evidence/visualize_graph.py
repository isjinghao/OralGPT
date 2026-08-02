from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path

from config import get_settings
from llm_client import ChatClient
from step2_evidence.graph import build_evidence_graph, stage_order


STAGE_COLORS = {
    "S0_PROFILE": "#7c5cff",
    "S1_FP": "#7c5cff",
    "S2_DP": "#f06423",
    "S3_XR_XLA": "#f7a21b",
    "S4_CT": "#21a663",
    "S5_TMJ": "#126be8",
}

STAGE_LABELS = {
    "S0_PROFILE": "Stage 0  Profile",
    "S1_FP": "Stage 1  FP",
    "S2_DP": "Stage 2  DP",
    "S3_XR_XLA": "Stage 3  XR / XLData",
    "S4_CT": "Stage 4  CT",
    "S5_TMJ": "Stage 5  TMJ",
}

DIMENSION_LABELS = {
    "profile": "profile",
    "sagittal_relationship": "sagittal",
    "vertical_pattern": "vertical",
    "facial_asymmetry": "asymmetry",
    "dental_status": "dental",
    "tmj_status": "TMJ",
    "treatment_risk": "risk",
    "other": "other",
}

IMPORTANT_FIELDS = [
    "chief_complaint",
    "chin_deviation",
    "facial_profile",
    "molar_relationship",
    "overjet",
    "overbite",
    "lower_midline_deviation",
    "caries",
    "gingival_fistula",
    "skeletal_class",
    "ANB",
    "Wits",
    "SNB",
    "impaction_status",
    "menton_deviation",
    "mandible_sagittal_position",
    "maxilla_sagittal_position",
    "mouth_opening",
    "opening_clicks",
]


def label_from_field(field: str, fact_text: str) -> str:
    """生成节点短标签(仅服务可视化)。

    功能: 优先用规范化字段名(下划线转空格)作标签, 否则截取 fact_text, 最长 34 字符。
    输入: field 规范化字段名; fact_text 事实文本。
    输出: str - 节点显示标签。
    """
    clean = str(field).replace("_", " ")
    if clean and clean != "None":
        return clean[:34]
    return fact_text[:34]


def select_visible_nodes(graph: dict) -> list[dict]:
    """挑选可视化展示的节点。

    功能: 按重点字段与临床维度打分, 每阶段择优挑选(S0 取 3, 其余取 6); 现场派生显示短标签。
    输入: graph 证据图对象。
    输出: list[dict] - 按阶段/标签排序的待展示节点。
    """
    by_stage = defaultdict(list)
    for node in graph["nodes"]:
        field = str(node.get("normalized", {}).get("field", ""))
        score = 0
        for rank, key in enumerate(IMPORTANT_FIELDS):
            if key.lower() in field.lower():
                score = 100 - rank
                break
        if node.get("clinical_dimension", "other") in {"sagittal_relationship", "facial_asymmetry", "dental_status", "tmj_status"}:
            score += 10
        node = dict(node)
        node["_score"] = score
        node["label"] = label_from_field(field, node["fact_text"])
        by_stage[node["introduced_stage"]].append(node)

    visible = []
    for stage, nodes in by_stage.items():
        limit = 3 if stage == "S0_PROFILE" else 6
        visible.extend(sorted(nodes, key=lambda n: (-n["_score"], n["source_turn_id"], n["label"]))[:limit])
    return sorted(visible, key=lambda n: (stage_order(n["introduced_stage"]), n["label"]))


def layout_nodes(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    """计算节点布局坐标。

    功能: 按阶段分层(纵向), 同层节点水平均布。
    输入: nodes 待展示节点列表。
    输出: dict - 节点 id 到 (x, y) 坐标的映射。
    """
    by_stage = defaultdict(list)
    for node in nodes:
        by_stage[node["introduced_stage"]].append(node)

    positions = {}
    width = 1380
    left = 190
    right = width - 260
    y_base = 920
    y_gap = 145
    for stage, items in by_stage.items():
        order = stage_order(stage)
        y = y_base - order * y_gap
        count = len(items)
        if count == 1:
            xs = [(left + right) / 2]
        else:
            xs = [left + (right - left) * i / (count - 1) for i in range(count)]
        for x, node in zip(xs, sorted(items, key=lambda n: n["label"])):
            positions[node["evidence_id"]] = (x, y)
    return positions


def edge_path(source: tuple[float, float], target: tuple[float, float], edge_type: str) -> str:
    """生成边的 SVG 路径。

    功能: 按边类型生成不同弯曲度的贝塞尔曲线路径(阶段内/跨阶段)。
    输入: source 起点坐标; target 终点坐标; edge_type 边类型。
    输出: str - SVG path 的 d 属性字符串。
    """
    sx, sy = source
    tx, ty = target
    if edge_type == "intra_stage_link":
        dx = tx - sx
        lift = 34 if dx >= 0 else -34
        return f"M {sx:.1f} {sy:.1f} C {sx + dx * 0.35:.1f} {sy - lift:.1f}, {sx + dx * 0.65:.1f} {ty - lift:.1f}, {tx:.1f} {ty:.1f}"
    mid_y = (sy + ty) / 2
    bend = 60 if tx >= sx else -60
    return f"M {sx:.1f} {sy:.1f} C {sx + bend:.1f} {mid_y:.1f}, {tx - bend:.1f} {mid_y:.1f}, {tx:.1f} {ty:.1f}"


def render_html(graph: dict, html_path: Path) -> None:
    """渲染证据图为交互式 HTML。

    功能: 选点→布局→绘制阶段平面/边/节点, 生成纯白背景、仅含图形主体的 SVG/HTML 并写文件。
    输入: graph 证据图对象; html_path 输出 HTML 路径。
    输出: 无返回值, 产生 HTML 文件。
    """
    nodes = select_visible_nodes(graph)
    visible_ids = {node["evidence_id"] for node in nodes}
    positions = layout_nodes(nodes)
    edge_list = [e for e in graph["edges"] if e["source"] in visible_ids and e["target"] in visible_ids]

    planes = []
    for stage, label in STAGE_LABELS.items():
        order = stage_order(stage)
        y = 920 - order * 145
        color = STAGE_COLORS[stage]
        points = f"120,{y+45} 960,{y+92} 1250,{y+18} 405,{y-28}"
        planes.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.055" stroke="{color}" stroke-opacity="0.42" stroke-width="1.5" />'
            f'<text x="1270" y="{y+18}" class="stage-label" fill="{color}">{html.escape(label)}</text>'
        )

    edge_svg = []
    for edge in edge_list:
        source = positions[edge["source"]]
        target = positions[edge["target"]]
        color = STAGE_COLORS[next(n["introduced_stage"] for n in nodes if n["evidence_id"] == edge["source"])]
        is_context = edge["type"] == "context_consistency"
        cls = "edge context" if is_context else "edge support"
        marker = "" if is_context else "url(#arrow)"
        dash = 'stroke-dasharray="5 5"' if is_context else ""
        title = html.escape(edge["reason"])
        edge_svg.append(
            f'<path class="{cls}" d="{edge_path(source, target, edge["type"])}" stroke="{color}" marker-end="{marker}" {dash}>'
            f'<title>{title}</title></path>'
        )

    node_svg = []
    for node in nodes:
        x, y = positions[node["evidence_id"]]
        color = STAGE_COLORS[node["introduced_stage"]]
        label = html.escape(node["label"])
        fact = html.escape(node["fact_text"])
        dim = html.escape(DIMENSION_LABELS.get(node.get("clinical_dimension", "other"), "other"))
        node_svg.append(
            f'<g class="node" transform="translate({x:.1f},{y:.1f})">'
            f'<title>{fact}</title>'
            f'<ellipse cx="0" cy="6" rx="23" ry="8" fill="#1d2430" opacity="0.20"/>'
            f'<ellipse cx="0" cy="0" rx="22" ry="12" fill="{color}" opacity="0.88" stroke="#0b315f" stroke-width="1.2"/>'
            f'<ellipse cx="0" cy="-3" rx="15" ry="7" fill="#ffffff" opacity="0.35"/>'
            f'<text x="0" y="38" class="node-label">{label}</text>'
            f'<text x="0" y="56" class="node-dim">{dim}</text>'
            f'</g>'
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OralMemBench Evidence Graph - CHENFANG</title>
  <style>
    :root {{
      --ink: #17324d;
    }}
    body {{
      margin: 0;
      background: #ffffff;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px;
    }}
    .canvas {{
      background: #ffffff;
    }}
    svg {{ width: 100%; height: auto; display: block; }}
    .stage-label {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: 23px;
      font-weight: 700;
    }}
    .edge {{
      fill: none;
      stroke-width: 2.1;
      opacity: 0.72;
    }}
    .edge.support {{ stroke-width: 2.4; }}
    .edge.context {{ opacity: 0.35; stroke-width: 1.4; }}
    .node-label {{
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 12px;
      fill: #18324f;
      text-anchor: middle;
      font-weight: 700;
    }}
    .node-dim {{
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 10px;
      fill: #667085;
      text-anchor: middle;
    }}
    .node:hover ellipse:nth-of-type(2) {{
      filter: drop-shadow(0 0 10px rgba(18, 107, 232, 0.28));
      stroke-width: 2.4;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="canvas">
      <svg viewBox="0 0 1560 1040" role="img" aria-label="Layered evidence graph">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L10,4 L0,8 Z" fill="#126be8" opacity="0.78"></path>
          </marker>
        </defs>
        {"".join(planes)}
        {"".join(edge_svg)}
        {"".join(node_svg)}
      </svg>
    </div>
  </div>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    """证据图可视化入口。

    功能: 读 evidence.json→(强关系规则+候选生成+独立审核)建图→写 evidence_graph.json 与 evidence_graph.html→打印统计。
    输入: 无(从 get_settings() 读取配置与 outputs 路径、LLM 凭据)。
    输出: 无返回值; 产生图 JSON/HTML, 控制台打印统计摘要。
    """
    settings = get_settings()
    base = settings.output_root
    evidence_json = base / "evidence" / "evidence.json"
    cfg = settings.llm_for("benchmark")
    client = ChatClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
    )
    graph = build_evidence_graph(evidence_json, client=client, cache_dir=base / "cache", max_edges=settings.graph_max_edges)
    graph_path = base / "graph" / "evidence_graph.json"
    html_path = base / "graph" / "evidence_graph.html"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    # 落盘精简图: step3 只消费 edges; nodes 仅服务可视化, 渲染时由内存图即时生成, 不落盘。
    persisted = {key: value for key, value in graph.items() if key != "nodes"}
    graph_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(graph, html_path)
    print(json.dumps({
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "graph_json": str(graph_path),
        "graph_html": str(html_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
