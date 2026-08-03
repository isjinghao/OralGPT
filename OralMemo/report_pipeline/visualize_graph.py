from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent

MAX_NODES_PER_STAGE = 10  # 每个阶段最多展示的节点数, 避免过密

# 阶段配色, 按阶段序号循环取色(通用: 与固定 6 阶段无关)。
STAGE_PALETTE = [
    "#7c5cff", "#f06423", "#f7a21b", "#21a663", "#126be8", "#dc2626",
    "#0891b2", "#7c3aed", "#ca8a04", "#0f766e",
]

# 画布几何参数
CANVAS_W = 1560
CONTENT_LEFT = 210
CONTENT_RIGHT = 1300
TOP_Y = 150
ROW_GAP = 170


def stage_order(stage: str) -> int:
    m = re.search(r"\d+", stage or "")
    return int(m.group()) if m else 0


def node_label(node: dict) -> str:
    field = str((node.get("normalized") or {}).get("field") or "").replace("_", " ").strip()
    text = field if field and field != "None" else (node.get("fact_text") or "")
    return text[:26]


def load_graph(out_dir: Path) -> dict:
    path = out_dir / "graph" / "evidence_graph.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_tmonths(out_dir: Path) -> dict[str, int]:
    """从标准轨迹读取每个阶段的 t_months，供图上显示。"""
    path = out_dir / "trajectories" / "standard_trajectory.json"
    stages = json.loads(path.read_text(encoding="utf-8"))["stages"]
    return {stage["stage_id"]: stage["timepoint"]["t_months"] for stage in stages}


def select_visible(nodes: list[dict]) -> list[dict]:
    by_stage: dict[str, list[dict]] = {}
    for n in nodes:
        by_stage.setdefault(n["introduced_stage"], []).append(n)
    visible = []
    for stage, items in by_stage.items():
        items = sorted(items, key=lambda n: (n.get("source_turn_id", 0), n["evidence_id"]))
        visible.extend(items[:MAX_NODES_PER_STAGE])
    return visible


def layout(nodes: list[dict]) -> tuple[dict[str, tuple[float, float]], list[str], dict[str, str]]:
    """分层布局: 阶段序号越小越靠上; 同层节点水平均布。

    返回: (节点id->(x,y)坐标, 有序阶段列表, 阶段->颜色)。
    """
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        by_stage[n["introduced_stage"]].append(n)
    stages = sorted(by_stage, key=stage_order)
    stage_color = {s: STAGE_PALETTE[i % len(STAGE_PALETTE)] for i, s in enumerate(stages)}

    pos: dict[str, tuple[float, float]] = {}
    for row, stage in enumerate(stages):
        items = sorted(by_stage[stage], key=node_label)
        y = TOP_Y + row * ROW_GAP
        count = len(items)
        if count == 1:
            xs = [(CONTENT_LEFT + CONTENT_RIGHT) / 2]
        else:
            xs = [CONTENT_LEFT + (CONTENT_RIGHT - CONTENT_LEFT) * i / (count - 1) for i in range(count)]
        for x, node in zip(xs, items):
            pos[node["evidence_id"]] = (x, y)
    return pos, stages, stage_color


def edge_path(source: tuple[float, float], target: tuple[float, float], cross: bool) -> str:
    """生成边的 SVG 贝塞尔路径(阶段内轻微抬起, 跨阶段走中点弯曲)。"""
    sx, sy = source
    tx, ty = target
    if not cross:  # 同层/近层, 轻微上抬
        dx = tx - sx
        lift = 34 if dx >= 0 else -34
        return f"M {sx:.1f} {sy:.1f} C {sx + dx * 0.35:.1f} {sy - lift:.1f}, {sx + dx * 0.65:.1f} {ty - lift:.1f}, {tx:.1f} {ty:.1f}"
    mid_y = (sy + ty) / 2
    bend = 60 if tx >= sx else -60
    return f"M {sx:.1f} {sy:.1f} C {sx + bend:.1f} {mid_y:.1f}, {tx - bend:.1f} {mid_y:.1f}, {tx:.1f} {ty:.1f}"


def render_html(graph: dict, html_path: Path, tmonths: dict[str, int] | None = None) -> None:
    tmonths = tmonths or {}
    nodes = select_visible(graph["nodes"])
    visible_ids = {n["evidence_id"] for n in nodes}
    pos, stages, stage_color = layout(nodes)
    stage_of = {n["evidence_id"]: n["introduced_stage"] for n in nodes}

    canvas_h = TOP_Y + max(len(stages), 1) * ROW_GAP + 40

    # 阶段带 + 阶段标题(居中, 放在每个阶段框上方; 带 t_months)
    box_left = CONTENT_LEFT - 70
    box_w = CONTENT_RIGHT - CONTENT_LEFT + 140
    box_cx = box_left + box_w / 2
    planes = []
    for row, stage in enumerate(stages):
        y = TOP_Y + row * ROW_GAP
        color = stage_color[stage]
        t = tmonths.get(stage)
        title = html.escape(stage) + ("" if t is None else f"  ·  {t} months")
        planes.append(
            f'<rect x="{box_left:.0f}" y="{y - 44:.0f}" width="{box_w:.0f}" '
            f'height="88" rx="16" fill="{color}" fill-opacity="0.05" stroke="{color}" '
            f'stroke-opacity="0.35" stroke-width="1.4" />'
            f'<text x="{box_cx:.0f}" y="{y - 54:.0f}" class="stage-label" '
            f'fill="{color}" text-anchor="middle">{title}</text>'
        )

    # 边
    edge_svg = []
    for edge in graph.get("edges", []):
        if edge["source"] not in visible_ids or edge["target"] not in visible_ids:
            continue
        source = pos[edge["source"]]
        target = pos[edge["target"]]
        cross = edge.get("type") == "cross_stage_dependency"
        color = stage_color.get(stage_of.get(edge["source"]), STAGE_PALETTE[0])
        cls = "edge cross" if cross else "edge intra"
        marker = "url(#arrow)" if cross else ""
        title = html.escape(str(edge.get("reason", "")))
        edge_svg.append(
            f'<path class="{cls}" d="{edge_path(source, target, cross)}" stroke="{color}" '
            f'marker-end="{marker}"><title>{title}</title></path>'
        )

    # 节点 + 标签
    node_svg = []
    for node in nodes:
        x, y = pos[node["evidence_id"]]
        color = stage_color[node["introduced_stage"]]
        label = html.escape(node_label(node))
        fact = html.escape(node.get("fact_text", ""))
        node_svg.append(
            f'<g class="node" transform="translate({x:.1f},{y:.1f})">'
            f'<title>{fact}</title>'
            f'<ellipse cx="0" cy="6" rx="23" ry="8" fill="#1d2430" opacity="0.20"/>'
            f'<ellipse cx="0" cy="0" rx="22" ry="12" fill="{color}" opacity="0.88" stroke="#0b315f" stroke-width="1.2"/>'
            f'<ellipse cx="0" cy="-3" rx="15" ry="7" fill="#ffffff" opacity="0.35"/>'
            f'<text x="0" y="34" class="node-label">{label}</text>'
            f'</g>'
        )

    patient_id = html.escape(str(graph.get("patient_id", "")))
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evidence Graph (time axis) - {patient_id}</title>
  <style>
    :root {{ --ink: #17324d; }}
    body {{ margin: 0; background: #ffffff; color: var(--ink); font-family: Georgia, "Times New Roman", serif; }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
    h1 {{ font-size: 20px; font-weight: 700; margin: 0 0 16px; color: var(--ink); }}
    svg {{ width: 100%; height: auto; display: block; }}
    .stage-label {{ font-family: Georgia, "Times New Roman", serif; font-size: 18px; font-weight: 700; }}
    .edge {{ fill: none; stroke-width: 2.1; opacity: 0.7; }}
    .edge.intra {{ opacity: 0.5; }}
    .edge.cross {{ stroke-width: 2.4; }}
    .node-label {{ font-family: "Trebuchet MS", Verdana, sans-serif; font-size: 12px; fill: #18324f; text-anchor: middle; font-weight: 700; }}
    .node:hover ellipse:nth-of-type(2) {{ filter: drop-shadow(0 0 10px rgba(18,107,232,0.28)); stroke-width: 2.4; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Evidence graph (time axis) &mdash; {patient_id}</h1>
    <svg viewBox="0 0 {CANVAS_W} {canvas_h:.0f}" role="img" aria-label="Layered evidence graph">
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
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    out_dir = BENCH_ROOT / "outputs" / "report" / args.name
    graph = load_graph(out_dir)
    html_path = out_dir / "graph" / "evidence_graph.html"
    render_html(graph, html_path, load_tmonths(out_dir))
    print(f"[viz] {args.name}: nodes={len(graph['nodes'])} edges={len(graph.get('edges', []))} -> {html_path}")


if __name__ == "__main__":
    main()
