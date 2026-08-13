from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from playwright.sync_api import sync_playwright

from config import get_settings
from llm_client import ChatClient
from step2_evidence.graph import build_evidence_graph, stage_order


STAGE_COLORS = {
    "S0_PROFILE": "#0f9fb4",
    "S1_FP": "#7c5cff",
    "S2_DP": "#f06423",
    "S3_XR_XLA": "#f7a21b",
    "S4_CT": "#21a663",
    "S5_TMJ": "#126be8",
}
PALETTE = ["#0f9fb4", "#7c5cff", "#f06423", "#f7a21b", "#21a663", "#126be8", "#d14f8f", "#6f7d3c"]

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


def layout_nodes(
    nodes: list[dict],
    width: int,
    stage_indexes: dict[str, int],
) -> dict[str, tuple[float, float]]:
    """按阶段分层并水平排列全部节点。"""
    by_stage = defaultdict(list)
    for node in nodes:
        by_stage[node["introduced_stage"]].append(node)

    positions = {}
    left = 100
    right = width - 300
    for stage, items in by_stage.items():
        y = 100 + stage_indexes[stage] * 145
        ordered = sorted(items, key=lambda node: (node["source_turn_id"], node["label"]))
        count = len(ordered)
        xs = [(left + right) / 2] if count == 1 else [left + (right - left) * i / (count - 1) for i in range(count)]
        for x, node in zip(xs, ordered):
            positions[node["evidence_id"]] = (x, y)
    return positions


def edge_path(source: tuple[float, float], target: tuple[float, float]) -> str:
    """生成跨阶段边的 SVG 贝塞尔曲线路径。"""
    sx, sy = source
    tx, ty = target
    mid_y = (sy + ty) / 2
    bend = 60 if tx >= sx else -60
    return f"M {sx:.1f} {sy:.1f} C {sx + bend:.1f} {mid_y:.1f}, {tx - bend:.1f} {mid_y:.1f}, {tx:.1f} {ty:.1f}"


def render_html(
    graph: dict,
    evidence: list[dict],
    stages: list[dict],
    html_path: Path,
) -> None:
    """使用完整阶段、evidence 节点与 graph 边渲染证据图 HTML。"""
    nodes = []
    for item in evidence:
        node = dict(item)
        node["label"] = label_from_field(
            str(node.get("normalized", {}).get("field", "")),
            node["fact_text"],
        )
        nodes.append(node)
    nodes.sort(key=lambda node: (stage_order(node["introduced_stage"]), node["source_turn_id"], node["label"]))
    stage_counts = defaultdict(int)
    for node in nodes:
        stage_counts[node["introduced_stage"]] += 1
    ordered_stages = [stage["stage_id"] for stage in sorted(stages, key=lambda item: item["order"])]
    stages_by_id = {stage["stage_id"]: stage for stage in stages}
    stage_indexes = {stage: index for index, stage in enumerate(ordered_stages)}
    stage_colors = {
        stage: STAGE_COLORS.get(stage, PALETTE[index % len(PALETTE)])
        for index, stage in enumerate(ordered_stages)
    }
    canvas_width = max(1560, max(stage_counts.values(), default=1) * 115 + 360)
    canvas_height = max(360, len(ordered_stages) * 145 + 100)
    positions = layout_nodes(nodes, canvas_width, stage_indexes)
    node_by_id = {node["evidence_id"]: node for node in nodes}
    edge_list = graph["edges"]

    planes = []
    for stage in ordered_stages:
        stage_data = stages_by_id[stage]
        timepoint = stage_data.get("timepoint", {})
        date_text = str(timepoint.get("date_text") or "").strip()
        stage_type = stage_data["stage_type"].capitalize()
        label = STAGE_LABELS.get(stage, f"{date_text} · {stage_type}" if date_text else stage_type)
        label = f"{label}  ·  n={stage_counts[stage]}"
        y = 100 + stage_indexes[stage] * 145
        color = stage_colors[stage]
        points = f"60,{y+45} {canvas_width-620},{y+92} {canvas_width-300},{y+18} 380,{y-28}"
        planes.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.055" stroke="{color}" stroke-opacity="0.42" stroke-width="1.5" />'
            f'<text x="{canvas_width-40}" y="{y+18}" class="stage-label" fill="{color}">{html.escape(label)}</text>'
        )

    edge_svg = []
    for edge in edge_list:
        source = positions[edge["source"]]
        target = positions[edge["target"]]
        color = stage_colors[node_by_id[edge["source"]]["introduced_stage"]]
        title = html.escape(edge["reason"])
        path = edge_path(source, target)
        attributes = (
            f'data-edge-id="{html.escape(str(edge.get("edge_id", "")), quote=True)}" '
            f'data-source="{html.escape(edge["source"], quote=True)}" '
            f'data-target="{html.escape(edge["target"], quote=True)}" '
            f'data-type="{html.escape(str(edge.get("type", "")), quote=True)}" '
            f'data-relation="{html.escape(str(edge.get("relation", "")), quote=True)}" '
            f'data-reason="{html.escape(edge["reason"], quote=True)}"'
        )
        edge_svg.append(
            f'<path class="edge-hit" d="{path}" {attributes}></path>'
            f'<path class="edge support" d="{path}" stroke="{color}" marker-end="url(#arrow)" {attributes}>'
            f'<title>{title}</title></path>'
        )

    node_svg = []
    for node in nodes:
        x, y = positions[node["evidence_id"]]
        color = stage_colors[node["introduced_stage"]]
        label = html.escape(node["label"])
        fact = html.escape(node["fact_text"])
        dim = html.escape(DIMENSION_LABELS.get(node.get("clinical_dimension", "other"), "other"))
        search_text = html.escape(
            f"{node['label']} {node['fact_text']} {node['introduced_stage']} {dim}".lower(),
            quote=True,
        )
        node_svg.append(
            f'<g class="node" data-id="{html.escape(node["evidence_id"], quote=True)}" '
            f'data-search="{search_text}" data-fact="{fact}" '
            f'data-stage="{html.escape(node["introduced_stage"], quote=True)}" '
            f'data-source-turn="{html.escape(str(node["source_turn_id"]), quote=True)}" '
            f'data-modality="{html.escape(", ".join(node.get("modality", [])), quote=True)}" '
            f'data-fact-type="{html.escape(str(node.get("fact_type", "")), quote=True)}" '
            f'data-dimension="{html.escape(str(node.get("clinical_dimension", "")), quote=True)}" '
            f'data-normalized="{html.escape(json.dumps(node.get("normalized", {}), ensure_ascii=False), quote=True)}" '
            f'data-x="{x:.1f}" data-y="{y:.1f}" transform="translate({x:.1f},{y:.1f})">'
            f'<title>{fact}</title>'
            f'<ellipse cx="0" cy="6" rx="23" ry="8" fill="#1d2430" opacity="0.20"/>'
            f'<ellipse cx="0" cy="0" rx="22" ry="12" fill="{color}" opacity="0.88" stroke="#0b315f" stroke-width="1.2"/>'
            f'<ellipse cx="0" cy="-3" rx="15" ry="7" fill="#ffffff" opacity="0.35"/>'
            f'<text x="0" y="38" class="node-label">{label}</text>'
            f'<text x="0" y="56" class="node-dim">{dim}</text>'
            f'</g>'
        )

    patient_slug = graph["patient_id"].split("__")[-1].lower()
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evidence Graph - {html.escape(graph['patient_id'])}</title>
  <style>
    :root {{
      --ink: #17324d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f4f7fb;
      color: var(--ink);
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 12px 18px;
      background: rgba(255,255,255,0.96);
      border-bottom: 1px solid #d9e2ec;
      box-shadow: 0 2px 10px rgba(23,50,77,0.08);
    }}
    .title {{ font-size: 17px; font-weight: 700; white-space: nowrap; }}
    .summary {{ color: #667085; font-size: 13px; white-space: nowrap; }}
    .toolbar input {{
      width: min(360px, 32vw);
      padding: 8px 11px;
      border: 1px solid #cbd5e1;
      border-radius: 7px;
      font-size: 13px;
    }}
    .toolbar label {{ font-size: 13px; white-space: nowrap; }}
    .toolbar a {{ color: #126be8; text-decoration: none; font-size: 13px; white-space: nowrap; }}
    .toolbar a:hover {{ text-decoration: underline; }}
    .toolbar button {{ padding: 7px 10px; border: 1px solid #cbd5e1; border-radius: 7px; background: #fff; color: var(--ink); cursor: pointer; font-size: 13px; }}
    .toolbar button:hover {{ background: #eef5ff; border-color: #93b4df; }}
    .detail {{
      min-height: 42px;
      padding: 10px 18px;
      background: #eef5ff;
      border-bottom: 1px solid #d9e2ec;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .viewport {{ overflow: hidden; max-height: calc(100vh - 104px); padding: 16px; cursor: grab; }}
    .viewport.panning {{ cursor: grabbing; }}
    .canvas {{
      width: {canvas_width}px;
      background: #ffffff;
      border: 1px solid #dfe7ef;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(23,50,77,0.08);
      overflow: hidden;
    }}
    svg {{ width: {canvas_width}px; height: {canvas_height}px; display: block; }}
    .stage-label {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: 23px;
      font-weight: 700;
      text-anchor: end;
    }}
    .edge {{
      fill: none;
      stroke-width: 2.1;
      opacity: 0.72;
      pointer-events: stroke;
      cursor: pointer;
    }}
    .edge.support {{ stroke-width: 2.4; }}
    .edge-hit {{ fill:none; stroke:transparent; stroke-width:14; pointer-events:stroke; cursor:pointer; }}
    .edge:hover, .edge.selected {{ opacity: 1; stroke-width: 4; filter: drop-shadow(0 0 4px rgba(18,107,232,.45)); }}
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
    .node {{ cursor: grab; transition: opacity 0.15s ease; }}
    .node.dragging {{ cursor: grabbing; }}
    .node.muted {{ opacity: 0.10; }}
    .node.related ellipse:nth-of-type(2) {{ stroke: #f59e0b; stroke-width: 2.8; }}
    .node.selected ellipse:nth-of-type(2),
    .node:hover ellipse:nth-of-type(2) {{
      filter: drop-shadow(0 0 10px rgba(18, 107, 232, 0.35));
      stroke-width: 2.8;
    }}
    body.hide-edges .edge, body.hide-edges .edge-hit {{ display: none; }}
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="title">{html.escape(graph['patient_id'])}</div>
    <a href="../evaluation/{patient_slug}_results.html">← 返回病人结果</a>
    <div class="summary">{len(nodes)} evidence · {len(edge_list)} edges · {len(ordered_stages)} timepoints</div>
    <input id="search" type="search" placeholder="Search evidence or timepoint" />
    <label><input id="edges" type="checkbox" checked /> Show edges</label>
    <button id="zoom-out" type="button">−</button>
    <button id="zoom-in" type="button">＋</button>
    <button id="reset" type="button">Reset view</button>
  </div>
  <div id="detail" class="detail">拖动画布可平移，滚轮可缩放，节点可拖动；点击节点或边查看详细信息。</div>
  <div class="viewport">
    <div class="canvas">
      <svg viewBox="0 0 {canvas_width} {canvas_height}" role="img" aria-label="Layered evidence graph">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L10,4 L0,8 Z" fill="#126be8" opacity="0.78"></path>
          </marker>
        </defs>
        <g id="graph-layer">
          {"".join(planes)}
          {"".join(edge_svg)}
          {"".join(node_svg)}
        </g>
      </svg>
    </div>
  </div>
  <script>
    const svg = document.querySelector('svg');
    const layer = document.getElementById('graph-layer');
    const viewport = document.querySelector('.viewport');
    const nodes = [...document.querySelectorAll('.node')];
    const edges = [...document.querySelectorAll('.edge')];
    const edgeHits = [...document.querySelectorAll('.edge-hit')];
    const nodeById = Object.fromEntries(nodes.map(node => [node.dataset.id, node]));
    const search = document.getElementById('search');
    const detail = document.getElementById('detail');
    let scale = 1, panX = 0, panY = 0, draggingNode = null, panning = false, lastX = 0, lastY = 0;

    function applyView() {{ layer.setAttribute('transform', `translate(${{panX}} ${{panY}}) scale(${{scale}})`); }}
    function edgePath(source, target) {{
      const sx = Number(source.dataset.x), sy = Number(source.dataset.y);
      const tx = Number(target.dataset.x), ty = Number(target.dataset.y);
      const midY = (sy + ty) / 2, bend = tx >= sx ? 60 : -60;
      return `M ${{sx.toFixed(1)}} ${{sy.toFixed(1)}} C ${{(sx+bend).toFixed(1)}} ${{midY.toFixed(1)}}, ${{(tx-bend).toFixed(1)}} ${{midY.toFixed(1)}}, ${{tx.toFixed(1)}} ${{ty.toFixed(1)}}`;
    }}
    function updateEdges(nodeId) {{
      [...edges, ...edgeHits].filter(edge => edge.dataset.source === nodeId || edge.dataset.target === nodeId).forEach(edge => {{
        edge.setAttribute('d', edgePath(nodeById[edge.dataset.source], nodeById[edge.dataset.target]));
      }});
    }}
    function clearSelection() {{
      nodes.forEach(node => node.classList.remove('selected', 'related'));
      edges.forEach(edge => edge.classList.remove('selected'));
    }}
    function showNode(node) {{
      clearSelection(); node.classList.add('selected');
      edges.filter(edge => edge.dataset.source === node.dataset.id || edge.dataset.target === node.dataset.id).forEach(edge => {{
        edge.classList.add('selected');
        nodeById[edge.dataset.source].classList.add('related');
        nodeById[edge.dataset.target].classList.add('related');
      }});
      detail.textContent = `节点 ${{node.dataset.id}} | 阶段 ${{node.dataset.stage}} | 来源 turn ${{node.dataset.sourceTurn}} | 模态 ${{node.dataset.modality || '-'}} | 类型 ${{node.dataset.factType || '-'}} | 维度 ${{node.dataset.dimension || '-'}}\n${{node.dataset.fact}}\n规范化: ${{node.dataset.normalized}}`;
    }}
    function showEdge(edge) {{
      clearSelection();
      const visibleEdge = edges.find(item => item.dataset.edgeId === edge.dataset.edgeId);
      if (visibleEdge) visibleEdge.classList.add('selected');
      nodeById[edge.dataset.source].classList.add('related');
      nodeById[edge.dataset.target].classList.add('related');
      detail.textContent = `边 ${{edge.dataset.edgeId || '-'}} | ${{edge.dataset.type || '-'}} / ${{edge.dataset.relation || '-'}}\n${{edge.dataset.source}} → ${{edge.dataset.target}}\n${{edge.dataset.reason}}`;
    }}
    function pointInLayer(event) {{
      const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      return point.matrixTransform(layer.getScreenCTM().inverse());
    }}
    function setScale(next) {{ scale = Math.max(.35, Math.min(2.5, next)); applyView(); }}

    search.addEventListener('input', () => {{
      const query = search.value.trim().toLowerCase();
      nodes.forEach(node => node.classList.toggle('muted', Boolean(query && !node.dataset.search.includes(query))));
    }});
    document.getElementById('edges').addEventListener('change', event => document.body.classList.toggle('hide-edges', !event.target.checked));
    document.getElementById('zoom-in').addEventListener('click', () => setScale(scale * 1.2));
    document.getElementById('zoom-out').addEventListener('click', () => setScale(scale / 1.2));
    document.getElementById('reset').addEventListener('click', () => {{
      scale = 1; panX = 0; panY = 0; applyView(); clearSelection();
      detail.textContent = '拖动画布可平移，滚轮可缩放，节点可拖动；点击节点或边查看详细信息。';
    }});
    viewport.addEventListener('wheel', event => {{ event.preventDefault(); setScale(scale * (event.deltaY < 0 ? 1.12 : .89)); }}, {{passive:false}});
    svg.addEventListener('pointerdown', event => {{
      const node = event.target.closest('.node');
      if (node) {{ draggingNode = node; node.classList.add('dragging'); event.stopPropagation(); return; }}
      if (event.target.closest('.edge, .edge-hit')) return;
      panning = true; lastX = event.clientX; lastY = event.clientY; viewport.classList.add('panning');
    }});
    window.addEventListener('pointermove', event => {{
      if (draggingNode) {{
        const point = pointInLayer(event);
        draggingNode.dataset.x = point.x.toFixed(1); draggingNode.dataset.y = point.y.toFixed(1);
        draggingNode.setAttribute('transform', `translate(${{point.x.toFixed(1)}},${{point.y.toFixed(1)}})`);
        updateEdges(draggingNode.dataset.id);
      }} else if (panning) {{
        panX += (event.clientX-lastX)/scale; panY += (event.clientY-lastY)/scale;
        lastX = event.clientX; lastY = event.clientY; applyView();
      }}
    }});
    window.addEventListener('pointerup', () => {{
      if (draggingNode) draggingNode.classList.remove('dragging');
      draggingNode = null; panning = false; viewport.classList.remove('panning');
    }});
    nodes.forEach(node => node.addEventListener('click', event => {{ event.stopPropagation(); showNode(node); }}));
    [...edges, ...edgeHits].forEach(edge => edge.addEventListener('click', event => {{ event.stopPropagation(); showEdge(edge); }}));
  </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")


def render_png(html_path: Path, png_path: Path) -> None:
    """使用无头浏览器截图 HTML 中的完整证据图。"""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1)
        page.goto(html_path.resolve().as_uri())
        page.locator(".canvas").screenshot(path=str(png_path))
        browser.close()


def main() -> None:
    """证据图可视化入口。

    功能: 读 evidence.json→(强关系规则+候选生成+独立审核)建图→写 JSON、HTML 与 PNG→打印统计。
    输入: 无(从 get_settings() 读取配置与 outputs 路径、LLM 凭据)。
    输出: 无返回值; 产生图 JSON/HTML/PNG, 控制台打印统计摘要。
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
    evidence = json.loads(evidence_json.read_text(encoding="utf-8"))["evidence"]
    standard = json.loads(
        (base / "trajectories" / "standard_trajectory.json").read_text(encoding="utf-8")
    )
    graph = build_evidence_graph(evidence_json, client=client, cache_dir=base / "cache", max_edges=settings.graph_max_edges)
    graph_path = base / "graph" / "evidence_graph.json"
    html_path = base / "graph" / "evidence_graph.html"
    png_path = base / "graph" / "evidence_graph.png"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(graph, evidence, standard["stages"], html_path)
    render_png(html_path, png_path)
    print(
        f"[benchmark][{graph['patient_id']}][step2/visualization] "
        f"nodes={len(evidence)} edges={len(graph['edges'])} "
        f"graph_json={graph_path} graph_html={html_path} graph_png={png_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
