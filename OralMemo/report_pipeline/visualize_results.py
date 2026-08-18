"""生成报告抽取时间线的交互式 HTML。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from utils.json_utils import read_json

BENCH_ROOT = Path(__file__).resolve().parent.parent


def relative_asset_path(raw_path: str, output_dir: Path) -> str:
    asset = BENCH_ROOT / raw_path.lstrip("/\\")
    if not asset.is_file():
        raise FileNotFoundError(f"Timeline image not found: {asset}")
    return Path(os.path.relpath(asset, output_dir)).as_posix()


def collect_report_data(report_dir: Path) -> dict:
    trajectory_path = report_dir / "trajectories" / "standard_trajectory.json"
    captions_path = report_dir / "raw" / "captions.json"
    trajectory = read_json(trajectory_path)
    captions = read_json(captions_path) if captions_path.exists() else {}

    stages = []
    for stage in sorted(trajectory["stages"], key=lambda item: item["order"]):
        qa_pairs = []
        for qa in stage.get("qa_pairs", []):
            qa_pairs.append(
                {
                    "source_turn_id": qa["source_turn_id"],
                    "question": qa.get("human", ""),
                    "answer": qa.get("assistant", ""),
                    "role": qa.get("role", ""),
                    "image_paths": [
                        relative_asset_path(path, report_dir)
                        for path in qa.get("image_paths", [])
                    ],
                }
            )
        stages.append(
            {
                "stage_id": stage["stage_id"],
                "order": stage["order"],
                "stage_type": stage["stage_type"],
                "modality": stage.get("modality", []),
                "timepoint": stage.get("timepoint", {}),
                "qa_pairs": qa_pairs,
            }
        )

    figures = []
    for figure, entry in captions.items():
        figures.append(
            {
                "figure": figure,
                "caption": entry.get("caption", ""),
                "image_paths": [
                    relative_asset_path(path, report_dir)
                    for path in entry.get("images", [])
                ],
            }
        )

    return {
        "patient_name": trajectory.get("patient_name", report_dir.name),
        "patient_id": trajectory.get("patient_id", ""),
        "stages": stages,
        "figures": figures,
        "stats": {
            "stages": len(stages),
            "qa_pairs": sum(len(stage["qa_pairs"]) for stage in stages),
            "images": sum(len(figure["image_paths"]) for figure in figures),
        },
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__REPORT_NAME__ 报告时间线</title>
<style>
:root { --bg:#f6f8fb; --card:#fff; --text:#15202b; --muted:#667085; --line:#e5e7eb; --blue:#2563eb; --blue-dark:#1d4ed8; --green:#16a34a; --amber:#d97706; --purple:#7c3aed; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif; background:var(--bg); color:var(--text); }
header { padding:24px 28px 18px; background:linear-gradient(135deg,#0f172a,#1d4ed8); color:#fff; }
h1 { margin:0 0 8px; font-size:26px; }
.subtitle { opacity:.88; font-size:14px; }
.container { padding:20px 28px 36px; max-width:1500px; margin:0 auto; }
.tabs { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
.tab { border:1px solid var(--line); border-radius:10px; padding:10px 18px; background:#fff; color:#475467; cursor:pointer; font-size:14px; }
.tab:hover { border-color:#93c5fd; color:var(--blue); }
.tab.active { color:#fff; border-color:var(--blue); background:var(--blue); }
.tab.timeline-tab { font-weight:700; border-width:2px; border-color:#60a5fa; box-shadow:0 4px 12px rgba(37,99,235,.16); }
.tab.timeline-tab.active { background:linear-gradient(135deg,#2563eb,#7c3aed); border-color:transparent; box-shadow:0 8px 22px rgba(79,70,229,.28); }
.tab-panel { display:none; min-height:1px; }
.tab-panel.active { display:block; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; box-shadow:0 4px 14px rgba(15,23,42,.05); margin-bottom:16px; }
.timeline-panel { border:2px solid #bfdbfe; background:linear-gradient(180deg,#f8fbff 0,#fff 180px); box-shadow:0 12px 30px rgba(37,99,235,.10); }
.timeline-heading { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:16px; }
.timeline-heading h2 { margin:0 0 6px; font-size:20px; color:#1e3a8a; }
.small { color:var(--muted); font-size:12px; line-height:1.5; }
.stats { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
.stat { border:1px solid #bfdbfe; background:#eff6ff; border-radius:999px; padding:5px 10px; color:#1e40af; font-size:12px; font-weight:650; white-space:nowrap; }
.fold { border:1px solid var(--line); border-radius:12px; background:#fcfcfd; overflow:hidden; }
.fold + .fold { margin-top:12px; }
.fold > summary { cursor:pointer; list-style:none; }
.fold > summary::-webkit-details-marker { display:none; }
.fold > summary::after { content:'展开'; flex:none; color:var(--blue); font-size:12px; font-weight:650; }
.fold[open] > summary::after { content:'收起'; }
.figure-fold { margin-bottom:22px; border-color:#c7d2fe; background:#f5f7ff; }
.figure-fold > summary { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:13px 15px; color:#3730a3; font-weight:700; }
.fold-body { border-top:1px solid var(--line); padding:14px; background:#fff; }
.image-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.image-item { margin:0; border:1px solid var(--line); border-radius:10px; background:#fff; padding:7px; }
.image-item img { display:block; width:100%; height:220px; object-fit:contain; background:#f8fafc; border-radius:7px; cursor:zoom-in; }
.image-item figcaption { margin-top:7px; color:var(--muted); font-size:11px; line-height:1.45; }
.timeline { position:relative; padding-left:26px; }
.timeline::before { content:''; position:absolute; left:8px; top:12px; bottom:12px; width:3px; border-radius:3px; background:linear-gradient(#2563eb,#7c3aed,#16a34a); }
.event { position:relative; border:1px solid #dbeafe; border-radius:14px; background:#fff; margin:0 0 15px; box-shadow:0 4px 14px rgba(15,23,42,.05); overflow:visible; }
.event::before { content:''; position:absolute; left:-25px; top:21px; width:13px; height:13px; border:3px solid #fff; border-radius:50%; background:var(--blue); box-shadow:0 0 0 2px var(--blue); }
.event.treatment::before { background:var(--purple); box-shadow:0 0 0 2px var(--purple); }
.event.followup::before { background:var(--green); box-shadow:0 0 0 2px var(--green); }
.event > summary { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; padding:15px 16px; cursor:pointer; list-style:none; }
.event > summary::-webkit-details-marker { display:none; }
.event-main { min-width:0; }
.event-title { font-size:16px; font-weight:700; color:#111827; line-height:1.4; }
.event-index { color:var(--blue); margin-right:7px; }
.event-meta { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
.event-action { color:var(--blue); font-size:12px; font-weight:650; padding-top:3px; white-space:nowrap; }
.event[open] > summary .event-action::before,
.qa[open] > summary .event-action::before { content:'收起'; }
.event:not([open]) > summary .event-action::before,
.qa:not([open]) > summary .event-action::before { content:'展开'; }
.event-body { border-top:1px solid #dbeafe; padding:15px 16px; background:#fbfdff; }
.pill { display:inline-flex; align-items:center; border-radius:999px; padding:2px 8px; font-size:12px; border:1px solid var(--line); background:#fff; color:#475467; }
.pill.blue { color:var(--blue); border-color:#bfdbfe; background:#eff6ff; }
.pill.purple { color:var(--purple); border-color:#ddd6fe; background:#f5f3ff; }
.pill.green { color:var(--green); border-color:#bbf7d0; background:#f0fdf4; }
.pill.amber { color:var(--amber); border-color:#fed7aa; background:#fff7ed; }
.qa { border:1px solid var(--line); border-radius:11px; background:#fff; overflow:hidden; }
.qa + .qa { margin-top:10px; }
.qa > summary { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; padding:12px 13px; cursor:pointer; list-style:none; }
.qa > summary::-webkit-details-marker { display:none; }
.qa-title { font-size:14px; font-weight:650; line-height:1.5; }
.qa-body { border-top:1px solid var(--line); padding:13px; }
.dialogue { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.bubble { border:1px solid var(--line); border-radius:11px; padding:11px 12px; background:#fcfcfd; }
.bubble.answer { border-color:#bfdbfe; background:#eff6ff; }
.bubble-label { margin-bottom:6px; color:var(--muted); font-size:12px; font-weight:650; }
.text { white-space:pre-wrap; font-size:14px; line-height:1.6; word-break:break-word; }
.qa-images { margin-top:12px; }
.qa-images h4 { margin:0 0 8px; font-size:13px; color:#344054; }
@media (max-width:760px) { header,.container{padding-left:16px;padding-right:16px;} .timeline-heading{display:block;} .stats{justify-content:flex-start;margin-top:12px;} .dialogue{grid-template-columns:1fr;} .image-grid{grid-template-columns:1fr;} .event > summary{display:block;} .event-action{margin-top:8px;} }
</style>
</head>
<body>
<header>
  <h1><span id="report-name"></span> 报告结果</h1>
  <div class="subtitle">报告抽取时间线与后续评测结果</div>
</header>
<div class="container">
  <nav class="tabs" aria-label="报告阶段">
    <button class="tab timeline-tab active" data-panel="timeline-panel">抽取时间线</button>
    <button class="tab" data-panel="perception-panel">感知</button>
    <button class="tab" data-panel="diagnosis-panel">诊断</button>
    <button class="tab" data-panel="followup-panel">随访</button>
  </nav>
  <section id="timeline-panel" class="tab-panel active">
    <section class="panel timeline-panel">
      <div class="timeline-heading">
        <div>
          <h2>抽取时间线</h2>
          <div class="small">按报告中的临床时间点展示感知、诊断与随访 QA；每个时间点、QA 对话和图片均可折叠查看。</div>
        </div>
        <div id="stats" class="stats"></div>
      </div>
      <div id="figure-gallery"></div>
      <div id="timeline" class="timeline"></div>
    </section>
  </section>
  <section id="perception-panel" class="tab-panel" aria-label="感知评测"></section>
  <section id="diagnosis-panel" class="tab-panel" aria-label="诊断评测"></section>
  <section id="followup-panel" class="tab-panel" aria-label="随访评测"></section>
</div>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cleanQuestion = (text) => String(text ?? '').replaceAll('<image>', '').trim();
const STAGE_LABELS = {perception:'感知', treatment:'诊断', followup:'随访'};
const ROLE_LABELS = {observation:'观察 QA', evaluation:'评测 QA'};
const stageClass = (type) => type === 'treatment' ? 'purple' : type === 'followup' ? 'green' : 'blue';
const pill = (text, cls='') => `<span class="pill ${cls}">${esc(text)}</span>`;
function imageGrid(paths, caption='') {
  if (!paths.length) return '';
  return `<div class="image-grid">${paths.map((path, index) => `<figure class="image-item"><a href="${esc(path)}" target="_blank" rel="noopener"><img src="${esc(path)}" alt="报告图片 ${index + 1}" loading="lazy" /></a>${caption ? `<figcaption>${esc(caption)}</figcaption>` : ''}</figure>`).join('')}</div>`;
}
function renderFigures() {
  const figures = DATA.figures || [];
  $('figure-gallery').innerHTML = figures.length ? `<details class="fold figure-fold"><summary><span>报告抽取图片（${DATA.stats.images} 张）</span></summary><div class="fold-body"><div class="image-grid">${figures.map(item => item.image_paths.map((path, index) => `<figure class="image-item"><a href="${esc(path)}" target="_blank" rel="noopener"><img src="${esc(path)}" alt="${esc(item.figure)}" loading="lazy" /></a><figcaption><b>${esc(item.figure)}${item.image_paths.length > 1 ? ` · ${index + 1}` : ''}</b><br>${esc(item.caption)}</figcaption></figure>`).join('')).join('')}</div></div></details>` : '';
}
function qaHtml(qa) {
  const images = qa.image_paths || [];
  const imageMeta = images.length ? pill(`图片 ${images.length}`, 'blue') : '';
  return `<details class="qa"><summary><div><div class="qa-title">${esc(cleanQuestion(qa.question))}</div><div class="event-meta">${pill(`QA ${qa.source_turn_id}`,'amber')}${pill(ROLE_LABELS[qa.role] || qa.role)}${imageMeta}</div></div><span class="event-action"></span></summary><div class="qa-body"><div class="dialogue"><div class="bubble"><div class="bubble-label">Q · 问题</div><div class="text">${esc(cleanQuestion(qa.question))}</div></div><div class="bubble answer"><div class="bubble-label">A · 抽取答案</div><div class="text">${esc(qa.answer)}</div></div></div>${images.length ? `<div class="qa-images"><h4>关联图片</h4>${imageGrid(images)}</div>` : ''}</div></details>`;
}
function eventHtml(stage, index) {
  const timepoint = stage.timepoint || {};
  const qas = stage.qa_pairs || [];
  const imageCount = qas.reduce((total, qa) => total + (qa.image_paths || []).length, 0);
  const monthText = Number.isFinite(timepoint.t_months) ? `${timepoint.t_months} months` : '';
  const meta = [pill(STAGE_LABELS[stage.stage_type] || stage.stage_type, stageClass(stage.stage_type)), pill(`QA ${qas.length}`,'amber'), imageCount ? pill(`图片 ${imageCount}`,'blue') : '', ...(stage.modality || []).map(value => pill(value))].join('');
  return `<details class="event ${esc(stage.stage_type)}" ${index === 0 ? 'open' : ''}><summary><div class="event-main"><div class="event-title"><span class="event-index">${String(index + 1).padStart(2,'0')}</span>${monthText ? `<span class="event-index">${esc(monthText)}</span>` : ''}${esc(timepoint.date_text || stage.stage_id)}</div><div class="event-meta">${meta}</div></div><span class="event-action"></span></summary><div class="event-body">${qas.map(qaHtml).join('')}</div></details>`;
}
function setPanel(panelId) {
  document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.panel === panelId));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === panelId));
}
function init() {
  $('report-name').textContent = DATA.patient_name;
  $('stats').innerHTML = `<span class="stat">${DATA.stats.stages} 个时间点</span><span class="stat">${DATA.stats.qa_pairs} 组 QA</span><span class="stat">${DATA.stats.images} 张图片</span>`;
  renderFigures();
  $('timeline').innerHTML = DATA.stages.map(eventHtml).join('');
  document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => setPanel(tab.dataset.panel)));
}
init();
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="生成报告抽取时间线 HTML")
    parser.add_argument("--name", required=True, help="outputs/report 下的报告目录名")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report_dir = BENCH_ROOT / "outputs" / "report" / args.name
    data = collect_report_data(report_dir)
    output = args.output or (report_dir / f"{args.name}_results.html")
    html = HTML_TEMPLATE.replace("__REPORT_NAME__", args.name).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False)
    )
    output.write_text(html, encoding="utf-8")
    print(f"HTML written to: {output}")


if __name__ == "__main__":
    main()
