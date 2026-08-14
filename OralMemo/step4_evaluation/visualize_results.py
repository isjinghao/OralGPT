"""生成 Step4 评测结果的交互式 HTML 可视化。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_perception(eval_root: Path, answer_model: str) -> dict:
    trajectory_root = eval_root.parent / "trajectories" / "model_perception_trajectory"
    standard_path = eval_root.parent / "trajectories" / "standard_trajectory.json"
    model_dir = trajectory_root / answer_model
    model_path = model_dir / "model_perception_trajectory.json"
    report_path = model_dir / "perception_report.json"
    if not model_path.exists():
        return {"profile": [], "items": [], "report": {}, "answer_model": answer_model}

    model = read_json(model_path)
    trajectory_dir = eval_root.parent / "trajectories"
    standard = read_json(standard_path) if standard_path.exists() else {"stages": []}
    report = read_json(report_path) if report_path.exists() else {}
    evaluated_keys = {
        (item["stage_id"], item["source_turn_id"])
        for item in report.get("per_question", [])
    }
    standard_answers = {
        (stage["stage_id"], qa["source_turn_id"]): qa.get("assistant", "")
        for stage in standard.get("stages", [])
        for qa in stage.get("qa_pairs", [])
    }
    profile = []
    items = []
    assets_dir = eval_root / "step4_results_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for stage in sorted(model.get("stages", []), key=lambda item: item.get("order", 0)):
        if stage.get("stage_id") == "S0_PROFILE":
            profile = [
                {"question": qa.get("human", ""), "answer": qa.get("assistant", "")}
                for qa in stage.get("qa_pairs", [])
            ]
            continue
        for qa in stage.get("qa_pairs", []):
            key = (stage["stage_id"], qa["source_turn_id"])
            if key not in evaluated_keys:
                continue
            image_paths = []
            missing_images = []
            for image_index, image_path in enumerate(qa.get("image_paths", []) or [], start=1):
                absolute = BENCH_ROOT / image_path
                if not absolute.is_file():
                    missing_images.append(str(image_path))
                    continue
                destination = assets_dir / (
                    f"{stage['stage_id']}_turn{qa['source_turn_id']}_img{image_index}_{absolute.name}"
                )
                shutil.copy2(absolute, destination)
                image_paths.append(Path(os.path.relpath(destination, eval_root)).as_posix())
            items.append({
                "stage_id": stage["stage_id"],
                "stage_type": stage.get("stage_type", ""),
                "modality": stage.get("modality", []),
                "source_turn_id": qa["source_turn_id"],
                "question": qa.get("human", ""),
                "ground_truth_answer": standard_answers.get(key, ""),
                "model_answer": qa.get("assistant", ""),
                "image_paths": image_paths,
                "missing_images": missing_images,
            })
    items.sort(key=lambda item: item["source_turn_id"])
    return {"profile": profile, "items": items, "report": report, "answer_model": model_dir.name}


def collect_timeline(eval_root: Path) -> list[dict]:
    trajectory_path = eval_root.parent / "trajectories" / "standard_trajectory.json"
    if not trajectory_path.exists():
        return []
    trajectory = read_json(trajectory_path)
    stages = []
    for stage in sorted(trajectory.get("stages", []), key=lambda item: item.get("order", 0)):
        qa_pairs = []
        for qa in stage.get("qa_pairs", []):
            images = []
            for image_path in qa.get("image_paths", []) or []:
                absolute = BENCH_ROOT / image_path
                if absolute.is_file():
                    images.append(Path(os.path.relpath(absolute, eval_root)).as_posix())
            qa_pairs.append({
                "question": qa.get("human", ""),
                "answer": qa.get("assistant", ""),
                "role": qa.get("role", ""),
                "image_paths": images,
            })
        stages.append({
            "stage_id": stage.get("stage_id", ""),
            "stage_type": stage.get("stage_type", ""),
            "modality": stage.get("modality", []),
            "timepoint": stage.get("timepoint", {}),
            "qa_pairs": qa_pairs,
        })
    return stages


def collect_results(eval_root: Path) -> dict:
    trajectories = []
    answer_models = set()
    patient_id = eval_root.parent.name
    for trajectory_root in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        for model_root in sorted(path for path in trajectory_root.iterdir() if path.is_dir()):
            for mode_dir in sorted(path for path in model_root.iterdir() if path.is_dir() and (path / "report.json").exists()):
                report = read_json(mode_dir / "report.json")
                answer_model = report.get("answer_model", model_root.name)
                mode = report.get("mode", mode_dir.name)
                trajectory_type = report.get("trajectory_type", trajectory_root.name)
                answer_models.add(answer_model)
                patient_id = report.get("patient_id", patient_id)
                method_reports = {m["method"]: m for m in report.get("methods", [])}
                rows = []
                for method in sorted(method_reports):
                    answers_path = model_root / method / mode_dir.name / "answers.json"
                    if not answers_path.exists():
                        continue
                    answers = read_json(answers_path)
                    method_report = method_reports.get(method, {})
                    per_task = {x["task_id"]: x for x in method_report.get("per_task", [])}
                    detail_by_task = {}
                    for item in method_report.get("tps", {}).get("per_task", []) or []:
                        detail_by_task[item["task_id"]] = item

                    for ans in answers:
                        task_id = ans["task_id"]
                        score = per_task.get(task_id, {})
                        detail = detail_by_task.get(task_id, {})
                        evidence_judgement = {
                            str(e.get("evidence_id", "")).strip(): e
                            for e in score.get("evidence", []) or []
                        }
                        selected_evidence = []
                        for ev in ans.get("selected_evidence", []) or []:
                            ev = dict(ev)
                            judged = evidence_judgement.get(str(ev.get("evidence_id", "")).strip(), {})
                            if judged:
                                ev["covered"] = bool(judged.get("covered"))
                                ev["coverage_reason"] = judged.get("reason", "")
                            selected_evidence.append(ev)

                        rows.append({
                            "trajectory": trajectory_type,
                            "answer_model": answer_model,
                            "mode": mode,
                            "method": method,
                            "task_id": task_id,
                            "task_type": ans.get("task_type", ""),
                            "ask_after_stage": ans.get("ask_after_stage", ""),
                            "question": ans.get("question", ""),
                            "gold_answer": ans.get("gold_answer", ""),
                            "model_answer": ans.get("model_answer", ""),
                            "memory_context": ans.get("memory_context", ""),
                            "n_images": ans.get("n_images", 0),
                            "score": score,
                            "detail": detail,
                            "selected_evidence": selected_evidence,
                        })
                trajectories.append({
                    "name": f"{trajectory_type}/{answer_model}/{mode}",
                    "trajectory_type": trajectory_type,
                    "answer_model": answer_model,
                    "mode": mode,
                    "report": report,
                    "rows": rows,
                })
    perceptions = {model: collect_perception(eval_root, model) for model in sorted(answer_models)}
    return {
        "patient_id": patient_id,
        "trajectories": trajectories,
        "perceptions": perceptions,
        "timeline": collect_timeline(eval_root),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Step4 评测结果</title>
<style>
:root { --bg:#f6f8fb; --card:#fff; --text:#15202b; --muted:#667085; --line:#e5e7eb; --blue:#2563eb; --green:#16a34a; --red:#dc2626; --amber:#d97706; --purple:#7c3aed; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif; background:var(--bg); color:var(--text); }
header { padding:24px 28px 18px; background:linear-gradient(135deg,#0f172a,#1d4ed8); color:#fff; }
h1 { margin:0 0 8px; font-size:26px; }
.back-link { display:inline-block; margin-top:10px; color:#fff; text-decoration:none; border:1px solid rgba(255,255,255,.55); border-radius:8px; padding:7px 12px; font-size:13px; }
.back-link:hover { background:rgba(255,255,255,.12); }
.container { padding:20px 28px 36px; max-width:1500px; margin:0 auto; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; box-shadow:0 4px 14px rgba(15,23,42,.05); margin-bottom:16px; }
.controls { display:grid; grid-template-columns:repeat(4,minmax(160px,1fr)); gap:12px; align-items:end; }
label { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; }
select,input { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:10px; background:#fff; color:var(--text); }
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.metric-card { border:1px solid var(--line); border-radius:12px; padding:12px; background:#fbfdff; }
.metric-card h3 { margin:0 0 8px; font-size:15px; color:#111827; }
.metric-line { display:flex; justify-content:space-between; gap:8px; font-size:13px; margin:4px 0; }
.metric-line b { color:#111827; }
.pill { display:inline-flex; align-items:center; gap:4px; border-radius:999px; padding:2px 8px; font-size:12px; border:1px solid var(--line); background:#fff; color:#374151; }
.pill.green { color:var(--green); border-color:#bbf7d0; background:#f0fdf4; }
.pill.red { color:var(--red); border-color:#fecaca; background:#fef2f2; }
.pill.blue { color:var(--blue); border-color:#bfdbfe; background:#eff6ff; }
.pill.amber { color:var(--amber); border-color:#fed7aa; background:#fff7ed; }
.pill.purple { color:var(--purple); border-color:#ddd6fe; background:#f5f3ff; }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; margin:12px 0; overflow:hidden; box-shadow:0 3px 10px rgba(15,23,42,.04); }
.card-head { padding:14px 16px; cursor:pointer; display:flex; gap:12px; justify-content:space-between; align-items:flex-start; }
.card-title { font-weight:650; line-height:1.4; }
.card-meta { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
.card-body { display:none; border-top:1px solid var(--line); padding:16px; }
.card.open .card-body { display:block; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.block { border:1px solid var(--line); border-radius:12px; padding:12px; background:#fcfcfd; }
.block h4 { margin:0 0 8px; font-size:14px; color:#344054; }
.text { white-space:pre-wrap; font-size:14px; line-height:1.55; }
.ev-id { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; color:#475467; }
.criteria { width:100%; border-collapse:collapse; font-size:13px; }
.criteria th,.criteria td { border-bottom:1px solid var(--line); padding:7px 6px; text-align:left; vertical-align:top; }
.criteria th { color:#475467; background:#f9fafb; }
.score-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }
.score-box { border:1px solid var(--line); border-radius:10px; background:#fff; padding:10px; }
.score-box .label { color:var(--muted); font-size:12px; margin-bottom:4px; }
.score-box .value { font-weight:700; font-size:18px; }
.score-box .desc { margin-top:6px; color:var(--muted); font-size:12px; line-height:1.45; }
.small { color:var(--muted); font-size:12px; }
.empty { text-align:center; color:var(--muted); padding:32px; }
.tabs { display:flex; gap:8px; margin-bottom:16px; }
.tab { border:1px solid var(--line); border-radius:10px; padding:10px 18px; background:#fff; color:#475467; cursor:pointer; font-size:14px; }
.tab.active { color:#fff; border-color:var(--blue); background:var(--blue); }
.tab-panel { display:none; }
.tab-panel.active { display:block; }
.perception-card { background:var(--card); border:1px solid var(--line); border-radius:14px; margin:12px 0; padding:16px; box-shadow:0 3px 10px rgba(15,23,42,.04); }
.perception-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:12px; }
.profile-list { display:grid; gap:10px; }
.profile-item { border:1px solid var(--line); border-radius:10px; padding:10px 12px; background:#fcfcfd; }
.profile-question { color:var(--muted); font-size:12px; margin-bottom:5px; }
.profile-answer { font-size:14px; line-height:1.5; }
.fold { border:1px solid var(--line); border-radius:10px; background:#fcfcfd; padding:10px 12px; }
.fold + .fold { margin-top:10px; }
.fold summary { cursor:pointer; font-weight:650; color:#344054; }
.fold-body { margin-top:10px; }
.image-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }
.image-item { border:1px solid var(--line); border-radius:10px; background:#fff; padding:6px; }
.image-item img { display:block; width:100%; height:180px; object-fit:contain; background:#f8fafc; border-radius:7px; }
.image-item figcaption { margin-top:5px; color:var(--muted); font-size:11px; word-break:break-all; }
.image-missing { min-height:100px; display:flex; align-items:center; justify-content:center; padding:16px; border:1px dashed #f59e0b; border-radius:8px; background:#fffbeb; color:#92400e; font-size:12px; text-align:center; }
.timeline { border-left:3px solid #93c5fd; margin-left:10px; padding-left:20px; }
.timeline-event { position:relative; margin-bottom:14px; }
.timeline-event::before { content:''; position:absolute; left:-29px; top:18px; width:13px; height:13px; border-radius:50%; background:#2563eb; border:3px solid #fff; box-shadow:0 0 0 2px #2563eb; }
.timeline-event.treatment::before { background:var(--purple); box-shadow:0 0 0 2px var(--purple); }
.timeline-event.followup::before { background:var(--green); box-shadow:0 0 0 2px var(--green); }
.timeline-event.perception { border-color:#bfdbfe; }
.timeline-event.treatment { border-color:#ddd6fe; }
.timeline-event.followup { border-color:#bbf7d0; }
.timeline-event summary { cursor:pointer; font-weight:650; padding:13px 14px; }
.timeline-event .event-body { border-top:1px solid var(--line); padding:14px; }
.timeline-qa { border:1px solid var(--line); border-radius:10px; padding:12px; margin-top:10px; background:#fcfcfd; }
@media (max-width:900px){ .controls,.two-col{grid-template-columns:1fr;} .card-head{display:block;} .perception-head{display:block;} }
</style>
</head>
<body>
<header>
  <h1 id="page-title">Step4 评测结果</h1>
  <a class="back-link" href="../../four_report_summary.html">← 返回四报告汇总</a>
  <a class="back-link" href="../graph/evidence_graph.html">查看交互式证据图</a>
</header>
<div class="container">
  <section class="panel summary-grid">
    <div class="metric-card"><h3>answer model</h3><div id="meta-answer-model"></div></div>
    <div class="metric-card"><h3>mode</h3><div id="meta-mode"></div></div>
    <div class="metric-card"><h3>memory method</h3><div id="meta-memory-method"></div></div>
  </section>
  <nav class="tabs" aria-label="评测阶段">
    <button class="tab active" data-panel="timeline-panel">抽取时间轴</button>
    <button class="tab" data-panel="perception-panel">感知</button>
    <button class="tab" data-panel="treatment-panel">治疗</button>
    <button class="tab" data-panel="followup-panel">随访</button>
  </nav>
  <section class="panel controls">
    <div><label>轨迹</label><select id="traj"></select></div>
    <div><label>记忆方法</label><select id="method"></select></div>
    <div><label>搜索（问题 / 回答 / 证据）</label><input id="search" placeholder="输入关键词..." /></div>
  </section>
  <section id="timeline-panel" class="tab-panel active">
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">报告抽取时间轴</h2><div id="timeline" class="timeline"></div></section>
  </section>
  <section id="perception-panel" class="tab-panel">
    __PROFILE_SECTION__
    <section class="panel"><h2 style="margin:0 0 8px;font-size:18px;">模型感知轨迹</h2></section>
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">感知评估指标</h2><div id="perception-summary" class="summary-grid"></div><div style="height:12px"></div><div id="perception-metrics"></div></section>
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">感知逐题结果</h2><div id="perception-count" class="small"></div><div id="perception-cards"></div></section>
    <section class="panel"><h2 style="margin:0 0 8px;font-size:18px;">感知测评</h2></section>
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">感知测评指标</h2><div id="benchmark-perception-summary" class="summary-grid"></div></section>
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">感知测评逐题结果</h2><div id="benchmark-perception-count" class="small"></div><div id="benchmark-perception-cards"></div></section>
  </section>
  <section id="treatment-panel" class="tab-panel">
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">治疗指标</h2><div id="treatment-summary" class="summary-grid"></div></section>
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">治疗逐题结果</h2><div id="treatment-count" class="small"></div><div id="treatment-cards"></div></section>
  </section>
  <section id="followup-panel" class="tab-panel">
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">随访指标</h2><div id="followup-summary" class="summary-grid"></div></section>
    <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">随访逐题结果</h2><div id="followup-count" class="small"></div><div id="followup-cards"></div></section>
  </section>
</div>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (v) => (typeof v === 'number' ? v.toFixed(1) + '%' : 'n/a');
const ratio = (a,b) => `${a ?? 0}/${b ?? 0}`;
const TRAJECTORY_LABELS = {
  standard_full:'standard_trajectory', standard:'standard_trajectory', standard_trajectory:'standard_trajectory',
  model_perception:'model_perception_trajectory', model_perception_trajectory:'model_perception_trajectory',
  short_noisy:'short_noisy', medium_noisy:'medium_noisy', long_noisy:'long_noisy',
  no_dp:'no_dp', no_dp_trajectory:'no_dp',
  no_xr_ct:'no_xr_ct', no_xr_ct_trajectory:'no_xr_ct'
};
const METHOD_LABELS = {single_stage_memory:'单阶段记忆', full_context_memory:'全上下文记忆', summary_memory:'摘要记忆'};
const TASK_TYPE_LABELS = {
  modality_perception:'模态感知任务',
  longitudinal_evidence_recall:'纵向证据任务',
  cross_modal_reasoning:'跨模态推理任务',
  cross_temporal_reasoning:'跨时间推理任务',
  memory_update_conflict_correction:'记忆更新/冲突纠正任务',
  treatment:'治疗任务',
  followup:'随访任务'
};
const STAGE_LABELS = {
  S0_PROFILE:'基本信息阶段', S1_FP:'面像照片阶段', S2_DP:'口内照片阶段',
  S3_XR_XLA:'X线/头影测量阶段', S4_CT:'三维CT阶段', S5_TMJ:'颞下颌关节阶段', END:'轨迹结束'
};
function mapLabel(map, value){ return map[value] || value || ''; }
function trajLabel(v){ return mapLabel(TRAJECTORY_LABELS, v); }
function methodLabel(v){ return mapLabel(METHOD_LABELS, v); }
function taskTypeLabel(v){ return mapLabel(TASK_TYPE_LABELS, v); }
function stageLabel(v){ return mapLabel(STAGE_LABELS, v); }

function currentTraj(){ return DATA.trajectories.find(t => t.name === $('traj').value) || DATA.trajectories[0]; }
function trajectoryOptionLabel(t){ return `${trajLabel(t.trajectory_type)} · ${t.answer_model} · ${t.mode}`; }
function uniq(arr){ return [...new Set(arr)].filter(Boolean).sort(); }
function fillSelect(sel, values, allLabel, labelFn=(v)=>v){ sel.innerHTML = `<option value="__all__">${allLabel}</option>` + values.map(v=>`<option value="${esc(v)}">${esc(labelFn(v))}</option>`).join(''); }
function pill(text, cls=''){ return `<span class="pill ${cls}">${esc(text)}</span>`; }
function statLine(label, value){ return `<div class="metric-line"><span>${esc(label)}</span><b>${value}</b></div>`; }

function cleanDisplayQuestion(question){
  return String(question ?? '').replaceAll('<image>', '').trim();
}
function renderProfile(profile){
  $('profile').innerHTML = profile.length
    ? profile.map(item => `<div class="profile-item"><div class="profile-question">${esc(cleanDisplayQuestion(item.question))}</div><div class="profile-answer">${esc(item.answer)}</div></div>`).join('')
    : '<div class="empty">没有 profile 信息</div>';
}
function perceptionCardHtml(item){
  const images = item.image_paths || [];
  const missing = item.missing_images || [];
  const figures = images.map((path, index) => `<figure class="image-item"><img src="${esc(path)}" alt="${esc(item.stage_id)} image" loading="lazy" onerror="this.outerHTML='<div class=&quot;image-missing&quot;>图片文件不可用</div>'" /><figcaption>第 ${index + 1} 张图片</figcaption></figure>`);
  figures.push(...missing.map(path => `<div class="image-missing">感知图片缺失：${esc(path)}</div>`));
  const imageHtml = figures.length ? `<div class="image-grid">${figures.join('')}</div>` : '<div class="small">该感知问题没有图片</div>';
  return `<article class="perception-card">
    <div class="perception-head">
      <div>
        <div class="card-title">${esc(cleanDisplayQuestion(item.question))}</div>
        <div class="card-meta">${pill(stageLabel(item.stage_id),'amber')}${pill((item.modality || []).join(', '),'blue')}${pill(`source turn ${item.source_turn_id}`,'purple')}</div>
      </div>
      <div class="small">${esc(item.stage_type)}</div>
    </div>
    <details class="fold">
      <summary>查看图片（${images.length + missing.length}）</summary>
      <div class="fold-body">${imageHtml}</div>
    </details>
    <div style="height:10px"></div>
    <details class="fold">
      <summary>查看回答</summary>
      <div class="fold-body">
        <div class="two-col">
          <div class="block"><h4>标准参考答案</h4><div class="text">${esc(item.ground_truth_answer)}</div></div>
          <div class="block"><h4>模型感知答案</h4><div class="text">${esc(item.model_answer)}</div></div>
        </div>
      </div>
    </details>
  </article>`;
}
function renderPerceptionMetrics(report){
  const overall = report.overall || {};
  const cards = [
    ['Precision', overall.precision],
    ['Recall', overall.recall],
    ['F1', overall.f1],
    ['幻觉控制', overall.hallucination_control]
  ];
  $('perception-summary').innerHTML = cards.map(([label, value]) => `<div class="metric-card"><h3>${esc(label)}</h3><div class="value" style="font-size:24px;font-weight:700;">${pct(typeof value === 'number' ? value * 100 : undefined)}</div></div>`).join('');
  const rows = [...(report.per_question || [])].sort((a, b) => Number(a.source_turn_id) - Number(b.source_turn_id));
  $('perception-metrics').innerHTML = rows.length
    ? `<table class="criteria"><thead><tr><th>source turn</th><th>阶段</th><th>Precision</th><th>Recall</th><th>F1</th><th>幻觉控制</th><th>命中 / Gold</th><th>模型 claims</th></tr></thead><tbody>${rows.map(row => {
      const m = row.metrics || {};
      return `<tr><td>${esc(row.source_turn_id)}</td><td>${esc(stageLabel(row.stage_id))}</td><td>${pct(typeof m.precision === 'number' ? m.precision * 100 : undefined)}</td><td>${pct(typeof m.recall === 'number' ? m.recall * 100 : undefined)}</td><td>${pct(typeof m.f1 === 'number' ? m.f1 * 100 : undefined)}</td><td>${pct(typeof m.hallucination_control === 'number' ? m.hallucination_control * 100 : undefined)}</td><td>${esc(m.matched_evidence_count ?? 0)} / ${esc(m.gold_evidence_count ?? 0)}</td><td>${esc(m.predicted_claim_count ?? 0)}</td></tr>`;
    }).join('')}</tbody></table>`
    : '<div class="empty">没有感知评估报告</div>';
}
function renderPerception(){
  const t = currentTraj();
  const perception = DATA.perceptions[t.answer_model] || {profile: [], items: [], report: {}};
  const items = perception.items || [];
  if($('profile')) renderProfile(perception.profile || []);
  renderPerceptionMetrics(perception.report || {});
  $('perception-count').textContent = `共 ${items.length} 个感知问题（按 source turn 顺序）`;
  $('perception-cards').innerHTML = items.length ? items.map(perceptionCardHtml).join('') : '<div class="empty">没有找到模型感知轨迹</div>';
}
function timelineImages(paths){
  return paths.length ? `<div class="image-grid">${paths.map(path => `<figure class="image-item"><a href="${esc(path)}" target="_blank"><img src="${esc(path)}" loading="lazy" /></a></figure>`).join('')}</div>` : '';
}
function renderTimeline(){
  const stages = DATA.timeline || [];
  $('timeline').innerHTML = stages.length ? stages.map((stage, index) => {
    const tp = stage.timepoint || {};
    const title = `${String(index + 1).padStart(2,'0')} · ${tp.date_text || stage.stage_id}${Number.isFinite(tp.t_months) ? ` · ${tp.t_months} months` : ''}`;
    const qas = (stage.qa_pairs || []).map(qa => `<div class="timeline-qa"><div class="card-meta">${pill(qa.role || 'QA','amber')}${pill((qa.image_paths || []).length ? 'FIGURE_QA' : 'TEXT_QA','blue')}</div><div class="text"><b>Q:</b> ${esc(cleanDisplayQuestion(qa.question))}</div><div class="text" style="margin-top:8px"><b>A:</b> ${esc(qa.answer)}</div>${timelineImages(qa.image_paths || [])}</div>`).join('');
    const stageColor = stage.stage_type === 'treatment' ? 'purple' : stage.stage_type === 'followup' ? 'green' : 'blue';
    return `<details class="card timeline-event ${esc(stage.stage_type)}" ${index === 0 ? 'open' : ''}><summary>${esc(title)} <span class="card-meta">${pill(stage.stage_type,stageColor)}${(stage.modality || []).map(m => pill(m,'blue')).join('')}</span></summary><div class="event-body">${qas}</div></details>`;
  }).join('') : '<div class="empty">没有抽取时间轴</div>';
}
function setPanel(panelId){
  document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.panel === panelId));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === panelId));
}
function renderConfiguration(){
  const t = currentTraj();
  $('meta-answer-model').textContent = t.answer_model;
  $('meta-mode').textContent = t.mode;
  $('meta-memory-method').textContent = $('method').value === '__all__' ? 'all' : $('method').value;
}
function init(){
  document.title = `${DATA.patient_id} 评测结果`;
  $('page-title').textContent = document.title;
  $('traj').innerHTML = DATA.trajectories.map(t=>`<option value="${esc(t.name)}">${esc(trajectoryOptionLabel(t))}</option>`).join('');
  updateFilters();
  document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => setPanel(tab.dataset.panel)));
  ['traj','method','search'].forEach(id => $(id).addEventListener(id==='traj'?'change':'input', () => {
    if(id==='traj') updateFilters();
    renderConfiguration();
    render();
  }));
  renderConfiguration();
  renderTimeline();
  renderPerception();
  render();
}
function updateFilters(){
  const t = currentTraj();
  fillSelect($('method'), uniq(t.rows.map(r=>r.method)), '全部方法', methodLabel);
}
function filteredRows(){
  const t = currentTraj();
  const method = $('method').value;
  const q = $('search').value.trim().toLowerCase();
  return t.rows.filter(r => {
    if(method !== '__all__' && r.method !== method) return false;
    if(q){
      const hay = [r.task_id,r.question,r.gold_answer,r.model_answer, ...(r.selected_evidence||[]).map(e=>`${e.evidence_id} ${e.fact_text}`)].join(' ').toLowerCase();
      if(!hay.includes(q)) return false;
    }
    return true;
  });
}
function rowsForGroup(group){
  return filteredRows().filter(r => group === 'perception'
    ? !['treatment','followup'].includes(r.task_type)
    : r.task_type === group);
}
function groupMetrics(rows, group){
  const covered = rows.reduce((n,r)=>n+(r.score?.covered_evidence_count||0),0);
  const totalEvidence = rows.reduce((n,r)=>n+(r.score?.total_evidence_count||0),0);
  const ers = totalEvidence ? covered / totalEvidence * 100 : undefined;
  if(group === 'perception'){
    const judged = rows.filter(r=>String(r.score?.metric||'').includes('ACC'));
    const correct = judged.filter(r=>r.score?.correct).length;
    return [['准确率（ACC）', judged.length ? correct/judged.length*100 : undefined, `${correct}/${judged.length}`],['证据召回分数（ERS）',ers,`${covered}/${totalEvidence}`]];
  }
  const values = rows.map(r=>r.detail?.percent ?? r.score?.percent).filter(v=>typeof v==='number');
  const rubric = values.length ? values.reduce((a,b)=>a+b,0)/values.length : undefined;
  return [[group==='treatment'?'治疗分':'随访分',rubric,`${values.length} 题`],['证据召回分数（ERS）',ers,`${covered}/${totalEvidence}`]];
}
function renderGroup(group, target=group){
  const rows = rowsForGroup(group);
  const summary = groupMetrics(rows,group);
  $(`${target}-summary`).innerHTML = summary.map(([label,value,detail])=>`<div class="metric-card"><h3>${esc(label)}</h3><div class="value" style="font-size:24px;font-weight:700">${pct(value)}</div><div class="small">${esc(detail)}</div></div>`).join('');
  $(`${target}-count`).textContent = `当前显示 ${rows.length} 条任务记录`;
  $(`${target}-cards`).innerHTML = rows.length ? rows.map(cardHtml).join('') : '<div class="empty">没有匹配结果</div>';
}
function scorePills(r){
  const s = r.score || {};
  const d = r.detail || {};
  let out = [
    pill(trajLabel(r.trajectory),'blue'),
    pill(`answer model: ${r.answer_model}`),
    pill(`mode: ${r.mode}`,'blue'),
    pill(`memory method: ${r.method}`,'purple'),
    pill(taskTypeLabel(r.task_type)),
    pill(stageLabel(r.ask_after_stage || 'END'),'amber')
  ];
  if(s.metric && String(s.metric).includes('ACC')){
    out.push(pill(`准确率（ACC）${s.correct ? '正确' : '错误'}`, s.correct ? 'green' : 'red'));
    out.push(pill(`证据召回分数（ERS）${s.covered_evidence_count ?? 0}/${s.total_evidence_count ?? 0}`, 'blue'));
  } else if(d.percent !== undefined || s.percent !== undefined){
    out.push(pill(`Rubric得分 ${pct(d.percent ?? s.percent)}`, 'blue'));
    if(s.total_evidence_count !== undefined){
      out.push(pill(`证据召回分数（ERS）${s.covered_evidence_count ?? 0}/${s.total_evidence_count ?? 0}`, 'blue'));
    }
  }
  if(r.n_images) out.push(pill(`图片 ${r.n_images}`,'blue'));
  return out.join('');
}
function scoreSummaryHtml(r){
  const s = r.score || {};
  const d = r.detail || {};
  if(s.metric && String(s.metric).includes('ACC')){
    const accText = s.correct ? '正确' : '错误';
    const accCls = s.correct ? 'green' : 'red';
    const ersPct = pct(s.ers_score);
    return `<div class="block"><h4>单题评分</h4>
      <div class="score-grid">
        <div class="score-box"><div class="label">准确率（ACC，整题正确性）</div><div class="value">${pill(accText, accCls)}</div><div class="desc">判定理由：${esc(s.reason || '')}</div></div>
        <div class="score-box"><div class="label">证据召回分数（ERS）</div><div class="value">${ersPct}</div><div class="desc">正确召回证据数 / selected_evidence 总数：${esc(s.covered_evidence_count ?? 0)} / ${esc(s.total_evidence_count ?? 0)}</div></div>
      </div>
    </div>`;
  }
  const percent = d.percent ?? s.percent;
  const awarded = d.awarded ?? s.awarded;
  const maxTotal = d.max_total ?? s.max_total;
  const ersBox = (s.total_evidence_count !== undefined)
    ? `<div class="score-box"><div class="label">证据召回分数（ERS）</div><div class="value">${pct(s.ers_score)}</div><div class="desc">正确召回证据数 / selected_evidence 总数：${esc(s.covered_evidence_count ?? 0)} / ${esc(s.total_evidence_count ?? 0)}</div></div>`
    : '';
  return `<div class="block"><h4>单题评分</h4>
    <div class="score-grid">
      <div class="score-box"><div class="label">Rubric 总分</div><div class="value">${pct(percent)}</div><div class="desc">得分 / 满分：${esc(awarded ?? 'n/a')} / ${esc(maxTotal ?? 'n/a')}</div></div>
      <div class="score-box"><div class="label">评分类型</div><div class="value">${esc(s.metric || taskTypeLabel(r.task_type) || 'rubric')}</div><div class="desc">诊断或治疗任务按 rubric 逐项计分</div></div>
      ${ersBox}
    </div>
  </div>`;
}
function evidenceHtml(r){
  const evs = r.selected_evidence || [];
  if(!evs.length) return '<div class="small">无 selected_evidence</div>';
  const covered = evs.filter(e => e.covered === true).length;
  const judged = evs.some(e => Object.prototype.hasOwnProperty.call(e, 'covered'));
  const summary = judged ? `本题证据召回：${covered}/${evs.length}` : `本题 selected_evidence：${evs.length} 条`;
  return `<div class="small" style="margin-bottom:8px;">${summary}</div>
  <table class="criteria"><thead><tr><th style="width:84px;">召回情况</th><th style="width:220px;">证据ID</th><th>证据事实</th><th style="width:110px;">阶段 / 模态</th></tr></thead><tbody>${evs.map(e => {
    const hasJudge = Object.prototype.hasOwnProperty.call(e, 'covered');
    const status = hasJudge ? (e.covered ? pill('已召回','green') : pill('未召回','red')) : pill('未判定');
    return `<tr>
      <td>${status}</td>
      <td><div class="ev-id">${esc(e.evidence_id)}</div></td>
      <td>${esc(e.fact_text)}${e.value !== undefined && e.value !== null ? `<div class="small">取值: ${esc(e.value)} ${esc(e.unit || '')}</div>` : ''}</td>
      <td>${pill(stageLabel(e.stage || ''))}<br/>${pill((e.modality||[]).join(','))}</td>
    </tr>`;
  }).join('')}</tbody></table>`;
}
function criteriaHtml(r){
  const detail = r.detail || {};
  const criteria = detail.criteria || [];
  if(!criteria.length) return '';
  return `<div class="block"><h4>Rubric 细项评分</h4>
    <div class="small" style="margin-bottom:8px;">总分：${esc(detail.awarded ?? 'n/a')} / ${esc(detail.max_total ?? 'n/a')}，百分比：${pct(detail.percent)}</div>
    <table class="criteria"><thead><tr><th>评分项</th><th style="width:90px;">得分</th><th style="width:70px;">满分</th><th style="width:80px;">百分比</th><th>判定理由</th></tr></thead><tbody>${criteria.map(c=>{
      const p = (typeof c.awarded === 'number' && typeof c.max === 'number' && c.max) ? c.awarded / c.max * 100 : undefined;
      return `<tr><td>${esc(c.name)}</td><td>${esc(c.awarded)}</td><td>${esc(c.max)}</td><td>${pct(p)}</td><td>${esc(c.reason || '')}</td></tr>`;
    }).join('')}</tbody></table></div>`;
}
function rawScoreHtml(r){
  return `<details class="block"><summary style="cursor:pointer;font-weight:650;">原始评分 JSON</summary><pre class="text">${esc(JSON.stringify({score:r.score || {}, rubric_detail:r.detail || {}}, null, 2))}</pre></details>`;
}
function cardHtml(r){
  return `<article class="card">
    <div class="card-head" onclick="this.parentElement.classList.toggle('open')">
      <div><div class="card-title">${esc(r.question)}</div><div class="card-meta">${scorePills(r)}</div></div>
      <div class="small">${esc(r.task_id)}</div>
    </div>
    <div class="card-body">
      ${scoreSummaryHtml(r)}
      <div style="height:12px"></div>
      <div class="two-col">
        <div class="block"><h4>标准答案</h4><div class="text">${esc(r.gold_answer)}</div></div>
        <div class="block"><h4>模型答案</h4><div class="text">${esc(r.model_answer)}</div></div>
      </div>
      <div style="height:12px"></div>
      <div class="block"><h4>证据列表 / 证据召回分数（ERS）逐证据判定</h4>${evidenceHtml(r)}</div>
      <div style="height:12px"></div>
      ${criteriaHtml(r)}
      ${rawScoreHtml(r)}
    </div>
  </article>`;
}
function render(){
  renderGroup('perception','benchmark-perception');
  renderGroup('treatment');
  renderGroup('followup');
}
init();
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Step4 评测结果 HTML 可视化")
    parser.add_argument("--eval-root", type=Path, default=BENCH_ROOT / "outputs" / "group1" / "CHENFANG" / "evaluation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = collect_results(args.eval_root)
    output = args.output or (args.eval_root / f"{args.eval_root.parent.name.lower()}_step4_results.html")
    profile_section = "" if args.eval_root.parent.parent.name == "report" else (
        '<section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">患者 Profile</h2>'
        '<div id="profile" class="profile-list"></div></section>'
    )
    html = HTML_TEMPLATE.replace("__PROFILE_SECTION__", profile_section)
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    output.write_text(html, encoding="utf-8")
    patient_id = f"{args.eval_root.parent.parent.name}__{args.eval_root.parent.name}"
    print(f"[evaluation][{patient_id}][step4/html] written path={output}", flush=True)


if __name__ == "__main__":
    main()
