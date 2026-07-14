"""生成 CHENFANG 评测结果的交互式 HTML 可视化。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_results(eval_root: Path) -> dict:
    trajectories = []
    for traj_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        report_path = traj_dir / "report.json"
        if not report_path.exists():
            continue
        report = read_json(report_path)
        method_reports = {m["method"]: m for m in report.get("methods", [])}
        rows = []
        for answers_path in sorted(traj_dir.glob("answers_*.json")):
            method = answers_path.stem.removeprefix("answers_")
            answers = read_json(answers_path)
            method_report = method_reports.get(method, {})
            per_task = {x["task_id"]: x for x in method_report.get("per_task", [])}
            detail_by_task = {}
            if method_report.get("diagnosis"):
                detail_by_task[method_report["diagnosis"]["task_id"]] = method_report["diagnosis"]
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
                    "trajectory": traj_dir.name,
                    "method": method,
                    "task_id": task_id,
                    "task_type": ans.get("task_type", ""),
                    "ask_after_stage": ans.get("ask_after_stage", ""),
                    "question": ans.get("question", ""),
                    "gold_answer": ans.get("gold_answer", ""),
                    "model_answer": ans.get("model_answer", ""),
                    "validation_accepted": ans.get("validation_accepted", True),
                    "n_images": ans.get("n_images", 0),
                    "score": score,
                    "detail": detail,
                    "selected_evidence": selected_evidence,
                })
        trajectories.append({
            "name": traj_dir.name,
            "report": report,
            "rows": rows,
        })
    return {"trajectories": trajectories}


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CHENFANG 评测结果</title>
<style>
:root { --bg:#f6f8fb; --card:#fff; --text:#15202b; --muted:#667085; --line:#e5e7eb; --blue:#2563eb; --green:#16a34a; --red:#dc2626; --amber:#d97706; --purple:#7c3aed; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif; background:var(--bg); color:var(--text); }
header { padding:24px 28px 18px; background:linear-gradient(135deg,#0f172a,#1d4ed8); color:#fff; }
h1 { margin:0 0 8px; font-size:26px; }
.subtitle { opacity:.88; font-size:14px; }
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
.evidence { display:grid; gap:8px; }
.ev { border:1px solid var(--line); border-radius:10px; padding:9px; background:#fff; }
.ev.covered { border-color:#86efac; background:#f0fdf4; }
.ev.missed { border-color:#fecaca; background:#fff7f7; }
.ev-id { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; color:#475467; }
.ev-fact { margin-top:4px; font-size:14px; }
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
@media (max-width:900px){ .controls,.two-col{grid-template-columns:1fr;} .card-head{display:block;} }
</style>
</head>
<body>
<header>
  <h1>CHENFANG 评测结果</h1>
</header>
<div class="container">
  <section class="panel controls">
    <div><label>轨迹</label><select id="traj"></select></div>
    <div><label>记忆方法</label><select id="method"></select></div>
    <div><label>问题类型</label><select id="type"></select></div>
    <div><label>搜索（问题 / 回答 / 证据）</label><input id="search" placeholder="输入关键词..." /></div>
  </section>
  <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">总体指标</h2><div id="summary" class="summary-grid"></div></section>
  <section class="panel"><h2 style="margin:0 0 12px;font-size:18px;">逐题结果</h2><div id="count" class="small"></div><div id="cards"></div></section>
</div>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (v) => (typeof v === 'number' ? v.toFixed(1) + '%' : 'n/a');
const ratio = (a,b) => `${a ?? 0}/${b ?? 0}`;
const TRAJECTORY_LABELS = {standard_full:'标准轨迹', standard:'标准轨迹', long_noisy:'长噪声轨迹'};
const METHOD_LABELS = {single_stage_memory:'单阶段记忆', full_context_memory:'全上下文记忆', summary_memory:'摘要记忆'};
const TASK_TYPE_LABELS = {
  modality_perception:'模态感知任务',
  longitudinal_evidence_recall:'纵向证据任务',
  cross_modal_reasoning:'跨模态推理任务',
  memory_update_conflict_correction:'记忆更新/冲突纠正任务',
  heldout_diagnosis:'诊断任务',
  heldout_treatment:'治疗任务'
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
function uniq(arr){ return [...new Set(arr)].filter(Boolean).sort(); }
function fillSelect(sel, values, allLabel, labelFn=(v)=>v){ sel.innerHTML = `<option value="__all__">${allLabel}</option>` + values.map(v=>`<option value="${esc(v)}">${esc(labelFn(v))}</option>`).join(''); }
function pill(text, cls=''){ return `<span class="pill ${cls}">${esc(text)}</span>`; }
function statLine(label, value){ return `<div class="metric-line"><span>${esc(label)}</span><b>${value}</b></div>`; }

function init(){
  $('traj').innerHTML = DATA.trajectories.map(t=>`<option value="${esc(t.name)}">${esc(trajLabel(t.name))}</option>`).join('');
  updateFilters();
  ['traj','method','type','search'].forEach(id => $(id).addEventListener(id==='traj'?'change':'input', () => { if(id==='traj') updateFilters(); render(); }));
  render();
}
function updateFilters(){
  const t = currentTraj();
  fillSelect($('method'), uniq(t.rows.map(r=>r.method)), '全部方法', methodLabel);
  fillSelect($('type'), uniq(t.rows.map(r=>r.task_type)), '全部类型', taskTypeLabel);
}
function renderSummary(){
  const t = currentTraj();
  const methods = t.report.methods || [];
  $('summary').innerHTML = methods.map(m => {
    const acc = m.acc?.overall || {};
    const ers = m.ers?.overall || {};
    const diag = m.diagnosis?.percent;
    const treatment = m.tps?.overall_percent;
    return `<div class="metric-card">
      <h3>${esc(methodLabel(m.method))}</h3>
      ${statLine('准确率（ACC）', `${pct(acc.score)} (${ratio(acc.correct, acc.total)})`)}
      ${statLine('证据召回分数（ERS）', `${pct(ers.score)} (${ratio(ers.covered, ers.total)})`)}
      ${statLine('诊断分', pct(diag))}
      ${statLine('治疗分', pct(treatment))}
    </div>`;
  }).join('');
}
function filteredRows(){
  const t = currentTraj();
  const method = $('method').value;
  const type = $('type').value;
  const q = $('search').value.trim().toLowerCase();
  return t.rows.filter(r => {
    if(method !== '__all__' && r.method !== method) return false;
    if(type !== '__all__' && r.task_type !== type) return false;
    if(q){
      const hay = [r.task_id,r.question,r.gold_answer,r.model_answer, ...(r.selected_evidence||[]).map(e=>`${e.evidence_id} ${e.fact_text}`)].join(' ').toLowerCase();
      if(!hay.includes(q)) return false;
    }
    return true;
  });
}
function scorePills(r){
  const s = r.score || {};
  const d = r.detail || {};
  let out = [pill(trajLabel(r.trajectory),'blue'), pill(methodLabel(r.method),'purple'), pill(taskTypeLabel(r.task_type)), pill(stageLabel(r.ask_after_stage || 'END'),'amber')];
  if(s.metric && String(s.metric).includes('ACC')){
    out.push(pill(`准确率（ACC）${s.correct ? '正确' : '错误'}`, s.correct ? 'green' : 'red'));
    out.push(pill(`证据召回分数（ERS）${s.covered_evidence_count ?? 0}/${s.total_evidence_count ?? 0}`, 'blue'));
  } else if(d.percent !== undefined || s.percent !== undefined){
    out.push(pill(`Rubric得分 ${pct(d.percent ?? s.percent)}`, 'blue'));
    if(s.total_evidence_count !== undefined){
      out.push(pill(`证据召回分数（ERS）${s.covered_evidence_count ?? 0}/${s.total_evidence_count ?? 0}`, 'blue'));
    }
  }
  if(r.validation_accepted === false) out.push(pill('校验未通过','red'));
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
        <div class="score-box"><div class="label">问题校验</div><div class="value">${pill(r.validation_accepted === false ? '未通过' : '通过', r.validation_accepted === false ? 'red' : 'green')}</div><div class="desc">Step3 问题校验结果</div></div>
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
  return `<div class="small" style="margin-bottom:8px;">${judged ? `本题证据召回：${covered}/${evs.length}` : `本题 selected_evidence：${evs.length} 条`}</div>
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
function cardHtml(r, idx){
  return `<article class="card" data-i="${idx}">
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
  renderSummary();
  const rows = filteredRows();
  $('count').textContent = `当前显示 ${rows.length} 条 记忆方法-任务 记录`;
  $('cards').innerHTML = rows.length ? rows.map(cardHtml).join('') : '<div class="empty">没有匹配结果</div>';
}
init();
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CHENFANG 评测结果 HTML 可视化")
    parser.add_argument("--eval-root", type=Path, default=BENCH_ROOT / "outputs" / "group1" / "CHENFANG" / "evaluation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = collect_results(args.eval_root)
    output = args.output or (args.eval_root / "chenfang_step4_results.html")
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    output.write_text(html, encoding="utf-8")
    print(f"HTML written to: {output}")


if __name__ == "__main__":
    main()
