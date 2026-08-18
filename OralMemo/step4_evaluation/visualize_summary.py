"""生成多个病人的感知与 Step4 评估汇总 HTML。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from utils.json_utils import read_json

PATIENTS = ["CHENFANG", "CHENFENGQI", "CHENGYAO", "CHENJINGYUAN"]
TRAJECTORIES = ["standard_trajectory", "model_perception_trajectory"]


def method_report(patient_root: Path, trajectory: str, model: str, method: str) -> dict:
    report = read_json(patient_root / "evaluation" / trajectory / model / "text" / "report.json")
    return next(item for item in report["methods"] if item["method"] == method)


def collect(bench_root: Path, patients: list[str], model: str, method: str) -> dict:
    rows = []
    for patient in patients:
        root = bench_root / patient
        perception = read_json(
            root / "trajectories" / "model_perception_trajectory" / model / "perception_report.json"
        )["overall"]
        evaluations = {
            trajectory: method_report(root, trajectory, model, method)
            for trajectory in TRAJECTORIES
        }
        rows.append({
            "patient": patient,
            "href": f"{patient}/evaluation/{patient.lower()}_results.html",
            "perception": perception,
            "evaluations": evaluations,
        })

    averages = {
        "perception": {
            key: mean(row["perception"][key] for row in rows)
            for key in ("precision", "recall", "f1", "hallucination_control")
        },
        "evaluations": {
            trajectory: {
                "acc": mean(row["evaluations"][trajectory]["acc"]["overall"]["score"] for row in rows),
                "ers": mean(row["evaluations"][trajectory]["ers"]["overall"]["score"] for row in rows),
                "tps": mean(row["evaluations"][trajectory]["tps"]["overall_percent"] for row in rows),
            }
            for trajectory in TRAJECTORIES
        },
    }
    return {"model": model, "method": method, "rows": rows, "averages": averages}


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OralMemo 四病例结果汇总</title>
<style>
:root{--bg:#f4f7fb;--card:#fff;--text:#15202b;--muted:#667085;--line:#dfe5ee;--blue:#2563eb;--navy:#0f172a}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif}
header{padding:30px 32px;background:linear-gradient(135deg,var(--navy),#1d4ed8);color:#fff} h1{margin:0 0 8px}.sub{opacity:.82}
main{max-width:1400px;margin:auto;padding:24px 28px 44px}.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:18px;box-shadow:0 4px 14px rgba(15,23,42,.05)}
.cards{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}.card{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fbfdff;text-align:center}.value{font-size:25px;font-weight:750;margin-top:8px}.label{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px 9px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}th{background:#f8fafc;color:#475467}.group{background:#eef4ff;font-weight:700}.patient-link{color:var(--blue);font-weight:700;text-decoration:none}.patient-link:hover{text-decoration:underline}.hint{font-size:13px;color:var(--muted);line-height:1.6}.ok{color:#15803d}
@media(max-width:1100px){.cards{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:900px){main{padding:16px}.table-wrap{overflow:auto}}
</style></head><body>
<header><h1>OralMemo 四病例结果汇总</h1><div class="sub">模型：__MODEL__ · 记忆方法：full_context_memory · 评估模式：text</div></header>
<main>
<section class="panel"><h2>四病例平均分</h2><div id="cards" class="cards"></div></section>
<section class="panel"><h2>逐病例结果</h2><div class="hint">点击病人姓名进入该病人的综合页面，可查看感知问题、标准参考、模型感知答案，以及标准轨迹和模型感知轨迹的完整逐题评估。</div><div class="table-wrap"><table id="results"></table></div></section>
</main><script>
const DATA=__DATA__; const pct=v=>typeof v==='number'?(v*100).toFixed(1)+'%':'n/a'; const score=v=>typeof v==='number'?v.toFixed(1)+'%':'n/a';
const p=DATA.averages.perception,s=DATA.averages.evaluations.standard_trajectory,m=DATA.averages.evaluations.model_perception_trajectory;
const cards=[['感知 F1',pct(p.f1)],['感知 Precision',pct(p.precision)],['感知 Recall',pct(p.recall)],['幻觉控制',pct(p.hallucination_control)],['标准轨迹治疗分',score(s.tps)],['模型感知轨迹治疗分',score(m.tps)]];
document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('');
const head=`<thead><tr><th rowspan="2">病人</th><th colspan="4">感知</th><th colspan="3">标准轨迹</th><th colspan="3">模型感知轨迹</th></tr><tr><th>Precision</th><th>Recall</th><th>F1</th><th>幻觉控制</th><th>ACC</th><th>ERS</th><th>治疗分</th><th>ACC</th><th>ERS</th><th>治疗分</th></tr></thead>`;
const body=DATA.rows.map(r=>{const p=r.perception,s=r.evaluations.standard_trajectory,m=r.evaluations.model_perception_trajectory;return `<tr><td><a class="patient-link" href="${r.href}">${r.patient}</a></td><td>${pct(p.precision)}</td><td>${pct(p.recall)}</td><td>${pct(p.f1)}</td><td>${pct(p.hallucination_control)}</td><td>${score(s.acc.overall.score)}</td><td>${score(s.ers.overall.score)}</td><td>${score(s.tps.overall_percent)}</td><td>${score(m.acc.overall.score)}</td><td>${score(m.ers.overall.score)}</td><td>${score(m.tps.overall_percent)}</td></tr>`}).join('');
const avg=`<tr class="group"><td>四病例平均</td><td>${pct(p.precision)}</td><td>${pct(p.recall)}</td><td>${pct(p.f1)}</td><td>${pct(p.hallucination_control)}</td><td>${score(s.acc)}</td><td>${score(s.ers)}</td><td>${score(s.tps)}</td><td>${score(m.acc)}</td><td>${score(m.ers)}</td><td>${score(m.tps)}</td></tr>`;
document.getElementById('results').innerHTML=head+`<tbody>${body}${avg}</tbody>`;
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="生成四病人感知与评估汇总 HTML")
    parser.add_argument("--bench-root", type=Path, default=Path("outputs/group1"))
    parser.add_argument("--patients", nargs="+", default=PATIENTS)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--method", default="full_context_memory")
    parser.add_argument("--output", type=Path, default=Path("outputs/group1/four_patient_summary.html"))
    args = parser.parse_args()
    data = collect(args.bench_root, args.patients, args.model, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        HTML.replace("__MODEL__", args.model).replace("__DATA__", json.dumps(data, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"[evaluation][summary/html] written path={args.output}", flush=True)


if __name__ == "__main__":
    main()
