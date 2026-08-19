"""生成多个 Report 的模型感知与 Step4 评估汇总 HTML。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from utils.json_utils import read_json

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = ["standard_trajectory", "model_perception_trajectory"]


def read_method(report_root: Path, report: str, trajectory: str, model: str, method: str) -> dict:
    data = read_json(report_root / report / "evaluation" / trajectory / model / "report.json")
    return next(item for item in data["methods"] if item["method"] == method)


def collect(report_root: Path, reports: list[str], model: str, method: str) -> dict:
    rows = []
    for report in reports:
        perception = read_json(
            report_root / report / "trajectories" / "model_perception_trajectory" / model / "perception_report.json"
        )["overall"]
        evaluations = {
            trajectory: read_method(report_root, report, trajectory, model, method)
            for trajectory in TRAJECTORIES
        }
        rows.append({
            "report": report,
            "href": f"{report}/evaluation/{report.lower()}_results.html",
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
                "treatment": mean(row["evaluations"][trajectory]["tps"]["overall_percent"] for row in rows),
                "followup": mean(row["evaluations"][trajectory]["followup"]["overall_percent"] for row in rows),
            }
            for trajectory in TRAJECTORIES
        },
    }
    return {"model": model, "method": method, "rows": rows, "averages": averages}


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OralMemo 四 Report 结果汇总</title>
<style>
:root{--bg:#f4f7fb;--card:#fff;--text:#15202b;--muted:#667085;--line:#dfe5ee;--blue:#2563eb;--navy:#0f172a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif}
header{padding:30px 32px;background:linear-gradient(135deg,var(--navy),#1d4ed8);color:#fff}h1{margin:0 0 8px}.sub{opacity:.82}
main{max-width:1400px;margin:auto;padding:24px 28px 44px}.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:18px;box-shadow:0 4px 14px rgba(15,23,42,.05)}
.cards{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:12px}.card{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fbfdff;text-align:center}.value{font-size:25px;font-weight:750;margin-top:8px}.label{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px 9px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}th{background:#f8fafc;color:#475467}.group{background:#eef4ff;font-weight:700}.report-link{color:var(--blue);font-weight:700;text-decoration:none}.report-link:hover{text-decoration:underline}.hint{font-size:13px;color:var(--muted);line-height:1.7}.formula{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:2px 6px}
@media(max-width:1200px){.cards{grid-template-columns:repeat(4,minmax(0,1fr))}}@media(max-width:900px){main{padding:16px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.table-wrap{overflow:auto}}
</style></head><body>
<header><h1>OralMemo 四 Report 结果汇总</h1><div class="sub">模型：__MODEL__ · 记忆方法：full_context_memory</div></header>
<main>
<section class="panel"><h2>四 Report 平均分</h2><div id="cards" class="cards"></div></section>
<section class="panel"><h2>逐 Report 结果</h2><div class="hint">点击 Report 名称进入详情页，可查看模型感知问题、标准参考与模型感知答案，并切换标准轨迹和模型感知轨迹查看治疗、随访及完整逐题评估。</div><div class="table-wrap"><table id="results"></table></div></section>
</main><script>
const DATA=__DATA__;const pct=v=>typeof v==='number'?(v*100).toFixed(1)+'%':'n/a';const score=v=>typeof v==='number'?v.toFixed(1)+'%':'n/a';
const p=DATA.averages.perception,s=DATA.averages.evaluations.standard_trajectory,m=DATA.averages.evaluations.model_perception_trajectory;
const cards=[['感知 F1',pct(p.f1)],['感知 Precision',pct(p.precision)],['感知 Recall',pct(p.recall)],['幻觉控制',pct(p.hallucination_control)],['标准轨迹治疗分',score(s.treatment)],['标准轨迹随访分',score(s.followup)],['模型感知轨迹治疗分',score(m.treatment)],['模型感知轨迹随访分',score(m.followup)]];
document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('');
const head='<thead><tr><th rowspan="2">Report</th><th colspan="4">模型感知</th><th colspan="4">标准轨迹评估</th><th colspan="4">模型感知轨迹评估</th></tr><tr><th>Precision</th><th>Recall</th><th>F1</th><th>幻觉控制</th><th>ACC</th><th>ERS</th><th>治疗分</th><th>随访分</th><th>ACC</th><th>ERS</th><th>治疗分</th><th>随访分</th></tr></thead>';
const evalCells=e=>`<td>${score(e.acc.overall.score)}</td><td>${score(e.ers.overall.score)}</td><td>${score(e.tps.overall_percent)}</td><td>${score(e.followup.overall_percent)}</td>`;
const body=DATA.rows.map(r=>`<tr><td><a class="report-link" href="${r.href}">${r.report}</a></td><td>${pct(r.perception.precision)}</td><td>${pct(r.perception.recall)}</td><td>${pct(r.perception.f1)}</td><td>${pct(r.perception.hallucination_control)}</td>${evalCells(r.evaluations.standard_trajectory)}${evalCells(r.evaluations.model_perception_trajectory)}</tr>`).join('');
const avg=`<tr class="group"><td>四 Report 平均</td><td>${pct(p.precision)}</td><td>${pct(p.recall)}</td><td>${pct(p.f1)}</td><td>${pct(p.hallucination_control)}</td><td>${score(s.acc)}</td><td>${score(s.ers)}</td><td>${score(s.treatment)}</td><td>${score(s.followup)}</td><td>${score(m.acc)}</td><td>${score(m.ers)}</td><td>${score(m.treatment)}</td><td>${score(m.followup)}</td></tr>`;
document.getElementById('results').innerHTML=head+`<tbody>${body}${avg}</tbody>`;
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="生成四 Report Step4 汇总 HTML")
    parser.add_argument("--report-root", type=Path, default=ROOT / "outputs" / "report")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--method", default="full_context_memory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = collect(args.report_root, args.reports, args.model, args.method)
    output = args.output or args.report_root / "four_report_summary.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        HTML.replace("__MODEL__", args.model).replace("__DATA__", json.dumps(data, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"[evaluation][summary/html] written path={output}", flush=True)


if __name__ == "__main__":
    main()
