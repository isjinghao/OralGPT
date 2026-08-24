"""生成多个 Report 的模型感知与 Step4 评估汇总 HTML。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from utils.json_utils import read_json

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = ["standard_trajectory", "model_perception_trajectory"]
PERCEPTION_METRICS = ("precision", "recall", "f1", "hallucination_control")


def read_method(report_root: Path, report: str, trajectory: str, model: str, method: str) -> dict:
    data = read_json(report_root / report / "evaluation" / trajectory / model / "report.json")
    return next(item for item in data["methods"] if item["method"] == method)


def infer_stage_type(stage_id: str) -> str | None:
    for stage_type in ("perception", "followup"):
        if f"_{stage_type}_" in stage_id or stage_id.endswith(f"_{stage_type}"):
            return stage_type
    return None


def summarize_metric_records(metrics: list[dict]) -> dict:
    total_gold = sum(item["gold_evidence_count"] for item in metrics)
    total_predicted = sum(item["predicted_claim_count"] for item in metrics)
    total_matched_claims = sum(item["matched_claim_count"] for item in metrics)
    total_matched_evidence = sum(item["matched_evidence_count"] for item in metrics)
    total_hallucinations = sum(item["hallucination_claim_count"] for item in metrics)
    precision = round(total_matched_claims / total_predicted, 4) if total_predicted else 0.0
    recall = round(total_matched_evidence / total_gold, 4) if total_gold else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_control": round((total_predicted - total_hallucinations) / total_predicted, 4) if total_predicted else 0.0,
    }


def perception_by_stage_type(report: dict) -> dict[str, dict]:
    if "by_stage_type" in report:
        return report["by_stage_type"]
    grouped: dict[str, list[dict]] = {}
    for record in report.get("per_question", []):
        stage_type = infer_stage_type(record.get("stage_id", ""))
        if stage_type:
            grouped.setdefault(stage_type, []).append(record["metrics"])
    return {
        stage_type: summarize_metric_records(metrics)
        for stage_type, metrics in grouped.items()
    }


def collect(report_root: Path, reports: list[str], model: str, method: str) -> dict:
    rows = []
    for report in reports:
        perception_report = read_json(
            report_root / report / "trajectories" / "model_perception_trajectory" / model / "perception_report.json"
        )
        perception = perception_report["overall"]
        perception_stages = perception_by_stage_type(perception_report)
        evaluations = {
            trajectory: read_method(report_root, report, trajectory, model, method)
            for trajectory in TRAJECTORIES
        }
        rows.append({
            "report": report,
            "href": f"{report}/evaluation/{report.lower()}_results.html",
            "perception": perception,
            "perception_by_stage_type": perception_stages,
            "evaluations": evaluations,
        })
    averages = {
        "perception": {
            key: mean(row["perception"][key] for row in rows)
            for key in PERCEPTION_METRICS
        },
        "perception_by_stage_type": {
            stage_type: {
                key: mean(
                    row["perception_by_stage_type"][stage_type][key]
                    for row in rows
                    if stage_type in row["perception_by_stage_type"]
                )
                for key in PERCEPTION_METRICS
            }
            for stage_type in sorted({
                stage_type
                for row in rows
                for stage_type in row["perception_by_stage_type"]
            })
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
.cards{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:12px}.card{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fbfdff;text-align:center}.value{font-size:25px;font-weight:750;margin-top:8px}.label{color:var(--muted);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:11px 9px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}th{background:#f8fafc;color:#475467}.group{background:#eef4ff;font-weight:700}.report-link{color:var(--blue);font-weight:700;text-decoration:none}.report-link:hover{text-decoration:underline}.hint{font-size:13px;color:var(--muted);line-height:1.7}.formula{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:2px 6px}
@media(max-width:1200px){.cards{grid-template-columns:repeat(4,minmax(0,1fr))}}@media(max-width:900px){main{padding:16px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.table-wrap{overflow:auto}}
</style></head><body>
<header><h1>OralMemo 四 Report 结果汇总</h1><div class="sub">模型：__MODEL__ · 记忆方法：full_context_memory</div></header>
<main>
<section class="panel"><h2>四 Report 平均分</h2><div id="cards" class="cards"></div></section>
<section class="panel"><h2>逐 Report 结果</h2><div class="hint">点击 Report 名称进入详情页，可查看模型感知问题、标准参考与模型感知答案，并切换标准轨迹和模型感知轨迹查看治疗、随访及完整逐题评估。</div><div class="table-wrap"><table id="results"></table></div></section>
</main><script>
const DATA=__DATA__;const pct=v=>typeof v==='number'?(v*100).toFixed(1)+'%':'n/a';const score=v=>typeof v==='number'?v.toFixed(1)+'%':'n/a';
const p=DATA.averages.perception,bp=DATA.averages.perception_by_stage_type||{},ip=bp.perception||{},fp=bp.followup||{},s=DATA.averages.evaluations.standard_trajectory,m=DATA.averages.evaluations.model_perception_trajectory;
const cards=[['感知 F1',pct(p.f1)],['初诊感知 F1',pct(ip.f1)],['随访感知 F1',pct(fp.f1)],['感知 Precision',pct(p.precision)],['感知 Recall',pct(p.recall)],['幻觉控制',pct(p.hallucination_control)],['标准轨迹治疗分',score(s.treatment)],['标准轨迹随访分',score(s.followup)],['模型感知轨迹治疗分',score(m.treatment)],['模型感知轨迹随访分',score(m.followup)]];
document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('');
const head='<thead><tr><th rowspan="2">Report</th><th colspan="6">模型感知</th><th colspan="4">标准轨迹评估</th><th colspan="4">模型感知轨迹评估</th></tr><tr><th>Precision</th><th>Recall</th><th>F1</th><th>初诊F1</th><th>随访F1</th><th>幻觉控制</th><th>ACC</th><th>ERS</th><th>治疗分</th><th>随访分</th><th>ACC</th><th>ERS</th><th>治疗分</th><th>随访分</th></tr></thead>';
const evalCells=e=>`<td>${score(e.acc.overall.score)}</td><td>${score(e.ers.overall.score)}</td><td>${score(e.tps.overall_percent)}</td><td>${score(e.followup.overall_percent)}</td>`;
const stageF1=(r,k)=>pct((r.perception_by_stage_type&&r.perception_by_stage_type[k]||{}).f1);
const body=DATA.rows.map(r=>`<tr><td><a class="report-link" href="${r.href}">${r.report}</a></td><td>${pct(r.perception.precision)}</td><td>${pct(r.perception.recall)}</td><td>${pct(r.perception.f1)}</td><td>${stageF1(r,'perception')}</td><td>${stageF1(r,'followup')}</td><td>${pct(r.perception.hallucination_control)}</td>${evalCells(r.evaluations.standard_trajectory)}${evalCells(r.evaluations.model_perception_trajectory)}</tr>`).join('');
const avg=`<tr class="group"><td>四 Report 平均</td><td>${pct(p.precision)}</td><td>${pct(p.recall)}</td><td>${pct(p.f1)}</td><td>${pct(ip.f1)}</td><td>${pct(fp.f1)}</td><td>${pct(p.hallucination_control)}</td><td>${score(s.acc)}</td><td>${score(s.ers)}</td><td>${score(s.treatment)}</td><td>${score(s.followup)}</td><td>${score(m.acc)}</td><td>${score(m.ers)}</td><td>${score(m.treatment)}</td><td>${score(m.followup)}</td></tr>`;
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
