"""Step4 + Step5 评测: 按阶段流式提问作答并打分, 汇总对比报告
    python step4_evaluation/run_step4_chenfang.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from config import get_settings, load_env
from llm_client import ChatClient
from step4_evaluation.evaluator import CachedLLM, run_streaming
from step4_evaluation.memory import build_methods
from step4_evaluation.report import build_report, format_console


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def group_tasks_by_stage(tasks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for task in tasks:
        grouped.setdefault(task["ask_after_stage"], []).append(task)
    return grouped


def load_rubric_index(out: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for name in ("diagnosis_rubrics.json", "treatment_rubrics.json"):
        path = out / "rubrics" / name
        if path.exists():
            for rubric in read_json(path):
                index[rubric["task_id"]] = rubric
    return index


def resolve_trajectory_path(out: Path, name: str) -> Path:
    # 把轨迹名解析为文件路径
    if name in ("standard", "standard_full", "standard_trajectory"):
        return out / "trajectories" / "standard_trajectory.json"
    return out / "variants" / f"{name}.json"


def evaluate_trajectory(
    trajectory: dict,
    tasks_by_stage: dict[str, list[dict]],
    rubric_by_task: dict[str, dict],
    answer_client: ChatClient,
    verifier_client: ChatClient,
    out: Path,
    methods: list[str] | None = None,
    multimodal: bool = False,
    image_root: Path | None = None,
) -> dict:
    # 对单条轨迹跑选定记忆方法并打分, 返回报告
    # 缓存/输出按 trajectory_type(及模态)隔离
    ttype = trajectory["trajectory_type"]
    mode = "multimodal" if multimodal else "text"
    suffix = "_mm" if multimodal else ""
    present_stages = {s["stage_id"] for s in trajectory["stages"]}
    print(f"\n########## Trajectory: {trajectory['trajectory_id']} "
          f"({len(trajectory['stages'])} stages) | mode={mode} ##########", flush=True)
    missing = sorted({sid for sid in tasks_by_stage if sid not in present_stages})
    if missing:
        print(f"  (stages absent in this trajectory, their tasks released at END: {', '.join(missing)})", flush=True)

    eval_dir = out / "evaluation" / f"{ttype}{suffix}"
    cache_root = out / "cache" / "step4" / f"{ttype}{suffix}"

    records_by_method: dict[str, list[dict]] = {}
    llm_by_method: dict[str, CachedLLM] = {}
    for method in build_methods(names=methods, multimodal=multimodal):
        print(f"\n=== [{ttype}/{mode}] method: {method.name} ===", flush=True)
        method_dir = cache_root / method.name
        method.setup(method_dir)
        answer_llm = CachedLLM(answer_client, method_dir / "answer")
        verifier_llm = CachedLLM(verifier_client, method_dir / "verifier")
        llm_by_method[method.name] = verifier_llm
        records = run_streaming(method, trajectory, tasks_by_stage, answer_llm, image_root)
        records_by_method[method.name] = records
        write_json(eval_dir / f"answers_{method.name}.json", records)

    print(f"\n=== [{ttype}/{mode}] scoring ===", flush=True)
    report = build_report(records_by_method, rubric_by_task, llm_by_method)
    report["trajectory_id"] = trajectory["trajectory_id"]
    report["trajectory_type"] = ttype
    report["mode"] = mode
    report["patient_id"] = trajectory["patient_id"]

    write_json(eval_dir / "report.json", report)
    console = format_console(report)
    (eval_dir / "report.txt").write_text(console, encoding="utf-8")
    print("\n" + console)
    print(f"Report written to: {eval_dir}", flush=True)
    for name, llm in llm_by_method.items():
        print(f"  [{name}] LLM calls={llm.calls} cache_hits={llm.hits}", flush=True)
    return report


def main() -> None:
    # 加载 OralMemo/.env
    load_env(BENCH_ROOT / ".env")
    settings = get_settings()
    out = settings.output_root

    # 解析参数: --trajectories 轨迹; --multimodal 开多模态; --methods 选方法
    parser = argparse.ArgumentParser(description="Step4 + Step5 端到端评测")
    parser.add_argument("--trajectories", type=lambda s: s.split(","), default=["standard"])
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--methods", type=lambda s: s.split(","))
    args = parser.parse_args()

    multimodal = args.multimodal
    methods = args.methods
    names = args.trajectories or ["standard"]
    
    # 轨迹里的图片路径
    image_root = BENCH_ROOT if multimodal else None

    # 任务与 rubric 与轨迹无关
    tasks = read_json(out / "tasks" / "all_tasks.json")["tasks"]
    tasks_by_stage = group_tasks_by_stage(tasks)
    rubric_by_task = load_rubric_index(out)

    answer_cfg = settings.llm_for("answer")
    verifier_cfg = settings.llm_for("verifier")
    answer_client = ChatClient(
        api_key=answer_cfg.api_key,
        base_url=answer_cfg.base_url,
        model=answer_cfg.model,
    )
    verifier_client = ChatClient(
        api_key=verifier_cfg.api_key,
        base_url=verifier_cfg.base_url,
        model=verifier_cfg.model,
    )

    method_label = ", ".join(methods) if methods else "default(single_stage_memory)"
    print(f"Evaluating {len(names)} trajectory(ies): {', '.join(names)} | {len(tasks)} tasks each "
          f"| mode={'multimodal' if multimodal else 'text'} | methods={method_label}")

    reports = []
    for name in names:
        path = resolve_trajectory_path(out, name)
        if not path.exists():
            print(f"[skip] trajectory file not found: {path}", flush=True)
            continue
        reports.append(evaluate_trajectory(
            read_json(path), tasks_by_stage, rubric_by_task, answer_client, verifier_client, out,
            methods, multimodal, image_root,
        ))

    # 跨轨迹总览
    if len(reports) > 1:
        print("\n\n########## Cross-trajectory summary (summary_memory baseline) ##########")
        header = f"{'Trajectory':<20}{'ACC(mem)':>14}{'ERS(mem)':>14}{'Diag(mem)':>14}{'TPS(mem)':>14}"
        print(header)
        print("-" * len(header))
        for rep in reports:
            mem = next((m for m in rep["methods"] if m["method"] == "summary_memory"), None)
            if not mem:
                continue
            acc = f"{mem['acc']['overall']['score']:.1f}%"
            ers = f"{mem['ers']['overall']['score']:.1f}%"
            diag = f"{mem['diagnosis']['percent']:.1f}%" if mem.get("diagnosis") else "n/a"
            tps = f"{mem['tps']['overall_percent']:.1f}%" if mem['tps']['overall_percent'] is not None else "n/a"
            print(f"{rep['trajectory_type']:<20}{acc:>14}{ers:>14}{diag:>14}{tps:>14}")


if __name__ == "__main__":
    main()
