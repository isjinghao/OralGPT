"""report_pipeline: step0(PDF 摄取) + LLM 抽取 + 校验模型反馈循环 + step1(阶段化/轨迹)
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
from report_pipeline.config_reports import get_report_settings, name_paths
from report_pipeline.step0_ingest.pdf_extract import extract_pdf
from report_pipeline.step1_report_trajectory.qa_render import normalize_timepoints, render_turns
from report_pipeline.step0_ingest.timeline_llm import extract_timeline
from report_pipeline.step0_ingest.verify_llm import high_severity_issues, verify_timeline
from report_pipeline.step1_report_trajectory.report_dataset import build_report_dataset_entry
from report_pipeline.step1_report_trajectory.report_stages import build_report_stages
from report_pipeline.step1_report_trajectory.report_trajectories import build_report_standard_trajectory


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_with_feedback(client, raw_dir: Path, figures, max_iters: int) -> tuple[dict, list[dict]]:
    """抽取<->校验反馈循环。返回(最终时间线, 每轮校验记录)。"""
    history: list[dict] = []
    feedback: list[dict] | None = None
    timeline: dict = {}
    for it in range(1, max_iters + 1):
        # 抽取时间线
        print(f"[loop {it}/{max_iters}] 抽取模型生成时间线 ...", flush=True)
        timeline = extract_timeline(client, raw_dir, figures, feedback_issues=feedback)
        
        # 校验时间线
        print(f"[loop {it}/{max_iters}] 校验模型核验 ...", flush=True)
        verification = verify_timeline(client, raw_dir, timeline, captions=figures)
        highs = high_severity_issues(verification)
        history.append({"iteration": it, "passed": verification.get("passed"),
                        "n_issues": len(verification.get("issues", [])),
                        "n_high": len(highs), "verification": verification})
        print(f"[loop {it}/{max_iters}] passed={verification.get('passed')} "
              f"issues={len(verification.get('issues', []))} high={len(highs)}",
              flush=True)
        if verification.get("passed") and not highs:
            break
        
        # 把内容层面的 high/medium 问题作为反馈
        feedback = [i for i in verification.get("issues", [])
                    if i.get("severity") in ("high", "medium")]
        if not feedback:
            break
    return timeline, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    settings = get_report_settings()
    load_env(BENCH_ROOT / ".env")
    s = get_settings()
    client = ChatClient(
        api_key=s.openai_api_key,
        base_url=s.openai_base_url,
        model=args.model or s.openai_model,
    )

    name = args.name
    paths = name_paths(settings, name)
    out_dir, raw_dir, images_dir = paths["out_dir"], paths["raw_dir"], paths["images_dir"]
    pdf_path = Path(args.pdf) if Path(args.pdf).is_absolute() else (BENCH_ROOT / args.pdf)

    # Step0: PDF 抽取(MinerU)
    print(f"[step0] 抽取 PDF: {pdf_path}")
    summary = extract_pdf(pdf_path, raw_dir, images_dir, rel_base=BENCH_ROOT)
    images_map = summary["images_map"]
    captions = summary["captions"]
    write_json(raw_dir / "captions.json", images_map)
    print(f"[step0] 页数={summary['n_pages']} 有效图片={summary['n_images_kept']} "
          f"表格={summary['n_tables']} 图注={len(captions)}")

    source_pdf = (
        pdf_path.relative_to(BENCH_ROOT).as_posix() if str(pdf_path).startswith(str(BENCH_ROOT)) else str(pdf_path)
    )

    # 抽取 <-> 校验 反馈循环
    timeline, verif_history = extract_with_feedback(client, raw_dir, captions, args.max_iters)
    write_json(out_dir / "timeline.extracted.json", timeline)
    write_json(out_dir / "verification_report.json", verif_history)
    final_passed = bool(verif_history and verif_history[-1]["passed"] and verif_history[-1]["n_high"] == 0)

    # Step1: 阶段化 + 轨迹 + 数据集
    print("[step1] 规整时间点 / 问答构造 / 组装数据集 / 切分阶段 / 构造标准轨迹")
    patient = {"patient_id": f"report__{name}", "name": name, "group": "report"}
    normed = normalize_timepoints(timeline)
    rendered = render_turns(normed, timeline.get("held_out", {}), images_map)
    entry = build_report_dataset_entry(normed, rendered, patient, source_pdf)
    patient_stages = build_report_stages(normed, rendered, patient)
    standard = build_report_standard_trajectory(patient_stages)

    write_json(out_dir / "stages" / "patient_stages.json", patient_stages)
    write_json(out_dir / "trajectories" / "standard_trajectory.json", standard)
    write_json(out_dir / "dataset_entry.json", entry)

    print(f"[step1] 完成: 病人={entry['id']} 阶段={len(patient_stages['stages'])} "
          f"QA轮={entry['num_qa_pairs']} 图片={entry['num_images']} "
          f"held-out={len(patient_stages['heldout_turns'])}")
    print(f"[result] 校验最终结论: {'通过' if final_passed else '未通过'}")


if __name__ == "__main__":
    main()
