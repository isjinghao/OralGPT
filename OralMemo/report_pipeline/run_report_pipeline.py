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

from config import get_settings
from llm_client import ChatClient
from report_pipeline.step0_ingest.pdf_extract import extract_pdf
from report_pipeline.step1_report_trajectory.qa_render import normalize_timepoints, render_turns
from report_pipeline.step0_ingest.timeline_llm import extract_timeline
from report_pipeline.step0_ingest.verify_llm import high_severity_issues, verify_timeline
from report_pipeline.step1_report_trajectory.report_dataset import build_report_dataset_entry
from report_pipeline.step1_report_trajectory.report_stages import build_report_stages
from step1_patient_trajectory.trajectories import build_standard_trajectory


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_with_feedback(
    extract_client,
    verifier_client,
    raw_dir: Path,
    figures: list[dict],
    max_iters: int,
) -> tuple[dict, list[dict]]:
    """抽取并根据校验模型的 high/medium 问题迭代修正。"""
    history: list[dict] = []
    feedback: list[dict] | None = None
    timeline: dict = {}

    for it in range(1, max_iters + 1):
        print(f"[loop {it}/{max_iters}] 抽取模型生成时间线 ...", flush=True)
        timeline = extract_timeline(extract_client, raw_dir, figures, feedback_issues=feedback)

        print(f"[loop {it}/{max_iters}] 校验模型核验 ...", flush=True)
        verification = verify_timeline(verifier_client, raw_dir, timeline, captions=figures)
        highs = high_severity_issues(verification)
        feedback = [
            issue for issue in verification["issues"]
            if issue["severity"] in ("high", "medium")
        ]
        history.append(
            {
                "iteration": it,
                "passed": verification["passed"],
                "n_issues": len(verification["issues"]),
                "n_high": len(highs),
                "n_actionable": len(feedback),
                "verification": verification,
            }
        )
        print(
            f"[loop {it}/{max_iters}] passed={verification['passed']} "
            f"issues={len(verification['issues'])} high={len(highs)} "
            f"actionable={len(feedback)}",
            flush=True,
        )
        if not feedback:
            break

    return timeline, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reuse-ingest",
        action="store_true",
        help="Reuse existing raw/fulltext.json, tables.json, captions.json, and images",
    )
    args = parser.parse_args()

    s = get_settings()
    benchmark_cfg = s.llm_for("benchmark")
    verifier_cfg = s.llm_for("verifier")
    client = ChatClient(
        api_key=benchmark_cfg.api_key,
        base_url=benchmark_cfg.base_url,
        model=args.model or benchmark_cfg.model,
    )
    verifier_client = ChatClient(
        api_key=verifier_cfg.api_key,
        base_url=verifier_cfg.base_url,
        model=verifier_cfg.model,
    )

    name = args.name
    out_dir = BENCH_ROOT / "outputs" / "report" / name
    raw_dir = out_dir / "raw"
    images_dir = out_dir / "images"
    pdf_path = Path(args.pdf) if Path(args.pdf).is_absolute() else (BENCH_ROOT / args.pdf)

    # Step0: PDF 抽取(MinerU)，或在长时模型调用重试时复用已完成的摄取结果。
    if args.reuse_ingest:
        required = [raw_dir / "fulltext.json", raw_dir / "tables.json", raw_dir / "captions.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot reuse ingest; missing files: {missing}")
        images_map = json.loads((raw_dir / "captions.json").read_text(encoding="utf-8"))
        captions = [
            {"figure": figure, "caption": entry.get("caption", "")}
            for figure, entry in images_map.items()
        ]
        print(f"[step0] 复用已抽取 PDF: 图注={len(captions)} 图片={sum(len(x.get('images', [])) for x in images_map.values())}")
    else:
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
    timeline, verif_history = extract_with_feedback(
        client, verifier_client, raw_dir, captions, args.max_iters
    )
    write_json(out_dir / "timeline.extracted.json", timeline)
    write_json(out_dir / "verification_report.json", verif_history)
    final = verif_history[-1]
    final_passed = bool(
        final["passed"] and final["n_high"] == 0 and final["n_actionable"] == 0
    )

    # Step1: 阶段化 + 轨迹 + 数据集
    print("[step1] 规整时间点 / 问答构造 / 组装数据集 / 切分阶段 / 构造标准轨迹")
    patient = {"patient_id": f"report__{name}", "name": name, "group": "report"}
    normed = normalize_timepoints(timeline)
    rendered = render_turns(normed, images_map)
    stages = build_report_stages(normed, rendered, patient)
    standard = build_standard_trajectory(stages)
    entry = build_report_dataset_entry(standard, patient, source_pdf)

    write_json(out_dir / "trajectories" / "standard_trajectory.json", standard)
    write_json(out_dir / "dataset_entry.json", entry)

    evaluation_count = sum(
        qa["role"] == "evaluation"
        for stage in standard["stages"]
        for qa in stage["qa_pairs"]
    )
    print(f"[step1] 完成: 病人={entry['id']} 阶段={len(standard['stages'])} "
          f"QA轮={entry['num_qa_pairs']} 图片={entry['num_images']} "
          f"评测问题={evaluation_count}")
    print(f"[result] 校验最终结论: {'通过' if final_passed else '未通过'}")


if __name__ == "__main__":
    main()
