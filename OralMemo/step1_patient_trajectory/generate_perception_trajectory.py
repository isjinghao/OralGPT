"""基于患者 profile 与阶段图片生成模型感知轨迹。"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path

import yaml

from config import get_settings
from llm_client import ChatClient
from step1_patient_trajectory.perception_evaluation import PerceptionEvaluator


PROMPTS_DIR = Path(__file__).with_name("prompts")
DIRECT_CONTEXT_STAGE_IDS = {"S0_PROFILE", "S5_TMJ"}


def load_prompt(filename: str, key: str) -> str:
    data = yaml.safe_load((PROMPTS_DIR / filename).read_text(encoding="utf-8"))
    return data[key]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def encode_image(path: Path) -> str | None:
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def resolve_image_path(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else root / path


def image_urls(root: Path, image_paths: list[str]) -> list[str]:
    urls: list[str] = []
    for image_path in image_paths:
        path = resolve_image_path(root, image_path)
        url = encode_image(path)
        if url:
            urls.append(url)
        else:
            print(f"[warning] image not found: {path}", flush=True)
    return urls


def clean_question(question: str) -> str:
    return question.replace("<image>", "").strip()


def format_text_records(records: list[dict], empty_message: str) -> str:
    if not records:
        return empty_message
    return "\n\n".join(
        f"[{item['stage_id']} | source_turn_id={item['source_turn_id']}]\n"
        f"Q: {clean_question(item['question'])}\nA: {item['answer']}"
        for item in records
    )


def format_profile(records: list[dict]) -> str:
    return format_text_records(records, "(No separate profile text was provided.)")


def format_memory(memory: list[dict]) -> str:
    return format_text_records(memory, "(No previous image-based observations have been generated yet.)")


def format_text_memory(memory: list[dict]) -> str:
    return format_text_records(memory, "(No earlier textual clinical records are available.)")


def initial_profile_records(stages: list[dict]) -> list[dict]:
    """收集首个图片问题之前的文本记录，作为后续每轮感知都可见的 profile。"""
    records: list[dict] = []
    for stage in stages:
        for qa in stage.get("qa_pairs", []):
            images = qa.get("image_paths", []) or []
            if images:
                return records
            records.append({
                "stage_id": stage["stage_id"],
                "source_turn_id": qa["source_turn_id"],
                "question": qa.get("human", ""),
                "answer": (qa.get("assistant") or "").strip(),
            })
    return records


def cache_path(cache_dir: Path, stage_id: str, source_turn_id: int) -> Path:
    return cache_dir / f"perception_{stage_id}_{source_turn_id}.json"


def generate_answer(
    client: ChatClient,
    profile: str,
    text_memory: list[dict],
    memory: list[dict],
    stage: dict,
    qa: dict,
    image_root: Path,
    cache_dir: Path,
    force: bool,
) -> str:
    question = clean_question(qa.get("human", ""))
    system_prompt = load_prompt("perception_system.yaml", "system_prompt")
    prompt = load_prompt("perception_user.yaml", "user_prompt").format(
        profile=profile,
        text_memory=format_text_memory(text_memory),
        memory=format_memory(memory),
        stage_id=stage["stage_id"],
        modality=", ".join(stage.get("modality", [])) or "unknown",
        question=question,
    )
    cache_input = {
        "model": client.model,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "image_paths": qa.get("image_paths", []) or [],
        "max_tokens": 16000,
    }
    path = cache_path(cache_dir, stage["stage_id"], int(qa.get("source_turn_id", 0)))
    if path.exists() and not force:
        cached = read_json(path)
        if cached.get("input") == cache_input:
            return str(cached["answer"]).strip()

    urls = image_urls(image_root, qa.get("image_paths", []) or [])
    if not urls:
        raise FileNotFoundError(
            f"No readable images for {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}"
        )
    answer = client.complete_text(
        prompt,
        temperature=0.0,
        max_tokens=16000,
        images=urls,
        system_prompt=system_prompt,
    ).strip()
    if not answer:
        raise RuntimeError(
            f"Empty model answer for {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}"
        )
    write_json(
        path,
        {
            "input": cache_input,
            "answer": answer,
            "stage_id": stage["stage_id"],
            "source_turn_id": qa.get("source_turn_id"),
        },
    )
    return answer


def generate_trajectory(
    standard: dict,
    client: ChatClient,
    image_root: Path,
    cache_dir: Path,
    force: bool = False,
    evaluator: PerceptionEvaluator | None = None,
) -> dict:
    stages = sorted(standard.get("stages", []), key=lambda item: item.get("order", 0))
    profile_records = initial_profile_records(stages)
    profile = format_profile(profile_records)
    profile_keys = {
        (item["stage_id"], int(item["source_turn_id"])) for item in profile_records
    }
    text_memory: list[dict] = []
    memory: list[dict] = []
    generated_stages: list[dict] = []

    for stage in stages:
        generated = {key: value for key, value in stage.items() if key != "qa_pairs"}
        qa_pairs: list[dict] = []
        for qa in stage.get("qa_pairs", []):
            if qa["role"] == "evaluation":
                qa_pairs.append(dict(qa))
                continue
            images = qa.get("image_paths", []) or []
            key = (stage["stage_id"], int(qa.get("source_turn_id", 0)))
            if stage["stage_id"] in DIRECT_CONTEXT_STAGE_IDS or not images:
                # profile、TMJ/ECT 和无图文本直接保留标准答案，作为后续文本上下文。
                copied = dict(qa)
                if stage["stage_id"] in DIRECT_CONTEXT_STAGE_IDS:
                    copied["human"] = clean_question(copied["human"])
                    copied["image_paths"] = []
                    generated["image_paths"] = []
                qa_pairs.append(copied)
                if key not in profile_keys:
                    text_memory.append(
                        {
                            "stage_id": stage["stage_id"],
                            "source_turn_id": qa["source_turn_id"],
                            "question": qa.get("human", ""),
                            "answer": (qa.get("assistant") or "").strip(),
                        }
                    )
                continue

            print(
                f"[perception] {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}",
                flush=True,
            )
            answer = generate_answer(
                client, profile, text_memory, memory, stage, qa,
                image_root, cache_dir, force,
            )
            qa_pairs.append({**qa, "assistant": answer})
            if evaluator is not None:
                evaluator.add_and_write(stage, qa, answer)
            memory.append(
                {
                    "stage_id": stage["stage_id"],
                    "source_turn_id": qa["source_turn_id"],
                    "question": qa["human"],
                    "answer": answer,
                }
            )
        generated["qa_pairs"] = qa_pairs
        generated_stages.append(generated)

    result = {
        "trajectory_id": f"{standard['patient_id']}__model_perception",
        "patient_id": standard["patient_id"],
    }
    for key in ("patient_name", "group"):
        if key in standard:
            result[key] = standard[key]
    result.update({"trajectory_type": "model_perception", "stages": generated_stages})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a model-perception trajectory from profile and images")
    parser.add_argument("--role", choices=("benchmark", "answer", "verifier"), default="answer")
    parser.add_argument("--standard", type=Path, default=None, help="Path to standard_trajectory.json")
    parser.add_argument("--output", type=Path, default=None, help="Output model perception trajectory path")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--evidence", type=Path, default=None, help="Evidence JSON for perception scoring")
    parser.add_argument("--report", type=Path, default=None, help="Perception evaluation report path")
    parser.add_argument("--force", action="store_true", help="Ignore perception answer cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    standard_path = args.standard or settings.output_root / "trajectories" / "standard_trajectory.json"
    case_root = standard_path.parent.parent
    output_path = args.output or case_root / "trajectories" / "model_perception_trajectory.json"
    cache_dir = args.cache_dir or case_root / "cache" / "stage1_perception"
    evidence_path = args.evidence or case_root / "evidence" / "evidence.json"
    report_path = args.report or case_root / "evaluation" / "perception_report.json"
    cfg = settings.llm_for(args.role)
    verifier_cfg = settings.llm_for("verifier")
    client = ChatClient(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model)
    verifier = ChatClient(
        api_key=verifier_cfg.api_key,
        base_url=verifier_cfg.base_url,
        model=verifier_cfg.model,
    )
    standard = read_json(standard_path)
    evidence = read_json(evidence_path)
    evaluator = PerceptionEvaluator(
        verifier=verifier,
        standard=standard,
        evidence=evidence,
        cache_dir=cache_dir / "verifier",
        report_path=report_path,
    )
    result = generate_trajectory(
        standard=standard,
        client=client,
        image_root=settings.bench_root,
        cache_dir=cache_dir,
        force=args.force,
        evaluator=evaluator,
    )
    write_json(output_path, result)
    print(f"[perception] trajectory written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
