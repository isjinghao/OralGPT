"""Generate model-perception trajectories for one or more patients."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

from config import get_settings
from utils.batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients
from utils.image_utils import image_data_url
from utils.json_utils import read_json, write_json
from llm_client import ChatClient
from step1_patient_trajectory.perception_evaluation import PerceptionEvaluator


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "step1_patient_trajectory" / "prompts"
DIRECT_CONTEXT_STAGE_IDS = {"S0_PROFILE", "S5_TMJ"}


def load_prompt(filename: str, key: str) -> str:
    data = yaml.safe_load((PROMPTS_DIR / filename).read_text(encoding="utf-8"))
    return data[key]


def resolve_image_path(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else root / path


def image_urls(root: Path, image_paths: list[str]) -> list[str]:
    urls: list[str] = []
    for image_path in image_paths:
        url = image_data_url(resolve_image_path(root, image_path))
        if url:
            urls.append(url)
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
    records: list[dict] = []
    for stage in stages:
        for qa in stage.get("qa_pairs", []):
            if qa.get("image_paths"):
                return records
            records.append(
                {
                    "stage_id": stage["stage_id"],
                    "source_turn_id": qa["source_turn_id"],
                    "question": qa.get("human", ""),
                    "answer": (qa.get("assistant") or "").strip(),
                }
            )
    return records


def cache_path(
    cache_dir: Path,
    model: str,
    stage_id: str,
    source_turn_id: int,
    question: str,
    system_prompt: str,
    prompt: str,
    image_paths: list[str],
) -> Path:
    payload = "\n".join(
        [
            model,
            stage_id,
            str(source_turn_id),
            question,
            system_prompt,
            prompt,
            ",".join(image_paths),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"perception_{stage_id}_{source_turn_id}_{digest}.json"


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
    path = cache_path(
        cache_dir,
        client.model,
        stage["stage_id"],
        int(qa.get("source_turn_id", 0)),
        question,
        system_prompt,
        prompt,
        qa.get("image_paths", []) or [],
    )
    if path.exists() and not force:
        return read_json(path)["answer"]

    urls = image_urls(image_root, qa.get("image_paths", []) or [])
    if not urls:
        raise FileNotFoundError(f"No readable images for {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}")

    answer = client.complete_text(
        prompt,
        temperature=0.0,
        max_tokens=2048,
        images=urls,
        timeout=300,
        system_prompt=system_prompt,
    ).strip()
    if not answer:
        raise RuntimeError(f"Empty model answer for {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}")
    write_json(path, {"answer": answer})
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
    profile_keys = {(item["stage_id"], int(item["source_turn_id"])) for item in profile_records}
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

            answer = generate_answer(
                client,
                profile,
                text_memory,
                memory,
                stage,
                qa,
                image_root,
                cache_dir,
                force,
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
        "trajectory_id": f"{standard['patient_id']}__model_perception_trajectory",
        "patient_id": standard["patient_id"],
        "trajectory_type": "model_perception_trajectory",
        "stages": generated_stages,
    }
    for key in ("patient_name", "group"):
        if key in standard:
            result[key] = standard[key]
    return result


def run_patient(item: dict, settings, model: str | None, base_url: str | None, force: bool) -> str:
    patient_id = item["id"]
    case_root = patient_output_root(settings.bench_root, patient_id)
    standard_path = case_root / "trajectories" / "standard_trajectory.json"
    evidence_path = case_root / "evidence" / "evidence.json"

    answer_cfg = settings.llm_for("answer")
    model_name = model or answer_cfg.model
    model_base_url = base_url or answer_cfg.base_url
    output_path = case_root / "trajectories" / "model_perception_trajectory" / model_name / "model_perception_trajectory.json"
    cache_dir = case_root / "cache" / "stage1_perception" / model_name
    report_path = case_root / "trajectories" / "model_perception_trajectory" / model_name / "perception_report.json"

    if not standard_path.is_file():
        raise FileNotFoundError(f"Standard trajectory does not exist: {standard_path}")
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Evidence does not exist: {evidence_path}")
    if not force and output_path.is_file() and report_path.is_file():
        log(f"[perception][{patient_id}][resume] model={model_name} outputs found; skipped")
        return "skipped"

    verifier_cfg = settings.llm_for("verifier")
    with (
        ChatClient(
            api_key=answer_cfg.api_key,
            base_url=model_base_url,
            model=model_name,
            log_prefix=f"[perception][{patient_id}][{model_name}]",
        ) as client,
        ChatClient(
            api_key=verifier_cfg.api_key,
            base_url=verifier_cfg.base_url,
            model=verifier_cfg.model,
            log_prefix=f"[perception-verifier][{patient_id}]",
        ) as verifier,
    ):
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
            force=force,
            evaluator=evaluator,
        )
        write_json(output_path, result)
    log(f"[perception][{patient_id}][done] model={model_name} output={output_path}")
    return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model-perception trajectories")
    add_batch_arguments(parser)
    parser.add_argument("--model", default=None, help="Override ANSWER_OPENAI_MODEL for this run")
    parser.add_argument("--base-url", default=None, help="Override ANSWER_OPENAI_BASE_URL for this run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    patients = selected_patients(settings.dataset_json, args.all, args.limit)

    def worker(item: dict) -> str:
        return run_patient(item, settings, args.model, args.base_url, args.force)

    return run_patient_batch(patients, args.num_workers, "perception", worker)


if __name__ == "__main__":
    raise SystemExit(main())
