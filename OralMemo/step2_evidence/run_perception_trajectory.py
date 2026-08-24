"""Generate model-perception trajectories for patient or report benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from config import get_settings
from llm_client import ChatClient
from report_pipeline.paths import REPORT_OUTPUT_ROOT, REPORT_PDF_DIR
from utils.batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients, selected_reports
from utils.image_utils import image_data_url
from utils.json_utils import read_json, write_json


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "step1_patient_trajectory" / "prompts"
DIRECT_CONTEXT_STAGE_IDS = {"S0_PROFILE", "S5_TMJ"}
PERCEPTION_CACHE_VERSION = "raw_output_v1"


def perception_max_tokens() -> int:
    return int(os.environ.get("PERCEPTION_MAX_TOKENS", "2048"))


def load_prompt(filename: str, key: str) -> str:
    data = yaml.safe_load((PROMPTS_DIR / filename).read_text(encoding="utf-8"))
    return data[key]


def resolve_image_path(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else root / path


def image_urls(root: Path, image_paths: list[str]) -> list[str]:
    paths = [resolve_image_path(root, image_path) for image_path in image_paths]
    return [url for path in paths if path.is_file() and (url := image_data_url(path))]


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


def format_text_memory(memory: list[dict]) -> str:
    return format_text_records(memory, "(No earlier textual clinical records are available.)")


def observation_record(stage: dict, qa: dict, answer: str) -> dict:
    return {
        "stage_id": stage["stage_id"],
        "source_turn_id": qa["source_turn_id"],
        "question": qa.get("human", ""),
        "answer": answer,
    }


def initial_profile_records(stages: list[dict]) -> list[dict]:
    records: list[dict] = []
    for stage in stages:
        for qa in stage.get("qa_pairs", []):
            if qa.get("image_paths"):
                return records
            records.append(observation_record(stage, qa, (qa.get("assistant") or "").strip()))
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
            PERCEPTION_CACHE_VERSION,
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
        return read_json(path)["answer"].strip()

    urls = image_urls(image_root, qa.get("image_paths", []) or [])
    if not urls:
        raise FileNotFoundError(f"No readable images for {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}")

    answer = client.complete_text(
        prompt,
        temperature=0.0,
        max_tokens=perception_max_tokens(),
        images=urls,
        timeout=300,
        system_prompt=system_prompt,
    ).strip()
    if not answer:
        raise RuntimeError(f"Empty model answer for {stage['stage_id']} source_turn_id={qa.get('source_turn_id')}")
    write_json(path, {"answer": answer})
    return answer


def should_generate(dataset: str, stage: dict, qa: dict) -> bool:
    images = qa.get("image_paths", []) or []
    if dataset == "report":
        return qa.get("role") == "observation" and bool(images)
    return qa.get("role") != "evaluation" and bool(images) and stage["stage_id"] not in DIRECT_CONTEXT_STAGE_IDS


def should_remember_text(dataset: str, stage: dict, qa: dict, profile_keys: set[tuple[str, int]]) -> bool:
    if qa.get("role") != "observation":
        return False
    key = (stage["stage_id"], int(qa.get("source_turn_id", 0)))
    if dataset == "patient" and key in profile_keys:
        return False
    return not should_generate(dataset, stage, qa)


def generate_trajectory(
    standard: dict,
    client: ChatClient,
    image_root: Path,
    cache_dir: Path,
    dataset: str,
    force: bool = False,
    question_workers: int = 1,
    log_prefix: str | None = None,
) -> dict:
    if question_workers < 1:
        raise ValueError("--question-workers must be a positive integer")
    stages = sorted(standard.get("stages", []), key=lambda item: item.get("order", 0))
    profile_records = initial_profile_records(stages) if dataset == "patient" else []
    profile = format_profile(profile_records)
    profile_keys = {(item["stage_id"], int(item["source_turn_id"])) for item in profile_records}
    text_memory: list[dict] = []
    generated_stages: list[dict] = []
    jobs: list[dict] = []

    for stage_index, stage in enumerate(stages):
        generated = {key: value for key, value in stage.items() if key != "qa_pairs"}
        qa_pairs: list[dict | None] = []
        for qa_index, qa in enumerate(stage.get("qa_pairs", [])):
            if should_generate(dataset, stage, qa):
                qa_pairs.append(None)
                jobs.append(
                    {
                        "stage_index": stage_index,
                        "qa_index": qa_index,
                        "stage": stage,
                        "qa": qa,
                        "profile": profile,
                        "text_memory": list(text_memory),
                    }
                )
                continue

            copied = dict(qa)
            if dataset == "patient" and stage["stage_id"] in DIRECT_CONTEXT_STAGE_IDS and qa.get("role") != "evaluation":
                copied["human"] = clean_question(copied.get("human", ""))
                copied["image_paths"] = []
                generated["image_paths"] = []
            qa_pairs.append(copied)

            if should_remember_text(dataset, stage, qa, profile_keys):
                text_memory.append(observation_record(stage, qa, (qa.get("assistant") or "").strip()))

        generated["qa_pairs"] = qa_pairs
        generated_stages.append(generated)

    def run_job(job: dict) -> tuple[int, int, dict]:
        answer = generate_answer(
            client=client,
            profile=job["profile"],
            text_memory=job["text_memory"],
            stage=job["stage"],
            qa=job["qa"],
            image_root=image_root,
            cache_dir=cache_dir,
            force=force,
        )
        stage_id = job["stage"]["stage_id"]
        source_turn_id = job["qa"].get("source_turn_id")
        if log_prefix:
            log(f"{log_prefix}[question-done] stage={stage_id} source_turn_id={source_turn_id} chars={len(answer)}")
        return job["stage_index"], job["qa_index"], {**job["qa"], "assistant": answer}

    if question_workers == 1 or len(jobs) <= 1:
        results = [run_job(job) for job in jobs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=question_workers) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    for stage_index, qa_index, qa in results:
        generated_stages[stage_index]["qa_pairs"][qa_index] = qa

    result = {
        **{key: value for key, value in standard.items() if key not in {"trajectory_id", "trajectory_type", "stages"}},
        "trajectory_id": f"{standard['patient_id']}__model_perception_trajectory",
        "trajectory_type": "model_perception_trajectory",
        "stages": generated_stages,
    }
    return result


def case_root_for_item(dataset: str, settings, item: dict) -> Path:
    if dataset == "report":
        return REPORT_OUTPUT_ROOT / item["name"]
    return patient_output_root(settings.bench_root, item["id"])


def run_case(dataset: str, item: dict, settings, model: str | None, base_url: str | None, force: bool, question_workers: int) -> str:
    item_id = item["id"]
    case_root = case_root_for_item(dataset, settings, item)
    standard_path = case_root / "trajectories" / "standard_trajectory.json"

    answer_cfg = settings.llm_for("answer")
    model_name = model or answer_cfg.model
    model_base_url = base_url or answer_cfg.base_url
    model_root = case_root / "trajectories" / "model_perception_trajectory" / model_name
    output_path = model_root / "model_perception_trajectory.json"
    cache_dir = case_root / "cache" / "stage1_perception" / model_name

    if not standard_path.is_file():
        raise FileNotFoundError(f"Standard trajectory does not exist: {standard_path}")
    if not force and output_path.is_file():
        log(f"[{dataset}-perception][{item_id}][resume] model={model_name} trajectory found; skipped")
        return "skipped"

    with ChatClient(
        api_key=answer_cfg.api_key,
        base_url=model_base_url,
        model=model_name,
        log_prefix=f"[{dataset}-perception][{item_id}][{model_name}]",
    ) as client:
        result = generate_trajectory(
            standard=read_json(standard_path),
            client=client,
            image_root=settings.bench_root,
            cache_dir=cache_dir,
            dataset=dataset,
            force=force,
            question_workers=question_workers,
            log_prefix=f"[{dataset}-perception][{item_id}][{model_name}]",
        )
        write_json(output_path, result)
    log(f"[{dataset}-perception][{item_id}][done] model={model_name} output={output_path}")
    return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model-perception trajectories")
    parser.add_argument("--dataset", choices=("patient", "report"), default="patient", help="Benchmark subset to process")
    add_batch_arguments(parser)
    parser.add_argument("--model", default=None, help="Override ANSWER_OPENAI_MODEL for this run")
    parser.add_argument("--base-url", default=None, help="Override ANSWER_OPENAI_BASE_URL for this run")
    parser.add_argument("--question-workers", type=int, default=1, help="Questions processed concurrently within each patient/report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    items = selected_reports(REPORT_PDF_DIR, args.all, args.limit) if args.dataset == "report" else selected_patients(settings.dataset_json, args.all, args.limit)

    def worker(item: dict) -> str:
        return run_case(args.dataset, item, settings, args.model, args.base_url, args.force, args.question_workers)

    return run_patient_batch(items, args.num_workers, f"{args.dataset}-perception", worker)


if __name__ == "__main__":
    raise SystemExit(main())
