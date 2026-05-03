#!/usr/bin/env python3
"""
CMF dataset -> local agent x-ray comprehensive report pipeline.

Input dataset structure:
  SH9HCMFdata/
    group1..groupN/
      PATIENT_NAME/
        XR/XP.png

Output:
  One JSON per patient, compatible with the existing per-patient JSON shape.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from PIL import Image


XRAY_TASK_TYPE = "XP"

XRAY_COMPREHENSIVE_REPORT_PROMPT = """
Please generate a comprehensive panoramic x-ray report based on the provided panoramic x-ray image.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local OpenAI-compatible agent on CMF x-ray images and save "
            "one comprehensive report JSON per patient."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata",
        help="Path to SH9HCMFdata root.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/patient_json_xray_agent_report",
        help="Directory to save per-patient JSON files.",
    )
    parser.add_argument(
        "--model",
        default="local-agent",
        help="Local agent model name for vision reasoning.",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible local agent API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help=(
            "Environment variable name that stores API key. If unset, "
            "--api-key will be used."
        ),
    )
    parser.add_argument(
        "--api-key",
        default="local-agent",
        help="Fallback API key for local OpenAI-compatible services.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing patient JSON files.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Only process first N patients (for debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the local agent; only scan dataset and write skeleton JSON.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for patient-level parallel processing.",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=1024,
        help="Resize image so its longest side is at most this many pixels before sending to the upstream vision model.",
    )
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality for compressed images sent to the upstream vision model.",
    )
    return parser.parse_args()


def image_to_data_url(image_path: Path, max_side: int, jpeg_quality: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_local_agent_vision(
    client: OpenAI,
    model: str,
    prompt: str,
    image_paths: List[Path],
    image_max_side: int,
    image_jpeg_quality: int,
) -> Dict[str, Any]:
    input_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        resolved_image_path = image_path.resolve()
        input_content.append(
            {"type": "text", "text": f"image_path: {resolved_image_path}"}
        )
        input_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        resolved_image_path,
                        max_side=image_max_side,
                        jpeg_quality=image_jpeg_quality,
                    ),
                    "image_path": str(resolved_image_path),
                },
            }
        )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional dental radiology assistant."},
            {"role": "user", "content": input_content},
        ],
    )
    message = completion.choices[0].message
    raw_answer = message.content or ""
    answer = unicodedata.normalize("NFKC", raw_answer)

    return {"answer": answer, "raw_response": completion.model_dump()}


def discover_patients(dataset_root: Path) -> List[Dict[str, Any]]:
    patients: List[Dict[str, Any]] = []
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    for group_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for patient_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            xr_dir = patient_dir / "XR"
            xp_image = xr_dir / "XP.png"
            xray_images = [xp_image]

            patients.append(
                {
                    "group": group_dir.name,
                    "patient_name": patient_dir.name,
                    "patient_dir": patient_dir,
                    "xray_images": xray_images,
                    "xray_ready": all(path.exists() for path in xray_images),
                }
            )
    return patients


def build_unique_patient_uid(
    group: str, patient_name: str, patient_dir: Path, used: Dict[str, int]
) -> str:
    base = f"{group}__{patient_name}"
    if base not in used:
        used[base] = 1
        return base
    used[base] += 1
    short_hash = hashlib.sha1(str(patient_dir).encode("utf-8")).hexdigest()[:8]
    return f"{base}__{short_hash}"


def json_serializable_patient_record(
    uid: str,
    group: str,
    patient_name: str,
    patient_dir: Path,
    model: str,
    api_base: str,
) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "patient_uid": uid,
        "patient_name": patient_name,
        "group": group,
        "source_path": str(patient_dir),
        "created_at_utc": now,
        "updated_at_utc": now,
        "meta": {
            "model": model,
            "api_base": api_base,
        },
        "Modalities": {},
    }


def add_task_result(
    patient_record: Dict[str, Any],
    task_type: str,
    question: str,
    input_images: List[Path],
    status: str,
    answer: str = "",
    error: str = "",
) -> None:
    patient_record["Modalities"][task_type] = {
        "question": question,
        "status": status,
        "input_images": [str(path) for path in input_images],
        "answer": answer,
        "error": error,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    patient_record["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_one_patient(
    patient: Dict[str, Any],
    uid: str,
    output_path: Path,
    args: argparse.Namespace,
    client: OpenAI | None,
) -> Dict[str, Any]:
    group = patient["group"]
    patient_name = patient["patient_name"]
    patient_dir = patient["patient_dir"]

    if output_path.exists() and not args.overwrite:
        return {
            "uid": uid,
            "output_path": output_path,
            "skipped": True,
            "task_status": {},
            "error": "",
        }

    record = json_serializable_patient_record(
        uid=uid,
        group=group,
        patient_name=patient_name,
        patient_dir=patient_dir,
        model=args.model,
        api_base=args.api_base,
    )

    xray_images: List[Path] = patient["xray_images"]
    if patient["xray_ready"]:
        if args.dry_run:
            add_task_result(
                patient_record=record,
                task_type=XRAY_TASK_TYPE,
                question=XRAY_COMPREHENSIVE_REPORT_PROMPT,
                input_images=xray_images,
                status="dry_run",
            )
        else:
            try:
                if client is None:
                    raise RuntimeError("Local agent client is not initialized.")
                result = call_local_agent_vision(
                    client=client,
                    model=args.model,
                    prompt=XRAY_COMPREHENSIVE_REPORT_PROMPT,
                    image_paths=xray_images,
                    image_max_side=args.image_max_side,
                    image_jpeg_quality=args.image_jpeg_quality,
                )
                add_task_result(
                    patient_record=record,
                    task_type=XRAY_TASK_TYPE,
                    question=XRAY_COMPREHENSIVE_REPORT_PROMPT,
                    input_images=xray_images,
                    status="success",
                    answer=result["answer"],
                )
            except Exception as exc:
                add_task_result(
                    patient_record=record,
                    task_type=XRAY_TASK_TYPE,
                    question=XRAY_COMPREHENSIVE_REPORT_PROMPT,
                    input_images=xray_images,
                    status="failed",
                    error=str(exc),
                )
    else:
        add_task_result(
            patient_record=record,
            task_type=XRAY_TASK_TYPE,
            question=XRAY_COMPREHENSIVE_REPORT_PROMPT,
            input_images=xray_images,
            status="missing_input",
            error="Missing required panoramic x-ray file: XR/XP.png.",
        )

    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    task_status = {
        key: value.get("status", "unknown")
        for key, value in record.get("Modalities", {}).items()
    }
    return {
        "uid": uid,
        "output_path": output_path,
        "skipped": False,
        "task_status": task_status,
        "error": "",
    }


def run_pipeline(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    api_key = os.getenv(args.api_key_env, "") or args.api_key
    client = (
        None
        if args.dry_run
        else OpenAI(
            api_key=api_key,
            base_url=args.api_base,
        )
    )

    patients = discover_patients(dataset_root)
    if args.max_patients is not None:
        patients = patients[: args.max_patients]

    used_uid: Dict[str, int] = {}
    total = len(patients)
    workers = max(1, args.workers)
    print(
        f"[INFO] Found {total} patients under {dataset_root}; "
        f"workers={workers}, dry_run={args.dry_run}, api_base={args.api_base}"
    )

    jobs: List[Tuple[Dict[str, Any], str, Path]] = []
    for patient in patients:
        uid = build_unique_patient_uid(
            patient["group"], patient["patient_name"], patient["patient_dir"], used_uid
        )
        output_path = output_dir / f"{uid}.json"
        jobs.append((patient, uid, output_path))

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(process_one_patient, patient, uid, output_path, args, client): (
                uid,
                output_path,
            )
            for patient, uid, output_path in jobs
        }
        for future in as_completed(future_map):
            uid, output_path = future_map[future]
            completed += 1
            progress = (completed / total * 100.0) if total else 100.0
            try:
                result = future.result()
                if result["skipped"]:
                    print(
                        f"[{completed}/{total} {progress:.1f}%] "
                        f"[SKIP] {uid}: output exists ({output_path})"
                    )
                else:
                    statuses = ", ".join(
                        f"{key}:{value}"
                        for key, value in sorted(result["task_status"].items())
                    )
                    print(
                        f"[{completed}/{total} {progress:.1f}%] "
                        f"[DONE] {uid} -> {output_path} | {statuses}"
                    )
            except Exception as exc:
                print(
                    f"[{completed}/{total} {progress:.1f}%] "
                    f"[ERROR] {uid}: {exc}"
                )

    print("[INFO] Pipeline completed.")


if __name__ == "__main__":
    run_pipeline(parse_args())
