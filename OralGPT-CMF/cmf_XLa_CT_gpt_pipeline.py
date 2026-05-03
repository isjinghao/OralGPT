#!/usr/bin/env python3
"""
CMF dataset -> GPT vision analysis pipeline.

Tasks implemented:
1) X-ray based dento-skeletal analysis
2) CT based craniofacial abnormality analysis

Input dataset structure:
  SH9HCMFdata/
    group1..groupN/
      PATIENT_NAME/
        XLadata.png
        XR/XF.png
        XR/XLa.png
        CT/*.png

Output:
  One JSON per patient, with stable unique id and extensible task records.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


XR_PROMPT = (
    "Based on the lateral and posteroanterior cephalometric radiographs and "
    "measurements, what are the key dento-skeletal characteristics of this patient?"
)

CT_PROMPT = (
    """
    Please analyze the dentofacial deformity based on the provided 3D CT craniofacial reconstruction.

    You may internally utilize prior sagittal information (e.g., from cephalometric analysis) as a reference to guide your understanding of maxillomandibular relationships. 

    Instructions:
    1. Determine the sagittal skeletal relationship (Class I / II / III) based on the relative spatial position of the maxilla and mandible;
    2. Use 3D CT morphology as the primary basis for all descriptions and conclusions;
    3. Ensure that the final sagittal assessment reflects the dominant skeletal morphology and is consistent throughout the analysis;
    4. Evaluate three-dimensional structural features, including asymmetry, transverse discrepancies, and vertical proportions;
    5. Ignore artifacts such as metal fixation, wires, and reconstruction noise;
    6. Focus only on bony structures (maxilla, mandible, orbit, zygoma);
    7. Follow standard orthognathic surgery analysis principles;
    8. Provide objective morphological description with morphology-based clinical inference;
    9. Avoid vague or neutral conclusions when a clear skeletal pattern is present.

    Output should be structured as follows:

    1. Frontal View Analysis:
    - Facial symmetry (presence of deviation)
    - Orbital region (vertical position and symmetry)
    - Zygomatic region (width and projection symmetry)
    - Maxilla (deviation or vertical asymmetry)
    - Mandible (chin deviation, mandibular angle symmetry)

    2. Lateral View Analysis:
    - Sagittal position of the maxilla (protrusion / retrusion / normal)
    - Sagittal position of the mandible (protrusion / retrusion / normal)
    - Maxillomandibular relationship (Class I / II / III)
    - Vertical facial proportions (long face / short face tendency)
    - Chin morphology (retruded / protruded / normal)

    3. Three-Dimensional Summary:
    - Sagittal discrepancies
    - Vertical discrepancies
    - Transverse asymmetry

    Use concise and professional medical terminology. 
    All conclusions must be expressed as morphology-based observations derived from CT data.
    """
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT vision tasks for SH9HCMFdata and save one JSON per patient."
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata",
        help="Path to SH9HCMFdata root.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/patient_json",
        help="Directory to save per-patient JSON files.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name for vision reasoning.",
    )
    parser.add_argument(
        "--api-base",
        default="http://35.164.11.19:3887/v1",
        help="OpenAI compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name that stores API key.",
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
        help="Do not call GPT API; only scan dataset and write skeleton JSON.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for patient-level parallel processing.",
    )
    return parser.parse_args()


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_gpt_vision(
    client: OpenAI,
    model: str,
    prompt: str,
    image_paths: List[Path],
) -> Dict[str, Any]:
    input_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        input_content.append(
            {"type": "image_url", "image_url": {"url": image_to_data_url(p)}}
        )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
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

    for group_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for patient_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            xr_dir = patient_dir / "XR"
            ct_dir = patient_dir / "CT"
            xla_candidates = sorted(patient_dir.glob("XLadata*.png"))
            xla_data = xla_candidates[0] if xla_candidates else (patient_dir / "XLadata.png")
            xr_xf = xr_dir / "XF.png"
            xr_xla = xr_dir / "XLa.png"
            ct_images = sorted(ct_dir.glob("*.png")) if ct_dir.exists() else []

            patients.append(
                {
                    "group": group_dir.name,
                    "patient_name": patient_dir.name,
                    "patient_dir": patient_dir,
                    "xray_images": [xla_data, xr_xf, xr_xla],
                    "ct_images": ct_images,
                    "xray_ready": all(p.exists() for p in [xla_data, xr_xf, xr_xla]),
                    "ct_ready": len(ct_images) > 0,
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
        "input_images": [str(p) for p in input_images],
        "answer": answer,
        "error": error,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    patient_record["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_one_patient(
    p: Dict[str, Any],
    uid: str,
    output_path: Path,
    args: argparse.Namespace,
    client: OpenAI | None,
) -> Dict[str, Any]:
    group = p["group"]
    patient_name = p["patient_name"]
    patient_dir = p["patient_dir"]

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
    )

    # Task 2: XLa-related analysis.
    xray_images: List[Path] = p["xray_images"]
    if p["xray_ready"]:
        if args.dry_run:
            add_task_result(
                patient_record=record,
                task_type="XLa",
                question=XR_PROMPT,
                input_images=xray_images,
                status="dry_run",
            )
        else:
            try:
                res = call_gpt_vision(
                    client=client,
                    model=args.model,
                    prompt=XR_PROMPT,
                    image_paths=xray_images,
                )
                add_task_result(
                    patient_record=record,
                    task_type="XLa",
                    question=XR_PROMPT,
                    input_images=xray_images,
                    status="success",
                    answer=res["answer"],
                )
            except Exception as e:
                add_task_result(
                    patient_record=record,
                    task_type="XLa",
                    question=XR_PROMPT,
                    input_images=xray_images,
                    status="failed",
                    error=str(e),
                )
    else:
        add_task_result(
            patient_record=record,
            task_type="XLa",
            question=XR_PROMPT,
            input_images=xray_images,
            status="missing_input",
            error="Missing one or more required X-ray files.",
        )

    # Task 3: CT analysis.
    ct_images: List[Path] = p["ct_images"]
    if p["ct_ready"]:
        if args.dry_run:
            add_task_result(
                patient_record=record,
                task_type="CT",
                question=CT_PROMPT,
                input_images=ct_images,
                status="dry_run",
            )
        else:
            try:
                res = call_gpt_vision(
                    client=client,
                    model=args.model,
                    prompt=CT_PROMPT,
                    image_paths=ct_images,
                )
                add_task_result(
                    patient_record=record,
                    task_type="CT",
                    question=CT_PROMPT,
                    input_images=ct_images,
                    status="success",
                    answer=res["answer"],
                )
            except Exception as e:
                add_task_result(
                    patient_record=record,
                    task_type="CT",
                    question=CT_PROMPT,
                    input_images=ct_images,
                    status="failed",
                    error=str(e),
                )
    else:
        add_task_result(
            patient_record=record,
            task_type="CT",
            question=CT_PROMPT,
            input_images=ct_images,
            status="missing_input",
            error="No CT images found under CT/ folder.",
        )

    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    task_status = {
        k: v.get("status", "unknown") for k, v in record.get("Modalities", {}).items()
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

    api_key = os.getenv(args.api_key_env, "")
    if (not args.dry_run) and (not api_key):
        raise RuntimeError(
            f"Missing API key. Set environment variable: {args.api_key_env}"
        )
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
        f"workers={workers}, dry_run={args.dry_run}"
    )

    jobs: List[Tuple[Dict[str, Any], str, Path]] = []
    for p in patients:
        uid = build_unique_patient_uid(
            p["group"], p["patient_name"], p["patient_dir"], used_uid
        )
        output_path = output_dir / f"{uid}.json"
        jobs.append((p, uid, output_path))

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(process_one_patient, p, uid, out, args, client): (
                uid,
                out,
            )
            for p, uid, out in jobs
        }
        for future in as_completed(future_map):
            uid, out = future_map[future]
            completed += 1
            progress = (completed / total * 100.0) if total else 100.0
            try:
                res = future.result()
                if res["skipped"]:
                    print(
                        f"[{completed}/{total} {progress:.1f}%] "
                        f"[SKIP] {uid}: output exists ({out})"
                    )
                else:
                    statuses = ", ".join(
                        f"{k}:{v}" for k, v in sorted(res["task_status"].items())
                    )
                    print(
                        f"[{completed}/{total} {progress:.1f}%] "
                        f"[DONE] {uid} -> {out} | {statuses}"
                    )
            except Exception as e:
                print(
                    f"[{completed}/{total} {progress:.1f}%] "
                    f"[ERROR] {uid}: {e}"
                )

    print("[INFO] Pipeline completed.")


if __name__ == "__main__":
    run_pipeline(parse_args())
