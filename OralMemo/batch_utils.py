from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from step1_patient_trajectory.dataset import load_dataset


_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    with _LOG_LOCK:
        print(message, flush=True)


def patient_output_root(bench_root: Path, patient_id: str) -> Path:
    group, name = patient_id.split("__", 1)
    return bench_root / "outputs" / group / name


def add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Run every patient in dataset order")
    selection.add_argument("--limit", type=int, help="Run the first N patients in dataset order")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of patients processed concurrently")
    parser.add_argument("--force", action="store_true", help="Run patients even when final outputs exist")


def selected_patients(dataset_json: Path, run_all: bool, limit: int | None) -> list[dict]:
    patients = load_dataset(dataset_json)
    if run_all:
        return patients
    if limit is None or limit < 1:
        raise ValueError("--limit must be a positive integer")
    return patients[:limit]


def selected_reports(pdf_dir: Path, run_all: bool, limit: int | None) -> list[dict]:
    reports = [
        {"id": f"report__{pdf_path.stem}", "name": pdf_path.stem, "pdf_path": pdf_path}
        for pdf_path in sorted(pdf_dir.glob("*.pdf"))
    ]
    if run_all:
        return reports
    if limit is None or limit < 1:
        raise ValueError("--limit must be a positive integer")
    return reports[:limit]


def run_patient_batch(
    patients: list[dict],
    num_workers: int,
    workflow: str,
    worker: Callable[[dict], str],
) -> int:
    if num_workers < 1:
        raise ValueError("--num-workers must be a positive integer")
    log(f"[{workflow}][batch][start] patients={len(patients)} num_workers={num_workers}")
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker, item): item["id"] for item in patients}
        for future in as_completed(futures):
            patient_id = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                counts["failed"] += 1
                log(f"[{workflow}][{patient_id}][error] {type(exc).__name__}: {exc}")
            else:
                counts[status] += 1
    log(
        f"[{workflow}][batch][done] completed={counts['completed']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0
