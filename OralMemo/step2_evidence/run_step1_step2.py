from __future__ import annotations

import argparse

from batch_utils import add_batch_arguments, log, patient_output_root, run_patient_batch, selected_patients
from config import get_settings
from step2_evidence.pipeline import build_client, process_patient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step1 trajectories and Step2 evidence graphs")
    add_batch_arguments(parser)
    parser.add_argument("--stage-workers", type=int, default=2)
    return parser.parse_args()


def completed(out) -> bool:
    required = (
        out / "trajectories" / "standard_trajectory.json",
        out / "trajectories" / "short_noisy" / "short_noisy.json",
        out / "trajectories" / "medium_noisy" / "medium_noisy.json",
        out / "trajectories" / "long_noisy" / "long_noisy.json",
        out / "evidence" / "evidence.json",
        out / "graph" / "evidence_graph.json",
        out / "graph" / "evidence_graph.html",
    )
    return all(path.is_file() for path in required)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    patients = selected_patients(settings.dataset_json, args.all, args.limit)

    def run_patient(item: dict) -> str:
        patient_id = item["id"]
        out = patient_output_root(settings.bench_root, patient_id)
        if not args.force and completed(out):
            log(f"[benchmark][{patient_id}][step1-step2/resume] completed outputs found; skipped")
            return "skipped"
        client = build_client(settings, patient_id)
        process_patient(item, settings, client, args.stage_workers)
        return "completed"

    return run_patient_batch(patients, args.num_workers, "benchmark", run_patient)


if __name__ == "__main__":
    raise SystemExit(main())
