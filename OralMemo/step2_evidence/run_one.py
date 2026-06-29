from __future__ import annotations

import argparse

from bench.config import get_settings
from bench.step1_patient_trajectory.dataset import index_by_patient_id, load_dataset
from bench.step2_evidence.pipeline import build_client, process_patient


DEFAULT_PATIENT_ID = "group1__CHENFANG"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_id", default=DEFAULT_PATIENT_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    index = index_by_patient_id(load_dataset(settings.dataset_json))
    item = index.get(args.patient_id)
    if item is None:
        raise SystemExit(f"patient not found: {args.patient_id}")

    client = build_client(settings)
    process_patient(item, settings, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
