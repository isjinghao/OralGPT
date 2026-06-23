#!/usr/bin/env python3
"""Aggregate metric JSONL files into benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from path_utils import bench_root, resolve_path
from summarize.registry import BENCHMARK_REGISTRY


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(rows), "metrics": {}}
    for metric in metric_names:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        if not values:
            continue
        summary["metrics"][metric] = {
            "mean": float(mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summary


def group_summary(
    rows: list[dict[str, Any]],
    group_key: str,
    metric_names: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_key, "unknown"))
        grouped[key].append(row)
    return {key: summarize_rows(items, metric_names) for key, items in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=str, required=True, choices=sorted(BENCHMARK_REGISTRY))
    parser.add_argument("--metrics_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = BENCHMARK_REGISTRY[args.benchmark]
    metrics_jsonl = resolve_path(args.metrics_jsonl)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(metrics_jsonl)
    overall = summarize_rows(rows, cfg["metrics"])
    by_group = {
        group_key: group_summary(rows, group_key, cfg["metrics"]) for group_key in cfg["groups"]
    }

    payload = {
        "benchmark": args.benchmark,
        "num_samples": len(rows),
        "primary_metrics": cfg["primary"],
        "overall": overall,
        "by_group": by_group,
        "metrics_jsonl": str(metrics_jsonl),
    }

    summary_path = output_dir / f"{args.benchmark}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    leaderboard_path = output_dir / "leaderboard.csv"
    write_header = not leaderboard_path.exists()
    with leaderboard_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["benchmark", "num_samples", "ssim", "psnr", "nmi", "mae", "lpips"])
        metrics = overall.get("metrics", {})
        writer.writerow(
            [
                args.benchmark,
                overall.get("count", 0),
                metrics.get("ssim", {}).get("mean", ""),
                metrics.get("psnr", {}).get("mean", ""),
                metrics.get("nmi", {}).get("mean", ""),
                metrics.get("mae", {}).get("mean", ""),
                metrics.get("lpips", {}).get("mean", ""),
            ]
        )

    print(json.dumps(payload["overall"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
