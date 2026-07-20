#!/usr/bin/env python3
"""Export PMCID targets from a PubMed JSON produced by pubmed_api.py.

The output CSV can be consumed by pmc_aws_downloader.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def extract_pmcid(article: dict[str, Any]) -> str | None:
    articleids = (article.get("esummary_raw") or {}).get("articleids") or []
    for item in articleids:
        if str(item.get("idtype") or "").lower() in {"pmc", "pmcid"}:
            value = str(item.get("value") or "").strip()
            if value:
                return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PMCID targets from PubMed JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON from pubmed_api.py or screening output.")
    parser.add_argument("--output", required=True, type=Path, help="Output targets CSV for pmc_aws_downloader.py.")
    parser.add_argument("--label", default="dental", help="Group label used as srma_pmid column.")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        payload = json.load(f)
    articles = payload.get("articles") or []
    if not isinstance(articles, list):
        raise ValueError("Input JSON must contain an articles list.")

    rows = []
    for article in articles:
        pmid = str(article.get("pmid") or "").strip()
        pmcid = extract_pmcid(article)
        if not pmid or not pmcid:
            continue
        title = str(article.get("title") or "").replace("\n", " ").strip()
        rows.append(
            {
                "srma_pmid": args.label,
                "study_key": "0",
                "pmid": pmid,
                "pmcid": pmcid,
                "title": title,
                "metadata_path": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["srma_pmid", "study_key", "pmid", "pmcid", "title", "metadata_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input articles: {len(articles)}")
    print(f"PMCID targets: {len(rows)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
