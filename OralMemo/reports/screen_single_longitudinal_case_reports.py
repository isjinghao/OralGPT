#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Second-stage screening for PubMed JSON exported by pubmed_api.py.

Default goal:
    Keep likely oral/dental single-patient longitudinal case reports.

The output JSON keeps the same broad structure as the input:
    query / api_key_used / record_count / articles
Only record_count and articles are changed by default.
"""

import argparse
import copy
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


BAD_PUBTYPES = {
    "Review",
    "Systematic Review",
    "Scoping Review",
    "Meta-Analysis",
}

REVIEW_RX = re.compile(
    r"\b("
    r"systematic review|scoping review|meta-analysis|"
    r"literature review|review of (?:the )?literature|review"
    r")\b",
    re.I,
)

CASE_SERIES_RX = re.compile(
    r"\bcase series\b|"
    r"\bcase-series\b|"
    r"\bseries of\b|"
    r"\breport(?:s|ed)? of (?:two|three|four|five|six|seven|eight|nine|ten|\d+) cases\b|"
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+) cases\b",
    re.I,
)

IN_VITRO_ANIMAL_RX = re.compile(
    r"\bin vitro\b|\brats?\b|\bmice\b|\bmouse\b|\banimal\b|\bdogs?\b|\brabbit\b",
    re.I,
)

SINGLE_PATIENT_RX = re.compile(
    r"\b(?:a|an|one)\s+"
    r"(?:\d{1,3}[- ]year[- ]old\s+)?"
    r"(?:male|female|man|woman|boy|girl|patient|adult|child|infant|adolescent)\b|"
    r"\bwe report (?:a|one) case\b|"
    r"\bthis case report\b|"
    r"\ba case report\b|"
    r"\ba rare case\b",
    re.I,
)

LONGITUDINAL_RX = re.compile(
    r"\bfollow[- ]?up\b|"
    r"\bfollowed(?: up)?\b|"
    r"\brecall\b|"
    r"\blong[- ]term\b|"
    r"\bpost[- ]?(?:operative|treatment)\b|"
    r"\bafter\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|several)\s+"
    r"(?:days?|weeks?|months?|years?)\b|"
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)[- ]?"
    r"(?:day|week|month|year)s?[- ]?(?:follow[- ]?up|recall|period)\b|"
    r"\bover\s+(?:a|the)?\s*"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|several)\s+"
    r"(?:days?|weeks?|months?|years?)\b",
    re.I,
)

DURATION_RX = re.compile(
    r"(?:after|at|for|over|during|following|followed(?: up)? for)\s+"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|several)\s+"
    r"(?:days?|weeks?|months?|years?)|"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)[- ]?"
    r"(?:day|week|month|year)s?[- ]?(?:follow[- ]?up|recall|period)|"
    r"\b\d+\s*(?:days?|weeks?|months?|years?)\s+later\b",
    re.I,
)

STRONG_PLURAL_RX = re.compile(
    r"\bcase series\b|"
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+) cases\b|"
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+) patients\b",
    re.I,
)

REPEATED_TIME_RX = re.compile(
    r"\bserial\b|\bperiodic\b|\bover time\b|\bbaseline\b|\bsequential\b",
    re.I,
)


def article_text(article: Dict[str, Any]) -> str:
    title = article.get("title") or ""
    abstract = article.get("abstract") or ""
    first_sentence = article.get("abstract_first_sentence") or ""
    return f"{title} {abstract} {first_sentence}".replace("\xa0", " ")


def article_pubtypes(article: Dict[str, Any]) -> List[str]:
    raw = article.get("esummary_raw") or {}
    pubtypes = raw.get("pubtype") or []
    return [str(p) for p in pubtypes]


def screen_article(article: Dict[str, Any]) -> Dict[str, Any]:
    text = article_text(article)
    pubtypes = set(article_pubtypes(article))
    duration_mentions = [m.group(0) for m in DURATION_RX.finditer(text)]

    flags: List[str] = []
    if pubtypes & BAD_PUBTYPES:
        flags.append("exclude_pubtype_review")
    if REVIEW_RX.search(text):
        flags.append("exclude_text_review")
    if CASE_SERIES_RX.search(text):
        flags.append("exclude_case_series_or_multi_cases")
    if IN_VITRO_ANIMAL_RX.search(text):
        flags.append("exclude_in_vitro_or_animal")
    if SINGLE_PATIENT_RX.search(text):
        flags.append("single_patient_signal")
    if LONGITUDINAL_RX.search(text):
        flags.append("longitudinal_signal")
    if duration_mentions:
        flags.append("explicit_duration")
    if STRONG_PLURAL_RX.search(text):
        flags.append("strong_plural_signal")
    if REPEATED_TIME_RX.search(text):
        flags.append("multi_time_signal")

    excluded = any(flag.startswith("exclude_") for flag in flags)
    candidate = (
        not excluded
        and "single_patient_signal" in flags
        and "longitudinal_signal" in flags
    )
    strict = (
        candidate
        and "explicit_duration" in flags
        and "strong_plural_signal" not in flags
    )
    multi_time = strict and (
        len(duration_mentions) >= 2
        or "multi_time_signal" in flags
    )

    return {
        "candidate": candidate,
        "strict": strict,
        "multi_time": multi_time,
        "flags": flags,
        "duration_mentions": duration_mentions,
    }


def keep_for_mode(screen: Dict[str, Any], mode: str) -> bool:
    if mode == "candidate":
        return bool(screen["candidate"])
    if mode == "strict":
        return bool(screen["strict"])
    if mode == "multi-time":
        return bool(screen["multi_time"])
    raise ValueError(f"Unknown mode: {mode}")


def add_screening_fields(article: Dict[str, Any], screen: Dict[str, Any], mode: str) -> Dict[str, Any]:
    item = copy.deepcopy(article)
    item["screening"] = {
        "mode": mode,
        "candidate": screen["candidate"],
        "strict": screen["strict"],
        "multi_time": screen["multi_time"],
        "flags": screen["flags"],
        "duration_mentions": screen["duration_mentions"],
    }
    return item


def write_csv(path: Path, articles: List[Dict[str, Any]], screens: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pmid",
                "title",
                "journal",
                "pubdate",
                "doi",
                "pubmed_url",
                "flags",
                "duration_mentions",
                "abstract_first_sentence",
            ],
        )
        writer.writeheader()
        for article in articles:
            pmid = str(article.get("pmid", ""))
            screen = screens.get(pmid, {})
            writer.writerow(
                {
                    "pmid": pmid,
                    "title": article.get("title", ""),
                    "journal": article.get("journal", ""),
                    "pubdate": article.get("pubdate", ""),
                    "doi": article.get("doi", ""),
                    "pubmed_url": article.get("pubmed_url", ""),
                    "flags": "; ".join(screen.get("flags", [])),
                    "duration_mentions": "; ".join(screen.get("duration_mentions", [])),
                    "abstract_first_sentence": article.get("abstract_first_sentence", ""),
                }
            )


def screening_sort_key(article: Dict[str, Any], screens: Dict[str, Dict[str, Any]]) -> tuple:
    pmid = str(article.get("pmid", ""))
    screen = screens.get(pmid, {})
    title = str(article.get("title") or "")
    title_has_followup = bool(re.search(r"\bfollow[- ]?up\b|\blong[- ]term\b", title, re.I))
    title_has_case_report = bool(re.search(r"\bcase report\b", title, re.I))
    duration_count = len(screen.get("duration_mentions", []))
    flags = set(screen.get("flags", []))
    return (
        -int(title_has_followup),
        -int(title_has_case_report),
        -duration_count,
        -int("multi_time_signal" in flags),
        str(article.get("journal") or ""),
        title.lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen PubMed JSON for likely single-patient longitudinal oral/dental case reports."
    )
    parser.add_argument("--input", default="reports/dental_test.json", help="Input JSON from pubmed_api.py.")
    parser.add_argument("--output", default="reports/dental_test_single_longitudinal_filtered.json", help="Output JSON.")
    parser.add_argument(
        "--mode",
        choices=["candidate", "strict", "multi-time"],
        default="multi-time",
        help=(
            "candidate: broad single-case + longitudinal signal; "
            "strict: also requires explicit duration; "
            "multi-time: strict plus repeated/multiple-time signal."
        ),
    )
    parser.add_argument(
        "--with-screening-fields",
        action="store_true",
        help="Add a per-article screening object. Off by default to keep article format unchanged.",
    )
    parser.add_argument(
        "--keep-input-order",
        action="store_true",
        help="Keep PubMed/input order. By default, higher-confidence longitudinal cases are listed first.",
    )
    parser.add_argument("--csv-output", default=None, help="Optional CSV summary for manual review.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    articles = payload.get("articles") or []
    if not isinstance(articles, list):
        raise ValueError("Input JSON must contain an articles list.")

    kept: List[Dict[str, Any]] = []
    kept_screens: Dict[str, Dict[str, Any]] = {}
    counts = {
        "total": len(articles),
        "candidate": 0,
        "strict": 0,
        "multi-time": 0,
    }

    for article in articles:
        screen = screen_article(article)
        if screen["candidate"]:
            counts["candidate"] += 1
        if screen["strict"]:
            counts["strict"] += 1
        if screen["multi_time"]:
            counts["multi-time"] += 1

        if keep_for_mode(screen, args.mode):
            item = (
                add_screening_fields(article, screen, args.mode)
                if args.with_screening_fields
                else copy.deepcopy(article)
            )
            kept.append(item)
            kept_screens[str(article.get("pmid", ""))] = screen

    if not args.keep_input_order:
        kept.sort(key=lambda article: screening_sort_key(article, kept_screens))

    out_payload = copy.deepcopy(payload)
    out_payload["record_count"] = len(kept)
    out_payload["articles"] = kept

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    if args.csv_output:
        write_csv(Path(args.csv_output), kept, kept_screens)

    print(f"Input records: {counts['total']}")
    print(f"Broad candidate records: {counts['candidate']}")
    print(f"Strict duration records: {counts['strict']}")
    print(f"Multi-time records: {counts['multi-time']}")
    print(f"Selected mode: {args.mode}")
    print(f"Output records: {len(kept)}")
    print(f"Saved JSON to: {output_path}")
    if args.csv_output:
        print(f"Saved CSV to: {args.csv_output}")


if __name__ == "__main__":
    main()
