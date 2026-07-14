#!/usr/bin/env python3
"""Explore and download PMC full text from the public PMC AWS dataset.

The script reads the SRMA rows from pmcid_top100.csv, extracts included-study
PMIDs/PMCIDs from local metadata JSON files, discovers article-version objects
in the pmc-oa-opendata S3 bucket, and downloads available TXT/XML/JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_TARGETS_CSV = SCRIPT_DIR / "pmcid_targets.csv"
DEFAULT_PMCID_CSV = SCRIPT_DIR / "pmcid_top100.csv"
DEFAULT_METADATA_DIR = SCRIPT_DIR / "metadata" / "srma"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "full_text"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
BUCKET = "pmc-oa-opendata"
BUCKET_HTTPS_ROOT = f"https://{BUCKET}.s3.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class StudyTarget:
    srma_pmid: str
    study_key: str
    pmid: str
    pmcid: str
    title: str
    metadata_path: pathlib.Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_pmcid(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    match = re.search(r"(?:PMC)?(\d+)", str(raw_value), flags=re.I)
    return f"PMC{match.group(1)}" if match else None


def normalize_pmid(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def iter_included_studies(metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    included = metadata.get("included_studies", {})
    if isinstance(included, dict):
        return [(str(k), v) for k, v in included.items() if isinstance(v, dict)]
    if isinstance(included, list):
        return [(str(i), v) for i, v in enumerate(included) if isinstance(v, dict)]
    return []


def metadata_paths_from_csv(
    csv_path: pathlib.Path,
    metadata_dir: pathlib.Path,
    selected_srmas: set[str] | None,
) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_path = row.get("file_path")
            if not raw_path:
                continue
            csv_path_value = pathlib.Path(raw_path)
            local_path = metadata_dir / csv_path_value.name
            path = local_path if local_path.exists() else csv_path_value
            if selected_srmas and path.stem not in selected_srmas:
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def collect_targets(
    csv_path: pathlib.Path,
    metadata_dir: pathlib.Path,
    selected_srmas: set[str] | None,
    limit_srmas: int | None,
    limit_studies: int | None,
) -> list[StudyTarget]:
    paths = metadata_paths_from_csv(csv_path, metadata_dir, selected_srmas)
    if limit_srmas is not None:
        paths = paths[:limit_srmas]

    targets: list[StudyTarget] = []
    for metadata_path in paths:
        metadata = load_json(metadata_path)
        if not isinstance(metadata, dict):
            continue
        srma_pmid = str(metadata.get("srma_pmid") or metadata_path.stem)
        studies = iter_included_studies(metadata)
        if limit_studies is not None:
            studies = studies[:limit_studies]
        for study_key, study in studies:
            pubmed = study.get("pubmed")
            pubmed = pubmed if isinstance(pubmed, dict) else {}
            pmid = normalize_pmid(pubmed.get("pmid"))
            pmcid = normalize_pmcid(pubmed.get("pmcid"))
            if not pmid or not pmcid:
                continue
            title = str(study.get("title") or pubmed.get("matched_title") or pubmed.get("title") or "")
            targets.append(
                StudyTarget(
                    srma_pmid=srma_pmid,
                    study_key=study_key,
                    pmid=pmid,
                    pmcid=pmcid,
                    title=title,
                    metadata_path=metadata_path,
                )
            )
    return targets


def collect_targets_from_targets_csv(
    targets_csv: pathlib.Path,
    selected_srmas: set[str] | None,
    limit_targets: int | None,
) -> list[StudyTarget]:
    targets: list[StudyTarget] = []
    with targets_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            srma_pmid = str(row.get("srma_pmid") or "").strip()
            if selected_srmas and srma_pmid not in selected_srmas:
                continue
            pmid = normalize_pmid(row.get("pmid"))
            pmcid = normalize_pmcid(row.get("pmcid"))
            if not srma_pmid or not pmid or not pmcid:
                continue
            targets.append(
                StudyTarget(
                    srma_pmid=srma_pmid,
                    study_key=str(row.get("study_key") or ""),
                    pmid=pmid,
                    pmcid=pmcid,
                    title=str(row.get("title") or ""),
                    metadata_path=pathlib.Path(str(row.get("metadata_path") or "")),
                )
            )
            if limit_targets is not None and len(targets) >= limit_targets:
                break
    return targets


def http_get(url: str, *, timeout: float, retries: int, sleep_seconds: float) -> bytes:
    last_error: Exception | None = None
    headers = {"User-Agent": "oral-research-pmc-aws-downloader/1.0"}
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def list_s3_keys_for_pmcid(pmcid: str, *, timeout: float, retries: int, sleep_seconds: float) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "list-type": "2",
            "prefix": f"{pmcid}.",
            "max-keys": "1000",
        }
    )
    url = f"{BUCKET_HTTPS_ROOT}/?{params}"
    payload = http_get(url, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
    root = ET.fromstring(payload)
    keys: list[dict[str, Any]] = []
    for item in root.findall("s3:Contents", S3_NS):
        key = item.findtext("s3:Key", namespaces=S3_NS)
        size = item.findtext("s3:Size", namespaces=S3_NS)
        last_modified = item.findtext("s3:LastModified", namespaces=S3_NS)
        if key:
            keys.append(
                {
                    "key": key,
                    "size": int(size or 0),
                    "last_modified": last_modified or "",
                }
            )
    return keys


def group_article_versions(keys: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    versions: dict[str, dict[str, dict[str, Any]]] = {}
    for item in keys:
        key = item["key"]
        parts = key.split("/", 1)
        if len(parts) != 2:
            continue
        version_id, filename = parts
        suffix = pathlib.Path(filename).suffix.lower().lstrip(".")
        if suffix in {"txt", "xml", "json", "pdf"} and filename.startswith(version_id + "."):
            versions.setdefault(version_id, {})[suffix] = item
    return versions


def version_sort_key(version_id: str) -> tuple[int, str]:
    match = re.search(r"\.(\d+)$", version_id)
    return (int(match.group(1)) if match else -1, version_id)


def select_best_version(versions: dict[str, dict[str, dict[str, Any]]]) -> tuple[str | None, dict[str, dict[str, Any]]]:
    candidates = [
        (version_id, files)
        for version_id, files in versions.items()
        if "txt" in files or "xml" in files
    ]
    if not candidates:
        return None, {}
    version_id, files = max(candidates, key=lambda pair: version_sort_key(pair[0]))
    return version_id, files


def s3_key_to_https_url(key: str) -> str:
    return f"{BUCKET_HTTPS_ROOT}/{urllib.parse.quote(key, safe='/')}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:120] or "unknown"


def copy_or_download(
    *,
    key: str,
    cache_path: pathlib.Path,
    output_path: pathlib.Path,
    timeout: float,
    retries: int,
    sleep_seconds: float,
    overwrite: bool,
) -> str:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return "skipped_existing"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() or cache_path.stat().st_size == 0 or overwrite:
        data = http_get(s3_key_to_https_url(key), timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(cache_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.resolve() != output_path.resolve():
        shutil.copy2(cache_path, output_path)
    return "downloaded"


def write_targets_csv(path: pathlib.Path, targets: list[StudyTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["srma_pmid", "study_key", "pmid", "pmcid", "title", "metadata_path"],
        )
        writer.writeheader()
        for target in targets:
            writer.writerow(
                {
                    "srma_pmid": target.srma_pmid,
                    "study_key": target.study_key,
                    "pmid": target.pmid,
                    "pmcid": target.pmcid,
                    "title": target.title,
                    "metadata_path": str(target.metadata_path),
                }
            )


def write_manifest_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "srma_pmid",
        "study_key",
        "pmid",
        "pmcid",
        "aws_status",
        "selected_version",
        "available_versions",
        "available_formats",
        "downloaded_formats",
        "error",
        "output_dir",
        "title",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PMC full text from the public PMC AWS dataset.")
    parser.add_argument("srma_pmids", nargs="*", help="Optional SRMA PMIDs to process.")
    parser.add_argument("--targets-csv", type=pathlib.Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument("--pmcid-csv", type=pathlib.Path, default=DEFAULT_PMCID_CSV)
    parser.add_argument("--metadata-dir", type=pathlib.Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=pathlib.Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=SCRIPT_DIR / "cache")
    parser.add_argument("--formats", nargs="+", default=["txt", "xml", "json"], choices=["txt", "xml", "json", "pdf"])
    parser.add_argument("--discover-only", action="store_true", help="Only discover AWS availability; do not download files.")
    parser.add_argument("--limit-srmas", type=int, default=None)
    parser.add_argument("--limit-studies", type=int, default=None)
    parser.add_argument("--limit-targets", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between targets and retry backoff base.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_srmas = {str(v).strip() for v in args.srma_pmids if str(v).strip()} or None
    if args.targets_csv and args.targets_csv.exists():
        targets = collect_targets_from_targets_csv(args.targets_csv, selected_srmas, args.limit_targets)
    else:
        targets = collect_targets(
            args.pmcid_csv,
            args.metadata_dir,
            selected_srmas,
            args.limit_srmas,
            args.limit_studies,
        )
        if args.limit_targets is not None:
            targets = targets[: args.limit_targets]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_targets_csv(args.report_dir / "pmcid_targets.csv", targets)

    discovery_cache: dict[str, tuple[str | None, dict[str, dict[str, Any]], str | None]] = {}
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    total = len(targets)
    progress_enabled = tqdm is not None and not args.no_progress
    progress = tqdm(
        targets,
        total=total,
        unit="study",
        dynamic_ncols=True,
        disable=not progress_enabled,
        file=sys.stderr,
    )

    for index, target in enumerate(progress, start=1):
        if progress_enabled:
            progress.set_description(f"{target.srma_pmid}/{target.pmid} {target.pmcid}")
        else:
            print(f"[{index}/{total}] {target.srma_pmid}/{target.pmid} {target.pmcid}", flush=True)
        error = None
        selected_version: str | None = None
        selected_files: dict[str, dict[str, Any]] = {}
        versions: dict[str, dict[str, dict[str, Any]]] = {}
        aws_status = "not_checked"
        downloaded_formats: list[str] = []

        try:
            if target.pmcid not in discovery_cache:
                keys = list_s3_keys_for_pmcid(
                    target.pmcid,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep_seconds=args.sleep,
                )
                versions = group_article_versions(keys)
                selected_version, selected_files = select_best_version(versions)
                discovery_cache[target.pmcid] = (selected_version, selected_files, None)
            else:
                selected_version, selected_files, cached_error = discovery_cache[target.pmcid]
                if cached_error:
                    raise RuntimeError(cached_error)

            if not selected_version:
                aws_status = "not_found_or_no_full_text"
            else:
                aws_status = "available"
                output_dir = args.output_root / safe_name(target.srma_pmid) / safe_name(target.pmid)
                cache_dir = args.cache_dir / safe_name(target.pmcid) / safe_name(selected_version)
                if not args.discover_only:
                    for fmt in args.formats:
                        item = selected_files.get(fmt)
                        if not item:
                            continue
                        output_path = output_dir / f"{target.pmcid}.{selected_version}.{fmt}"
                        cache_path = cache_dir / f"{selected_version}.{fmt}"
                        result = copy_or_download(
                            key=item["key"],
                            cache_path=cache_path,
                            output_path=output_path,
                            timeout=args.timeout,
                            retries=args.retries,
                            sleep_seconds=args.sleep,
                            overwrite=args.overwrite,
                        )
                        downloaded_formats.append(fmt if result == "downloaded" else f"{fmt}:existing")
                output_dir.mkdir(parents=True, exist_ok=True)
                provenance = {
                    "created_at": utc_now_iso(),
                    "srma_pmid": target.srma_pmid,
                    "study_key": target.study_key,
                    "pmid": target.pmid,
                    "pmcid": target.pmcid,
                    "selected_version": selected_version,
                    "source_bucket": BUCKET,
                    "available_files": {
                        fmt: {
                            "key": item["key"],
                            "size": item["size"],
                            "url": s3_key_to_https_url(item["key"]),
                        }
                        for fmt, item in selected_files.items()
                    },
                }
                write_json(output_dir / "pmc_aws_provenance.json", provenance)
        except Exception as exc:
            error = str(exc)
            aws_status = "error"
            discovery_cache[target.pmcid] = (None, {}, error)

        counts[aws_status] += 1
        if progress_enabled:
            progress.set_postfix(
                available=counts.get("available", 0),
                missing=counts.get("not_found_or_no_full_text", 0),
                errors=counts.get("error", 0),
            )
        available_versions = sorted({key.split("/", 1)[0] for key in [item["key"] for item in selected_files.values()]})
        rows.append(
            {
                "srma_pmid": target.srma_pmid,
                "study_key": target.study_key,
                "pmid": target.pmid,
                "pmcid": target.pmcid,
                "aws_status": aws_status,
                "selected_version": selected_version or "",
                "available_versions": ";".join(available_versions),
                "available_formats": ";".join(sorted(selected_files)),
                "downloaded_formats": ";".join(downloaded_formats),
                "error": error or "",
                "output_dir": str(args.output_root / safe_name(target.srma_pmid) / safe_name(target.pmid)),
                "title": target.title,
            }
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    if progress_enabled:
        progress.close()

    write_manifest_csv(args.report_dir / "pmc_aws_manifest.csv", rows)
    summary = {
        "created_at": utc_now_iso(),
        "target_rows": len(targets),
        "unique_pmcids": len({target.pmcid for target in targets}),
        "counts_by_status": dict(counts),
        "formats_requested": args.formats,
        "discover_only": args.discover_only,
        "output_root": str(args.output_root),
        "report_dir": str(args.report_dir),
        "aws_bucket": BUCKET,
    }
    write_json(args.report_dir / "pmc_aws_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
