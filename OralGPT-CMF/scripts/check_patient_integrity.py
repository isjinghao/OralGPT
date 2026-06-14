#!/usr/bin/env python3
"""
Check patient data integrity by strict structure match.

Reference patient example:
  /data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata/group9/JIANGJUCHENG

The checker enforces:
  - Folder names match exactly
  - File names match exactly
  - File counts match exactly (implicitly via name sets)
  - Paths are compared relative to the patient root
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Dict, Iterable, List, Set


REFERENCE_DEFAULT = "/data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata/group9/JIANGJUCHENG"
DATASET_ROOT_DEFAULT = "/data/OralGPT/OralGPT-CMF/dataset/SH9HCMFdata"

_TEXTRECORD_RE = re.compile(r"^textrecord_.+\.xlsx$")
_TMJ_PNG_RE = re.compile(r"^TMJ/[^/]+\.png$")


def _is_ignored_name(name: str) -> bool:
    # Ignore common filesystem noise but keep everything else strict.
    return name in {".DS_Store", "Thumbs.db"} or name.startswith("._")


def _normalize_rel_file_path(rel_posix: str) -> str:
    # Allow per-patient textrecord naming as equivalent:
    #   textrecord_<PATIENT>.xlsx  -> textrecord_*.xlsx
    # This is only applied to top-level files (no slash).
    if "/" not in rel_posix and _TEXTRECORD_RE.match(rel_posix):
        return "textrecord_*.xlsx"
    return rel_posix


def _apply_equivalence_rules(files: "Counter[str]") -> "Counter[str]":
    """
    Apply equivalence transformations so different-but-acceptable naming variants match.

    Rules:
      - CT/CTF.png  ≡  (CT/CTF1.png + CT/CTF2.png)
        Canonicalized into a single token: CT/CTF{1,2}.png
      - TMJ/*.png: as long as TMJ folder contains >=1 png, treat as OK
        Canonicalized into: TMJ/*.png (count=1)
    """

    out: "Counter[str]" = collections.Counter(files)

    canonical = "CT/CTF{1,2}.png"

    # Case A: single-file form exists
    while out.get("CT/CTF.png", 0) > 0:
        out["CT/CTF.png"] -= 1
        if out["CT/CTF.png"] <= 0:
            out.pop("CT/CTF.png", None)
        out[canonical] += 1

    # Case B: two-file form exists
    while out.get("CT/CTF1.png", 0) > 0 and out.get("CT/CTF2.png", 0) > 0:
        out["CT/CTF1.png"] -= 1
        out["CT/CTF2.png"] -= 1
        if out["CT/CTF1.png"] <= 0:
            out.pop("CT/CTF1.png", None)
        if out["CT/CTF2.png"] <= 0:
            out.pop("CT/CTF2.png", None)
        out[canonical] += 1

    # TMJ: only require that at least one png exists under TMJ/
    tmj_png_count = 0
    for k, v in list(out.items()):
        if _TMJ_PNG_RE.match(k):
            tmj_png_count += v
            out.pop(k, None)
    if tmj_png_count > 0:
        out["TMJ/*.png"] += 1

    return out


@dataclass(frozen=True)
class PatientManifest:
    # relative directory paths (posix style), excluding "."
    dirs: Set[str]
    # relative file paths (posix style) with normalization; keep counts for safety
    files: Counter[str]

    @staticmethod
    def from_patient_dir(patient_dir: Path) -> "PatientManifest":
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {patient_dir}")

        dirs: Set[str] = set()
        files: Counter[str] = collections.Counter()

        # rglob includes nested dirs/files; convert to posix relative strings.
        for p in patient_dir.rglob("*"):
            rel = p.relative_to(patient_dir)
            if rel.parts and any(_is_ignored_name(part) for part in rel.parts):
                continue

            rel_s = rel.as_posix()
            if p.is_dir():
                if rel_s != ".":
                    dirs.add(rel_s)
            elif p.is_file():
                files[_normalize_rel_file_path(rel_s)] += 1
            else:
                # Ignore special files (symlinks, sockets, etc.) to avoid surprises.
                # If you want to enforce symlinks too, change this behavior.
                continue

        files = _apply_equivalence_rules(files)
        return PatientManifest(dirs=dirs, files=files)


@dataclass(frozen=True)
class DiffResult:
    missing_dirs: List[str]
    extra_dirs: List[str]
    missing_files: List[str]
    extra_files: List[str]

    @property
    def ok(self) -> bool:
        return not (self.missing_dirs or self.extra_dirs or self.missing_files or self.extra_files)


def diff_manifests(ref: PatientManifest, cur: PatientManifest) -> DiffResult:
    missing_dirs = sorted(ref.dirs - cur.dirs)
    extra_dirs = sorted(cur.dirs - ref.dirs)
    missing_files: List[str] = []
    extra_files: List[str] = []

    for k in sorted(set(ref.files.keys()) | set(cur.files.keys())):
        r = ref.files.get(k, 0)
        c = cur.files.get(k, 0)
        if c < r:
            missing_files.extend([k] * (r - c))
        elif c > r:
            extra_files.extend([k] * (c - r))
    return DiffResult(
        missing_dirs=missing_dirs,
        extra_dirs=extra_dirs,
        missing_files=missing_files,
        extra_files=extra_files,
    )


def iter_patient_dirs(dataset_root: Path) -> Iterable[Path]:
    # Expect: dataset_root/group*/<patient_name>/
    for group in sorted(dataset_root.iterdir()):
        if not group.is_dir():
            continue
        if not group.name.startswith("group"):
            continue
        for patient in sorted(group.iterdir()):
            if patient.is_dir():
                yield patient


def _print_list(title: str, items: List[str], limit: int) -> None:
    if not items:
        return
    print(f"  - {title}: {len(items)}")
    for s in items[:limit]:
        print(f"    - {s}")
    if len(items) > limit:
        print(f"    - ... ({len(items) - limit} more)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Strictly validate patient folder structure against a reference.")
    ap.add_argument("--dataset-root", default=DATASET_ROOT_DEFAULT, help="Dataset root containing group*/patient dirs.")
    ap.add_argument("--reference", default=REFERENCE_DEFAULT, help="Reference patient directory path.")
    ap.add_argument("--only-fail", action="store_true", help="Only print patients that fail the check.")
    ap.add_argument("--limit", type=int, default=50, help="Max items to show per diff category.")
    ap.add_argument(
        "--json-out",
        default="",
        help="Optional: write full results as JSON to this path (e.g. ./integrity_report.json).",
    )
    args = ap.parse_args()

    # Avoid stack traces when output is piped to head/tail.
    # (BrokenPipeError can occur when the downstream closes early.)
    try:
        return _main_impl(args)
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        except Exception:
            pass
        return 0


def _main_impl(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    reference_dir = Path(args.reference)

    ref_manifest = PatientManifest.from_patient_dir(reference_dir)

    results: Dict[str, Dict] = {}
    total = 0
    ok = 0
    fail = 0

    for patient_dir in iter_patient_dirs(dataset_root):
        total += 1
        cur_manifest = PatientManifest.from_patient_dir(patient_dir)
        diff = diff_manifests(ref_manifest, cur_manifest)

        rel_id = patient_dir.as_posix()
        results[rel_id] = {
            "ok": diff.ok,
            "missing_dirs": diff.missing_dirs,
            "extra_dirs": diff.extra_dirs,
            "missing_files": diff.missing_files,
            "extra_files": diff.extra_files,
        }

        if diff.ok:
            ok += 1
            if not args.only_fail:
                print(f"[OK]   {patient_dir}")
        else:
            fail += 1
            print(f"[FAIL] {patient_dir}")
            _print_list("missing_dirs", diff.missing_dirs, args.limit)
            _print_list("extra_dirs", diff.extra_dirs, args.limit)
            _print_list("missing_files", diff.missing_files, args.limit)
            _print_list("extra_files", diff.extra_files, args.limit)

    print("---")
    print(f"reference: {reference_dir}")
    print(f"dataset_root: {dataset_root}")
    print(f"total_patients: {total}")
    print(f"ok: {ok}")
    print(f"fail: {fail}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reference": reference_dir.as_posix(),
            "dataset_root": dataset_root.as_posix(),
            "summary": {"total": total, "ok": ok, "fail": fail},
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json_out: {out_path.resolve()}")

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

