#!/usr/bin/env python3
"""Convert OralCorpus textbook JSONL into BAGEL vlm_sft training format.

Source layout (under --input-root):
  text_Chinese/*.jsonl   keys: 内容, 书名, 页码, ...
  text_English/*.jsonl   keys: Content, Title, Page_number, ...

Output layout (under --output-root):
  oral_corpus_bagel.jsonl
  dummy/                         # empty dir required by BAGEL data_dir
  manifest.json
  bagel_dataset_info_snippet.py
  bagel_config_snippet.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("/data/OralGPT/OralCorpus")
DEFAULT_OUTPUT = Path("/data/OralGPT/OralGPT-X/dataset_OralCorpus")

LANG_DIRS = {
    "chinese": ("text_Chinese", "内容", "书名", "页码"),
    "english": ("text_English", "Content", "Title", "Page_number"),
}


@dataclass
class PageRecord:
    page_number: int | None
    text: str


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return slug[:120] or "book"


def read_book_pages(
    jsonl_path: Path,
    content_key: str,
    page_key: str,
    min_chars: int,
) -> list[PageRecord]:
    pages: list[PageRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get(content_key) or "").strip()
            if len(text) < min_chars:
                continue
            page_raw = row.get(page_key)
            try:
                page_number = int(page_raw) if page_raw is not None else None
            except (TypeError, ValueError):
                page_number = None
            pages.append(PageRecord(page_number=page_number, text=text))
    return pages


def merge_pages(
    pages: list[PageRecord],
    max_chars: int,
) -> list[tuple[int | None, int | None, str]]:
    if not pages:
        return []

    chunks: list[tuple[int | None, int | None, str]] = []
    buffer: list[str] = []
    buffer_len = 0
    start_page: int | None = None
    end_page: int | None = None

    def flush() -> None:
        nonlocal buffer, buffer_len, start_page, end_page
        if not buffer:
            return
        chunks.append((start_page, end_page, "\n\n".join(buffer)))
        buffer = []
        buffer_len = 0
        start_page = None
        end_page = None

    for page in pages:
        text = page.text
        extra = 2 if buffer else 0
        if buffer and buffer_len + extra + len(text) <= max_chars:
            buffer.append(text)
            buffer_len += extra + len(text)
            end_page = page.page_number
            continue

        flush()

        if len(text) <= max_chars:
            buffer = [text]
            buffer_len = len(text)
            start_page = page.page_number
            end_page = page.page_number
            continue

        start = 0
        while start < len(text):
            piece = text[start : start + max_chars]
            chunks.append((page.page_number, page.page_number, piece))
            start += max_chars

    flush()
    return chunks


def build_records(
    input_root: Path,
    max_chars: int,
    min_chars: int,
    merge_pages_flag: bool,
    languages: set[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    stats: Counter[str] = Counter()
    book_counts: Counter[str] = Counter()

    for language in ("chinese", "english"):
        if language not in languages:
            continue
        subdir, content_key, title_key, page_key = LANG_DIRS[language]
        lang_dir = input_root / subdir
        if not lang_dir.is_dir():
            raise FileNotFoundError(f"Missing language directory: {lang_dir}")

        jsonl_files = sorted(lang_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {lang_dir}")

        for jsonl_path in jsonl_files:
            book_slug = slugify(jsonl_path.stem)
            pages = read_book_pages(jsonl_path, content_key, page_key, min_chars)
            stats[f"source_pages_{language}"] += len(pages)
            if not pages:
                stats["skipped_empty_books"] += 1
                continue

            if merge_pages_flag:
                chunks = merge_pages(pages, max_chars=max_chars)
            else:
                chunks = [(p.page_number, p.page_number, p.text) for p in pages]

            for chunk_idx, (start_page, end_page, text) in enumerate(chunks):
                if start_page is not None and end_page is not None:
                    lo, hi = sorted((start_page, end_page))
                    page_tag = f"p{lo}-{hi}" if lo != hi else f"p{lo}"
                elif start_page is not None:
                    page_tag = f"p{start_page}"
                else:
                    page_tag = f"chunk{chunk_idx:05d}"

                sample_id = f"{language}_{book_slug}_{page_tag}"
                records.append(
                    {
                        "id": sample_id,
                        "conversations": [{"from": "gpt", "value": text}],
                    }
                )
                stats[f"output_samples_{language}"] += 1
                book_counts[f"{language}/{jsonl_path.name}"] += 1

    manifest = {
        "input_root": str(input_root.resolve()),
        "output_format": "bagel_vlm_sft_jsonl",
        "languages": sorted(languages),
        "merge_pages": merge_pages_flag,
        "max_chars": max_chars,
        "min_chars": min_chars,
        "total_samples": len(records),
        "stats": dict(stats),
        "books_with_samples": len(book_counts),
    }
    return records, manifest


def write_outputs(output_root: Path, records: list[dict[str, object]], manifest: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dummy_dir = output_root / "dummy"
    dummy_dir.mkdir(exist_ok=True)

    jsonl_path = output_root / "oral_corpus_bagel.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    dataset_info = f"""# Add under DATASET_INFO['vlm_sft'] in Bagel/data/dataset_info.py
'oral_corpus': {{
    'data_dir': '{dummy_dir.resolve()}',
    'jsonl_path': '{jsonl_path.resolve()}',
    'num_total_samples': {len(records)},
}},
"""
    (output_root / "bagel_dataset_info_snippet.py").write_text(dataset_info, encoding="utf-8")

    config_yaml = f"""# Example Bagel config: data/configs/oral_corpus.yaml
vlm_sft:
  dataset_names:
    - oral_corpus
  image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040
  is_mandatory: true
  shuffle_lines: true
  shuffle_seed: 0
  num_used_data:
    - {len(records)}
  weight: 1
"""
    (output_root / "bagel_config_snippet.yaml").write_text(config_yaml, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3500,
        help="Max characters per training sample after page merging.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Skip source pages shorter than this.",
    )
    parser.add_argument(
        "--no-merge-pages",
        action="store_true",
        help="Keep one BAGEL sample per source page instead of merging.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=("chinese", "english"),
        default=("chinese", "english"),
        help="Which OralCorpus language folders to convert.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)

    records, manifest = build_records(
        input_root=input_root,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        merge_pages_flag=not args.no_merge_pages,
        languages=set(args.languages),
    )
    if not records:
        raise RuntimeError("No training samples were produced. Check input paths and filters.")

    write_outputs(output_root, records, manifest)

    print(f"Wrote {len(records)} samples to {output_root / 'oral_corpus_bagel.jsonl'}")
    print(f"Manifest: {output_root / 'manifest.json'}")
    print(f"BAGEL snippets: {output_root / 'bagel_dataset_info_snippet.py'}")
    print(
        "Next: copy bagel_dataset_info_snippet.py into Bagel/data/dataset_info.py, "
        "add bagel_config_snippet.yaml to Bagel/data/configs/, and train with --visual_gen False."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
