#!/usr/bin/env python3
"""
Filter samples from a large JSON/JSONL file using keywords read from a txt file,
and write up to N matched samples to an output JSON file.

Features:
- Supports input as JSON array (streaming) or JSONL (one JSON object per line).
- Reads keywords from a txt that is a JSON dict (keys are keywords). Falls back to line-per-keyword.
- Case-insensitive substring matching using a compiled regex (alternation of escaped keywords).
- Reservoir sampling to uniformly keep up to --limit items without storing all matches.

Usage example:
  python scripts/filter_by_keywords.py \
    --input /path/to/r1_mmoral_240k.json \
    --keywords /path/to/opg_keywords_en_dict.txt \
    --output /path/to/output_matched_20k.json \
    --limit 20000

Notes:
- If matched items < limit, all matched will be written.
- For reproducibility of sampling, set --seed (default: 42).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
from typing import Any, Dict, Generator, Iterable, List


def load_keywords(path: str) -> List[str]:
    """Load keywords from a file.

    Prefer JSON dict with keys as keywords, otherwise treat as one keyword per line.
    Returns a de-duplicated list filtered for non-empty strings.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        # Try parse as JSON (dict or list)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                keys = list(data.keys())
            elif isinstance(data, list):
                keys = [str(x) for x in data]
            else:
                keys = []
        except Exception:
            # Fallback to lines
            keys = [line.strip() for line in text.splitlines() if line.strip()]
    # Normalize and deduplicate
    out: List[str] = []
    seen = set()
    for k in keys:
        k_norm = str(k).strip()
        if not k_norm:
            continue
        if k_norm.lower() in seen:
            continue
        seen.add(k_norm.lower())
        out.append(k_norm)
    return out


def compile_keyword_regex(keywords: List[str]) -> re.Pattern:
    # Escape and sort by length desc to reduce catastrophic backtracking chances
    parts = [re.escape(k) for k in keywords if k]
    # Remove duplicates again just in case
    parts = list(dict.fromkeys(parts))
    # Sort by length descending for performance heuristics
    parts.sort(key=len, reverse=True)
    if not parts:
        # Match nothing
        return re.compile(r"(?!x)x")
    pattern = "|".join(parts)
    return re.compile(pattern, flags=re.IGNORECASE)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                # If not dict, still yield as a sample (could be list/str)
                yield {"value": obj}


def iter_json_array_stream(path: str) -> Iterable[Any]:
    """Stream top-level JSON objects from a file whose top-level is an array.

    Avoids loading entire array into memory by parsing objects via brace/quote tracking.
    Yields Python objects parsed via json.loads for each top-level element.
    """
    with open(path, 'r', encoding='utf-8') as f:
        # Skip whitespace until '['
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        if ch != '[':
            raise ValueError("JSON array expected (file should start with '[').")

        in_string = False
        escape = False
        depth = 0
        buf = io.StringIO()

        while True:
            ch = f.read(1)
            if not ch:
                break
            if in_string:
                buf.write(ch)
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch.isspace():
                # write delimiter whitespace inside element buffer
                if depth > 0:
                    buf.write(ch)
                continue

            if ch == '"':
                in_string = True
                buf.write(ch)
                continue

            if ch == ']':
                # end of array
                break

            if ch == ',':
                # delimiter between elements when not inside an element
                continue

            # Start of an element: could be object '{', array '[', number, true, false, null, string already handled.
            # We'll capture until we reach depth==0 for objects/arrays or we parse a primitive by looking ahead commas/brackets.
            buf = io.StringIO()
            if ch in '{[':
                depth = 1
                buf.write(ch)
                while True:
                    ch2 = f.read(1)
                    if not ch2:
                        break
                    buf.write(ch2)
                    if in_string:
                        if escape:
                            escape = False
                        elif ch2 == '\\':
                            escape = True
                        elif ch2 == '"':
                            in_string = False
                        continue
                    else:
                        if ch2 == '"':
                            in_string = True
                            continue
                        if ch2 in '{[':
                            depth += 1
                        elif ch2 in '}]':
                            depth -= 1
                            if depth == 0:
                                break
                try:
                    yield json.loads(buf.getvalue())
                except Exception:
                    # Skip malformed element
                    pass
            else:
                # Primitive (number, true, false, null) or a bare string was handled above
                token_chars = [ch]
                while True:
                    pos = f.tell()
                    ch2 = f.read(1)
                    if not ch2 or ch2 in ",]":
                        # step back if we consumed delimiter
                        if ch2:
                            f.seek(pos)
                        break
                    token_chars.append(ch2)
                token = "".join(token_chars).strip()
                try:
                    yield json.loads(token)
                except Exception:
                    pass


def detect_and_iter(path: str) -> Iterable[Any]:
    # Peek first non-space char to detect format
    with open(path, 'r', encoding='utf-8') as f:
        start = f.read(4096)
    first_non_ws = next((c for c in start if not c.isspace()), '')
    if first_non_ws == '[':
        return iter_json_array_stream(path)
    elif first_non_ws == '{':
        # Try whole-file JSON first
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                # Yield list items
                return (item for item in data)
            elif isinstance(data, dict):
                # If a dict contains list under common keys
                for key in ("data", "samples", "items", "records"):
                    if key in data and isinstance(data[key], list):
                        return (item for item in data[key])
                # Otherwise, treat the dict itself as one sample
                return (data,)
        except Exception:
            # Fallback to JSONL
            return iter_jsonl(path)
    else:
        # Assume JSONL
        return iter_jsonl(path)


def iter_strings(node: Any) -> Iterable[str]:
    """Yield all string values in a nested JSON-like structure."""
    if node is None:
        return
    if isinstance(node, str):
        yield node
        return
    if isinstance(node, dict):
        for v in node.values():
            yield from iter_strings(v)
        return
    if isinstance(node, list):
        for v in node:
            yield from iter_strings(v)
        return
    # Other primitives ignored


def sample_matched(items: Iterable[Any], pattern: re.Pattern, limit: int, seed: int = 42) -> List[Any]:
    random.seed(seed)
    reservoir: List[Any] = []
    seen = 0
    total = 0
    matched = 0

    for item in items:
        total += 1
        # Build a combined text lazily; stop early if matched
        found = False
        # Iterate string fields and test incrementally to avoid joining everything when not needed
        for s in iter_strings(item):
            if not s:
                continue
            if pattern.search(s):
                found = True
                break
        if not found:
            continue
        matched += 1
        if len(reservoir) < limit:
            reservoir.append(item)
        else:
            j = random.randint(0, matched - 1)
            if j < limit:
                reservoir[j] = item

        # Optional: short-circuit if matched == limit and we don't want uniform sampling.
        # We keep reservoir sampling for unbiased picks across the whole file.

        if matched % 10000 == 0:
            print(f"Scanned {total} items, matched {matched}...", file=sys.stderr)

    print(f"Done. Scanned {total}, matched {matched}, kept {len(reservoir)} (limit={limit}).", file=sys.stderr)
    return reservoir


def main():
    ap = argparse.ArgumentParser(description="Filter JSON/JSONL by keywords and output up to N samples.")
    ap.add_argument("--input", required=True, help="Input JSON/JSONL file path")
    ap.add_argument("--keywords", required=True, help="Keywords file path (JSON dict or line-per-key)")
    ap.add_argument("--output", required=True, help="Output JSON file path")
    ap.add_argument("--limit", type=int, default=20000, help="Maximum number of matched samples to output (default: 20000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reservoir sampling")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.limit <= 0:
        print("--limit must be > 0", file=sys.stderr)
        sys.exit(2)

    keywords = load_keywords(args.keywords)
    if not keywords:
        print("No keywords loaded. Exiting.", file=sys.stderr)
        sys.exit(3)
    print(f"Loaded {len(keywords)} keywords.", file=sys.stderr)
    pattern = compile_keyword_regex(keywords)

    items_iter = detect_and_iter(args.input)
    selected = sample_matched(items_iter, pattern, limit=args.limit, seed=args.seed)

    # Write output
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False)
    print(f"Wrote {len(selected)} samples to {args.output}")


if __name__ == "__main__":
    main()
