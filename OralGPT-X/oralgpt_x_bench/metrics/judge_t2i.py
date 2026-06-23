#!/usr/bin/env python3
"""GPT-based quality judge for Oral T2I benchmark (WISE-style, simplified)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

from metrics.io_utils import append_jsonl, load_completed_ids
from path_utils import bench_root, resolve_path

JUDGE_PROMPT_TEMPLATE = """You are evaluating a text-to-image model for oral/dental imaging.

PROMPT: "{prompt}"

Score each dimension from 0 to 2 (integers only):
- **Consistency**: Does the image match the prompt (intraoral view, caries if requested)?
  0 = mismatch, 1 = partial, 2 = strong match
- **Realism**: Does the image look like a plausible clinical photograph (not cartoon/artifact)?
  0 = unrealistic, 1 = somewhat plausible, 2 = photorealistic intraoral photo
- **Clinical_plausibility**: Are oral anatomy and caries presentation medically plausible?
  0 = implausible anatomy/artifacts, 1 = partially plausible, 2 = clinically plausible

Return ONLY three lines:
Consistency: <0-2>
Realism: <0-2>
Clinical_plausibility: <0-2>
"""


def load_samples(metadata_file: Path) -> list[dict]:
    text = metadata_file.read_text(encoding="utf-8")
    if metadata_file.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        return payload["samples"]
    return payload


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_scores(text: str) -> dict[str, float | None]:
    scores = {"consistency": None, "realism": None, "clinical_plausibility": None}
    patterns = {
        "consistency": r"Consistency\s*[:：]\s*(\d)",
        "realism": r"Realism\s*[:：]\s*(\d)",
        "clinical_plausibility": r"Clinical_plausibility\s*[:：]\s*(\d)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scores[key] = float(match.group(1))
    return scores


def judge_with_openai(prompt: str, image_path: Path, model: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))
    user_text = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
                    },
                ],
            }
        ],
    )
    evaluation = response.choices[0].message.content or ""
    scores = parse_scores(evaluation)
    return {"evaluation": evaluation.strip(), **scores}


def judge_stub(prompt: str, image_path: Path) -> dict:
    return {
        "evaluation": (
            "STUB MODE (no OPENAI_API_KEY): skipped GPT judge. "
            f"Set OPENAI_API_KEY to run real evaluation for prompt={prompt!r}, image={image_path.name}."
        ),
        "consistency": None,
        "realism": None,
        "clinical_plausibility": None,
        "judge_mode": "stub",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_file", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="t2i")
    parser.add_argument("--model", type=str, default=os.environ.get("JUDGE_MODEL", "gpt-4o"))
    parser.add_argument(
        "--judge_mode",
        choices=["auto", "openai", "stub"],
        default="auto",
        help="auto=OpenAI if OPENAI_API_KEY set, else stub",
    )
    args = parser.parse_args()

    metadata_file = resolve_path(args.metadata_file, bench_root())
    pred_dir = resolve_path(args.pred_dir)
    output_jsonl = resolve_path(args.output_jsonl)

    mode = args.judge_mode
    if mode == "auto":
        mode = "openai" if os.environ.get("OPENAI_API_KEY") else "stub"
    if mode == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("judge_mode=openai requires OPENAI_API_KEY")

    completed = load_completed_ids(output_jsonl)
    samples = load_samples(metadata_file)

    for sample in samples:
        sample_id = sample["id"]
        if sample_id in completed:
            continue
        pred_path = pred_dir / f"{sample_id}.png"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        prompt = sample["prompt"]
        if mode == "openai":
            result = judge_with_openai(prompt, pred_path, args.model)
            result["judge_mode"] = "openai"
        else:
            result = judge_stub(prompt, pred_path)

        meta = sample.get("metadata", {})
        row = {
            "id": sample_id,
            "benchmark": args.benchmark,
            "prompt": prompt,
            "pred_path": str(pred_path),
            **result,
        }
        for key in ("category", "modality", "finding"):
            if key in sample:
                row[key] = sample[key]
            elif meta.get(key) is not None:
                row[key] = meta[key]
        append_jsonl(output_jsonl, row)
        print(f"Judged {sample_id} ({result.get('judge_mode', mode)})")

    print(f"Wrote judge metrics to {output_jsonl}")


if __name__ == "__main__":
    main()
