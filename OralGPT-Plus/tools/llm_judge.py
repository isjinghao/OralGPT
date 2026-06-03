#!/usr/bin/env python3
"""LLM-as-a-judge scorer for the dental panoramic X-ray benchmark.

Given a predictions JSON file (a list of {"question", "ground_truth",
"prediction"} records), each case is scored on a 0.0-1.0 scale by an
LLM judge and a per-file average is reported.

The judge API key is read from the API_KEY environment variable:

    export API_KEY="your_judge_api_key"
    python llm_judge.py --input predictions.json --output scored.json
"""

import os
import re
import json
import time
import argparse
import http.client
from typing import List, Dict, Any

SYSTEM_PROMPT = """
You are an expert judge for a benchmark on dental panoramic X-ray images. Your job is to strictly evaluate how correct a model's prediction is compared to the ground truth (GT).

Output rule:
- Output ONLY a single numeric score from this set: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0. No words, no symbols, no explanations.

Clinical evaluation rules:
- Judge clinical meaning, not wording; accept clinically equivalent synonyms, such as:
  * "orthodontic appliance" approximately equals "fixed braces/brackets + archwire"
  * "surgical device/hardware" approximately equals "fixation plate/screws/osteosynthesis hardware"
  * "prosthetic restoration" approximately equals "crown/bridge/veneer"
  * "impaction" approximately equals "unerupted third molar"
- Accept equivalent tooth numbering systems (FDI / Universal / Palmer).
- Coordinates may be absolute or normalized <box>[x1,y1,x2,y2]</box>. If boxes are omitted but the clinical meaning is the same, treat as a minor omission.
- Penalize mismatched diseases/abnormalities/conditions as a major error.
- If prediction asserts additional findings that contradict GT or are clinically false, treat as a major error.
- Do not over-penalize phrasing; do penalize missing core findings.

Scoring guide:
- 1.0: Fully correct, matches GT on all clinically material aspects, no extra wrong claims.
- 0.7-0.9: Mostly correct, only minor omissions or clinically insignificant imprecision, no contradiction to GT.
- 0.3-0.6: Partially correct, some correct elements but misses at least one core aspect or introduces a major error.
- 0.1-0.2: Largely incorrect, minimal overlap with GT.
- 0.0: Totally incorrect, no meaningful overlap, fundamental contradiction, refusal, or generic non-answer.

Modality constraint:
- Images are dental panoramic X-rays only. If a prediction discusses non-radiographic features (mucosa color, soft-tissue surface appearance, etc.), treat as incorrect unless explicitly grounded in radiographic signs.
"""

QUERY_PROMPT = """
Score the case below by outputting ONLY one number from {{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}}. Do not output any words or explanations.

Question: {question}
Ground Truth: {ground_truth}
Prediction: {prediction}

Output ONLY one value from: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
"""

VALID_SCORES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def call_gpt_api(messages, api_key, api_host, model, max_tokens=64,
                 temperature=0.0, max_retries=5, timeout=240):
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            conn = http.client.HTTPSConnection(api_host, timeout=timeout)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            resp = conn.getresponse()
            data = resp.read().decode("utf-8", errors="ignore")
            conn.close()
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status}: {data[:200]}")
            choices = json.loads(data).get("choices", [])
            if not choices:
                raise ValueError("Response has no choices")
            content = choices[0].get("message", {}).get("content") or ""
            if isinstance(content, list):
                content = "\n".join(c if isinstance(c, str) else c.get("text", "") for c in content)
            content = str(content).strip()
            if not content:
                raise ValueError("Empty content")
            return content
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 10))
    raise last_err if last_err else RuntimeError("API call failed")


def parse_score(resp: str) -> float:
    resp = resp.strip()
    try:
        score = float(resp)
        if score in VALID_SCORES:
            return score
        return max(0.0, min(1.0, round(score * 10) / 10))
    except ValueError:
        pass
    numbers = re.findall(r"\b([0-1]\.?[0-9]?)\b", resp)
    if numbers:
        try:
            return max(0.0, min(1.0, round(float(numbers[0]) * 10) / 10))
        except ValueError:
            pass
    return 0.0


def gpt_judge(question: str, ground_truth: str, prediction: str,
              api_key: str, api_host: str, judge_model: str) -> float:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUERY_PROMPT.format(
            question=question, ground_truth=ground_truth, prediction=prediction)},
    ]
    try:
        resp = call_gpt_api(messages, api_key, api_host, judge_model)
    except Exception as e:
        print(f"judge call failed: {e}")
        return 0.0
    return parse_score(resp)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge scorer for dental panoramic X-ray predictions")
    parser.add_argument("--input", required=True, help="Predictions JSON: list of {question, ground_truth, prediction}")
    parser.add_argument("--output", default="scored.json", help="Output JSON with per-case scores")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="Judge API key (or set API_KEY env)")
    parser.add_argument("--api-host", default="jeniya.top", help="Judge API host")
    parser.add_argument("--judge-model", default="gpt-5-nano-2025-08-07", help="Judge model name")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("No API key. Set API_KEY env or pass --api-key.")

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = []
    total = 0.0
    for i, rec in enumerate(records):
        score = gpt_judge(
            rec.get("question", ""),
            rec.get("ground_truth", ""),
            rec.get("prediction", ""),
            args.api_key, args.api_host, args.judge_model,
        )
        total += score
        results.append({**rec, "score": score})
        print(f"[{i + 1}/{len(records)}] score={score}")

    avg = total / len(records) if records else 0.0
    output = {"average_score": avg, "total_samples": len(records), "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"average_score={avg:.4f} over {len(records)} samples -> {args.output}")


if __name__ == "__main__":
    main()
