#!/usr/bin/env python3
"""Generate strict multi-turn answers from OpenAI-compatible API models on the OralGPT-CMF test patients.

The prompt matches the fine-tuned model rollout: the dataset system context plus the model's own prior
answers as the clinical record, and the current question with only the current turn's images.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx
import openai
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.conversation import clean_question, conversation_pairs, rollout_system_context, system_context
from answer.utils import write_json_atomic

DEFAULT_DATASET = ROOT / "sft/data/turn_level" / "oralcmf_sft_test_rollout.json"
DEFAULT_OUTPUT = ROOT / "output"
EMPTY_RETRY_MAX_TOKENS = 8192
PRINT_LOCK = threading.Lock()
PROGRESS = {"patients": 0, "total_patients": 0, "turns": 0, "total_turns": 0}


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def advance(turns: int = 0, patients: int = 0) -> str:
    with PRINT_LOCK:
        PROGRESS["turns"] += turns
        PROGRESS["patients"] += patients
        return f"[progress patients {PROGRESS['patients']}/{PROGRESS['total_patients']} turns {PROGRESS['turns']}/{PROGRESS['total_turns']}]"


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-id", action="append", default=[], help="Run only these case IDs (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected cases")
    parser.add_argument("--num-workers", type=int, default=4, help="Patients answered in parallel")
    parser.add_argument("--answer-model", default=os.environ.get("ANSWER_OPENAI_MODEL"), help="Model name sent to the API")
    parser.add_argument("--answer-output-model", default=None, help="Model name used for output/cache folders")
    parser.add_argument("--answer-base-url", default=os.environ.get("ANSWER_OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="ANSWER_OPENAI_API_KEY", help="Name of the .env variable holding the API key")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("PERCEPTION_MAX_TOKENS", "2048")),
                        help="Initial max_tokens; doubled when the reply is empty or truncated")
    parser.add_argument("--max-token-limit", type=int, default=EMPTY_RETRY_MAX_TOKENS, help="Maximum token budget for empty or truncated answer retries")
    parser.add_argument("--retry-length", action="store_true", help="Clear caches from the first truncated turn onward; new requests start with at least 8192 tokens")
    parser.add_argument("--image-max-pixels", type=int, default=602112, help="Downscale images above this pixel count")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP read timeout; in streaming mode this limits idle time between chunks")
    parser.add_argument("--stream", action="store_true", help="Receive streaming responses and assemble the final answer")
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per request on transient API errors")
    parser.add_argument("--force", action="store_true", help="Discard saved answers and cache for this model")
    args = parser.parse_args()
    if not args.answer_model or not args.answer_base_url:
        parser.error("Set ANSWER_OPENAI_MODEL/ANSWER_OPENAI_BASE_URL in .env or pass --answer-model/--answer-base-url")
    if args.num_workers < 1 or args.max_tokens < 1 or args.max_attempts < 1:
        parser.error("--num-workers, --max-tokens and --max-attempts must be positive")
    initial_tokens = max(args.max_tokens, 8192) if args.retry_length else args.max_tokens
    if args.max_token_limit < initial_tokens:
        parser.error("--max-token-limit must be at least the initial request budget")
    return args


def image_data_url(path: Path, max_pixels: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width * image.height > max_pixels:
            scale = math.sqrt(max_pixels / (image.width * image.height))
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") or "" for part in content)
    raise TypeError(f"Unsupported message content type: {type(content).__name__}")


def complete(client: OpenAI, model: str, messages: list[dict[str, Any]], max_tokens: int, max_attempts: int, tag: str, stream: bool = False, max_token_limit: int = EMPTY_RETRY_MAX_TOKENS) -> dict[str, Any]:
    failures = 0
    started = time.time()
    while True:
        try:
            options = {"stream": True} if stream else {}
            log(f"{tag} requesting max_tokens={max_tokens} stream={stream}")
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, **options)
            if stream:
                parts = []
                finish_reason = None
                with response:
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        if choice.finish_reason is not None:
                            finish_reason = choice.finish_reason
                        part = content_text(choice.delta.content)
                        if part:
                            if not parts:
                                log(f"{tag} first text chunk after {time.time() - started:.1f}s")
                            parts.append(part)
                if finish_reason is None:
                    raise RuntimeError(f"{tag} stream ended without a finish reason; incomplete answer not cached")
                text = "".join(parts).strip()
            else:
                choice = response.choices[0]
                text = content_text(choice.message.content).strip()
                finish_reason = choice.finish_reason
        except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError, httpx.TransportError) as error:
            failures += 1
            if failures >= max_attempts:
                raise
            wait = 65 if isinstance(error, openai.RateLimitError) else min(60, 5 * 2**failures)
            log(f"{tag} {type(error).__name__}, retry {failures}/{max_attempts - 1} in {wait}s: {str(error)[:200]}")
            time.sleep(wait)
            continue
        if text and finish_reason == "stop":
            return {
                "answer": text,
                "finish_reason": finish_reason,
            }
        if finish_reason not in ("stop", "length"):
            raise RuntimeError(f"{tag} answer did not finish normally (finish_reason={finish_reason}); not cached")
        problem = "truncated answer" if finish_reason == "length" else "empty answer"
        if max_tokens >= max_token_limit:
            raise RuntimeError(f"{tag} {problem} at max_tokens={max_tokens} (finish_reason={finish_reason}); not cached")
        max_tokens = min(max_tokens * 2, max_token_limit)
        log(f"{tag} {problem} (finish_reason={finish_reason}), retry with max_tokens={max_tokens}")


def clear_truncated_suffix(answers_path: Path, cache_dir: Path) -> int | None:
    truncated = []
    if answers_path.is_file():
        truncated.extend(row["turn_index"] for row in json.loads(answers_path.read_text(encoding="utf-8")) if row["finish_reason"] == "length")
    cache_paths = list(cache_dir.glob("answer_turn_*.json"))
    for path in cache_paths:
        if json.loads(path.read_text(encoding="utf-8"))["finish_reason"] == "length":
            truncated.append(int(path.stem.rsplit("_", 1)[1]))
    if not truncated:
        return None
    first = min(truncated)
    answers_path.unlink(missing_ok=True)
    for path in cache_paths:
        if int(path.stem.rsplit("_", 1)[1]) >= first:
            path.unlink()
    return first


def answer_patient(case: dict[str, Any], args: argparse.Namespace, client: OpenAI, model_dir: str) -> str:
    group, name = case["id"].split("__", 1)
    patient_dir = args.output_root / group / name
    answers_path = patient_dir / "evaluation" / model_dir / "answers.json"
    cache_dir = patient_dir / "cache" / model_dir
    pairs = conversation_pairs(case)
    if args.retry_length:
        first = clear_truncated_suffix(answers_path, cache_dir)
        if first is not None:
            log(f"[answers][{case['id']}] cleared truncated turn {first} and all later caches")
    if args.force:
        answers_path.unlink(missing_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    elif answers_path.is_file() and len(json.loads(answers_path.read_text(encoding="utf-8"))) == len(pairs):
        log(f"[answers][{case['id']}] already complete {advance(turns=len(pairs), patients=1)}")
        return "skipped"

    dataset_dir = args.dataset.parent
    patient_context = system_context(case)
    history: list[tuple[str, str]] = []
    answers = []
    image_cursor = 0
    for turn_index, (question, reference) in enumerate(pairs, start=1):
        image_count = question["value"].count("<image>")
        images = case["images"][image_cursor : image_cursor + image_count]
        image_cursor += image_count
        if len(images) != image_count:
            raise ValueError(f"{case['id']}: image references do not match turn {turn_index}")

        system = rollout_system_context(patient_context, history)
        request = {
            "model": args.answer_model,
            "system": system,
            "question": question["value"],
            "images": images,
            "image_max_pixels": args.image_max_pixels,
            "max_tokens": args.max_tokens,
        }
        cache_path = cache_dir / f"answer_turn_{turn_index:02d}.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else None
        tag = f"[answers][{case['id']}] turn {turn_index}/{len(pairs)}"
        if cached and cached["input"] == request:
            result, source = cached, "cache"
        else:
            content = [{"type": "image_url", "image_url": {"url": image_data_url(dataset_dir / image, args.image_max_pixels)}} for image in images]
            content.append({"type": "text", "text": clean_question(question["value"])})
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            initial_tokens = max(args.max_tokens, 8192) if args.retry_length else args.max_tokens
            result = {"input": request, **complete(client, args.answer_model, messages, initial_tokens, args.max_attempts, tag, stream=args.stream, max_token_limit=args.max_token_limit)}
            write_json_atomic(cache_path, result)
            source = "api"

        history.append((question["value"], result["answer"]))
        answers.append(
            {
                "task_id": f"{case['id']}__turn_{turn_index:02d}",
                "turn_index": turn_index,
                "question": clean_question(question["value"]),
                "images": images,
                "gold_answer": reference["value"],
                "model_answer": result["answer"],
                "finish_reason": result["finish_reason"],

            }
        )
        log(f"{tag} {source} finish={result['finish_reason']} {advance(turns=1)}")

    write_json_atomic(answers_path, answers)
    log(f"[answers][{case['id']}] completed {advance(patients=1)}")
    return "completed"


def main() -> int:
    load_env(ROOT / ".env")
    args = parse_args()
    model_dir = (args.answer_output_model or args.answer_model).replace("/", "_")
    client = OpenAI(api_key=os.environ[args.api_key_env], base_url=args.answer_base_url, timeout=args.timeout, max_retries=0)

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
        if wanted - {case["id"] for case in cases}:
            raise ValueError(f"Unknown case IDs: {sorted(wanted - {case['id'] for case in cases})}")
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No cases selected")
    PROGRESS["total_patients"] = len(cases)
    PROGRESS["total_turns"] = sum(len(conversation_pairs(case)) for case in cases)
    log(f"[answers][batch] model={args.answer_model} output_model={model_dir} cases={len(cases)} workers={args.num_workers} stream={args.stream} timeout={args.timeout}s output={args.output_root}")

    status: dict[str, list[str]] = {"completed": [], "skipped": [], "failed": []}
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(answer_patient, case, args, client, model_dir): case["id"] for case in cases}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                log(f"[answers][batch][waiting] unfinished_patients={len(pending)} failed={len(status['failed'])} {advance()}")
            for future in done:
                case_id = futures[future]
                try:
                    status[future.result()].append(case_id)
                except Exception:
                    status["failed"].append(case_id)
                    log(f"[answers][{case_id}] FAILED\n{traceback.format_exc()}")
    log(f"[answers][batch][done] completed={len(status['completed'])} skipped={len(status['skipped'])} failed={len(status['failed'])} {sorted(status['failed'])}")
    return 1 if status["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
