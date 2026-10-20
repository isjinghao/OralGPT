#!/usr/bin/env python3
"""Run deterministic, strict multi-turn SFT rollouts on the OralCMF test split."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/OralGPT/OralGPT-CMF/sft")
LLAMA_ROOT = Path("/root/autodl-tmp/LLaMA-Factory-qwen35")
DEFAULT_DATASET = ROOT / "sft_data" / "oralcmf_sft_test_rollout.json"
DEFAULT_ADAPTER = ROOT / "sft_output" / "qwen3_5_9b_lora_turn_level"
DEFAULT_OUTPUT = ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true", help="Run the base model without a LoRA adapter")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-model", required=True, help="Model folder name, e.g. qwen3.5-9b-v0")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the selected cases across this many processes")
    parser.add_argument("--shard-index", type=int, default=0, help="Cases [shard_index::num_shards] are run")
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Qwen3.5-9B"))
    parser.add_argument("--template", default="qwen3_5_nothink")
    parser.add_argument("--llama-root", type=Path, default=LLAMA_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--case-id", action="append", default=[], help="Run only exact case IDs")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected cases")
    parser.add_argument("--overwrite", action="store_true", help="Discard saved rollout states")
    parser.add_argument("--question-prompts", type=Path, help="Inject only common and current-question rules from YAML")
    return parser.parse_args()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def patient_paths(output_root: Path, model: str, case_id: str) -> tuple[Path, Path]:
    group, name = case_id.split("__", 1)
    patient_dir = output_root / group / name
    return patient_dir / "cache" / model / "rollout_state.json", patient_dir / "evaluation" / model / "answers.json"


def write_answers(path: Path, state: dict[str, Any]) -> None:
    write_json_atomic(
        path,
        [
            {
                "task_id": f"{state['case_id']}__turn_{turn['turn_index']:02d}",
                "turn_index": turn["turn_index"],
                "question": clean_question(turn["question"]),
                "images": turn["images"],
                "gold_answer": turn["reference"],
                "model_answer": turn["prediction"],
                "finish_reason": turn["finish_reason"],
                "usage": {"prompt_tokens": turn["prompt_length"], "completion_tokens": turn["response_length"]},
                "max_tokens_used": state["decoding"]["max_new_tokens"],
                "seconds": turn["seconds"],
            }
            for turn in state["turns"]
        ],
    )


def system_context(case: dict[str, Any]) -> str:
    messages = case["conversations"]
    if not messages or messages[0].get("from") != "system" or not messages[0].get("value", "").strip():
        raise ValueError(f"{case['id']}: missing patient system context")
    return messages[0]["value"]


def conversation_pairs(case: dict[str, Any]) -> list[tuple[dict[str, str], dict[str, str]]]:
    messages = case["conversations"][1:]
    if len(messages) % 2:
        raise ValueError(f"{case['id']}: odd non-system conversation length")
    pairs = []
    for index in range(0, len(messages), 2):
        question, reference = messages[index : index + 2]
        if question["from"] != "human" or reference["from"] != "gpt":
            raise ValueError(f"{case['id']}: invalid roles at turns {index}/{index + 1}")
        pairs.append((question, reference))
    return pairs


def load_state(path: Path, case: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    if path.is_file() and not overwrite:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("case_id") != case["id"]:
            raise ValueError(f"State does not match requested case: {path}")
        return state
    return {
        "case_id": case["id"],
        "patient_uid": case.get("patient_uid"),
        "patient_name": case.get("patient_name"),
        "context_turn_labels": case.get("context_turn_labels", []),
        "status": "in_progress",
        "completed_turns": 0,
        "turns": [],
    }


def clean_question(question: str) -> str:
    return re.sub(r"\s*<image>\s*", " ", question).strip()


def rollout_system_context(
    patient_context: str,
    history: list[tuple[str, str]],
    history_heading: str = "Prior clinical reasoning record. Treat the following as established context.",
) -> str:
    if not history:
        return patient_context
    lines = [patient_context, "", history_heading]
    for index, (question, answer) in enumerate(history, start=1):
        lines.extend(("", f"[PRIOR TURN {index}]", f"Question: {clean_question(question)}", f"Recorded answer: {answer.strip()}"))
    return "\n".join(lines)


def rebuild_history(case: dict[str, Any], state: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = conversation_pairs(case)
    completed = state["turns"]
    if len(completed) > len(pairs):
        raise ValueError(f"{case['id']}: state has too many completed turns")

    history = []
    image_cursor = 0
    for turn_index, (question, _) in enumerate(pairs):
        image_count = question["value"].count("<image>")
        turn_images = case.get("images", [])[image_cursor : image_cursor + image_count]
        if len(turn_images) != image_count:
            raise ValueError(f"{case['id']}: image references do not match turn {turn_index}")
        image_cursor += image_count
        if turn_index >= len(completed):
            break
        turn = completed[turn_index]
        if turn["question"] != question["value"] or turn["images"] != turn_images:
            raise ValueError(f"{case['id']}: state does not match source at turn {turn_index}")
        history.append((question["value"], turn["prediction"]))
    return history


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be at least 1.0")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be non-negative")
    if not args.dataset.is_file() or not (args.no_adapter or args.adapter.is_dir()) or not args.model.is_dir():
        raise FileNotFoundError("Dataset, adapter, or base model path does not exist")

    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    if args.limit is not None:
        cases = cases[: args.limit]
    cases = cases[args.shard_index :: args.num_shards]
    if not cases:
        raise ValueError("No cases selected")
    from oral_question_prompts import bind_prompt_state, load_question_prompts, prompt_for_question, question_system

    prompt_bundle = load_question_prompts(args.question_prompts) if args.question_prompts else None
    if prompt_bundle is not None:
        for case in cases:
            for question, _ in conversation_pairs(case):
                prompt_for_question(prompt_bundle, question["value"])
    for case in cases:
        state_path, _ = patient_paths(args.output_root, args.output_model, case["id"])
        bind_prompt_state(load_state(state_path, case, args.overwrite), prompt_bundle)

    sys.path.insert(0, str(args.llama_root / "src"))
    from llamafactory.chat import ChatModel

    model_args = {
        "model_name_or_path": str(args.model),
        "template": args.template,
        "infer_backend": "huggingface",
        "trust_remote_code": True,
        "flash_attn": "sdpa",
        "cutoff_len": 128000,
        "image_max_pixels": 602112,
        "video_max_pixels": 49152,
    }
    if not args.no_adapter:
        model_args.update(adapter_name_or_path=str(args.adapter), finetuning_type="lora")
    model = ChatModel(model_args)
    model.engine.generating_args["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    dataset_dir = args.dataset.parent
    total_turns = sum(len(conversation_pairs(case)) for case in cases)
    shard = f"[{args.output_model} shard {args.shard_index + 1}/{args.num_shards}]"
    print(f"{shard} selected {len(cases)} cases with {total_turns} question-answer turns.", flush=True)
    done_turns = 0

    for case_index, case in enumerate(cases, start=1):
        path, answers_path = patient_paths(args.output_root, args.output_model, case["id"])
        state = load_state(path, case, args.overwrite)
        bind_prompt_state(state, prompt_bundle)
        decoding = {
            "do_sample": False,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_new_tokens": args.max_new_tokens,
        }
        if state.get("turns") and state.get("decoding") != decoding:
            raise ValueError(f"Cannot resume {case['id']} with different decoding settings")
        state["decoding"] = decoding
        pairs = conversation_pairs(case)
        if state.get("status") == "completed":
            write_answers(answers_path, state)
            done_turns += len(pairs)
            print(f"{shard}[case {case_index}/{len(cases)}][turns {done_turns}/{total_turns}] {case['id']}: already complete", flush=True)
            continue

        patient_context = system_context(case)
        history = rebuild_history(case, state)
        start_turn = len(state["turns"])
        done_turns += start_turn
        print(f"{shard}[case {case_index}/{len(cases)}] {case['id']}: resume at turn {start_turn + 1}/{len(pairs)}", flush=True)
        image_cursor = sum(question["value"].count("<image>") for question, _ in pairs[:start_turn])

        for turn_index in range(start_turn, len(pairs)):
            question, reference = pairs[turn_index]
            image_count = question["value"].count("<image>")
            turn_images = case.get("images", [])[image_cursor : image_cursor + image_count]
            if len(turn_images) != image_count:
                raise ValueError(f"{case['id']}: image references do not match turn {turn_index}")
            absolute_images = [str(dataset_dir / image) for image in turn_images]
            if any(not Path(image).is_file() for image in absolute_images):
                missing = next(image for image in absolute_images if not Path(image).is_file())
                raise FileNotFoundError(f"{case['id']}: missing image {missing}")

            if prompt_bundle is None:
                current_system = rollout_system_context(patient_context, history)
            else:
                task, _ = prompt_for_question(prompt_bundle, question["value"])
                context = rollout_system_context(patient_context, history, prompt_bundle["history_heading"])
                current_system = question_system(prompt_bundle, question["value"], context)
                print(f"{shard} question_type={task} rule_blocks={','.join(prompt_bundle['tasks'][task]['blocks'])}", flush=True)
            started = time.time()
            response = model.chat(
                [{"role": "user", "content": question["value"]}],
                system=current_system,
                images=absolute_images,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )[0]
            prediction = response.response_text.strip()
            if not prediction:
                raise RuntimeError(f"{case['id']}: empty response at turn {turn_index + 1}")
            history.append((question["value"], prediction))
            state["turns"].append(
                {
                    "turn_index": turn_index + 1,
                    "question": question["value"],
                    "images": turn_images,
                    "prediction": prediction,
                    "reference": reference["value"],
                    "finish_reason": response.finish_reason,
                    "response_length": response.response_length,
                    "prompt_length": response.prompt_length,
                    "seconds": round(time.time() - started, 1),
                }
            )
            state["completed_turns"] = len(state["turns"])
            write_json_atomic(path, state)
            done_turns += 1
            print(
                f"{shard}[case {case_index}/{len(cases)}][turns {done_turns}/{total_turns}] {case['id']} turn {turn_index + 1}/{len(pairs)} "
                f"prompt={response.prompt_length} generated={response.response_length} {response.finish_reason} "
                f"{state['turns'][-1]['seconds']}s",
                flush=True,
            )
            image_cursor += image_count

        state["status"] = "completed"
        write_json_atomic(path, state)
        write_answers(answers_path, state)

    print(f"{shard} finished; answers are under {args.output_root}/<group>/<name>/evaluation/{args.output_model}/", flush=True)


if __name__ == "__main__":
    main()
