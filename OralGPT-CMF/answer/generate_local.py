#!/usr/bin/env python3
"""Run deterministic, strict multi-turn SFT rollouts on the OralCMF test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.conversation import conversation_pairs, rollout_system_context, system_context
from answer.utils import load_state, patient_paths, rebuild_history, write_answers, write_json_atomic

LLAMA_ROOT = Path("/root/autodl-tmp/LLaMA-Factory-qwen35")
DEFAULT_DATASET = ROOT / "sft/data/turn_level" / "oralcmf_sft_test_rollout.json"
DEFAULT_ADAPTER = ROOT / "sft/models" / "qwen3_5_9b_lora_turn_level"
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
    parser.add_argument("--retry-length", action="store_true", help="Retry from the first truncated turn, keeping earlier answers")
    parser.add_argument("--question-prompts", type=Path, help="Inject only common and current-question rules from YAML")
    return parser.parse_args()


def run_case(args, model, case, state, prompt_bundle, prefix, done_turns, total_turns) -> int:
    from common.question_prompts import prompt_for_question, question_system

    path, answers_path = patient_paths(args.output_root, args.output_model, case["id"])
    decoding = {
        "do_sample": False,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.retry_length:
        first = next((i for i, turn in enumerate(state["turns"]) if turn["finish_reason"] == "length"), None)
        if first is not None:
            rebuild_history(case, state)
            for turn in state["turns"][:first]:
                turn.setdefault("decoding", state["decoding"])
            state["turns"] = state["turns"][:first]
            state["completed_turns"] = first
            state["status"] = "in_progress"
            state["decoding"] = decoding
            write_json_atomic(path, state)
            write_answers(answers_path, state)
    if state.get("turns") and state.get("decoding") != decoding:
        raise ValueError(f"Cannot resume {case['id']} with different decoding settings")
    state["decoding"] = decoding
    pairs = conversation_pairs(case)
    if state.get("status") == "completed":
        write_answers(answers_path, state)
        done_turns += len(pairs)
        print(f"{prefix}[turns {done_turns}/{total_turns}] {case['id']}: already complete", flush=True)
        return done_turns

    patient_context = system_context(case)
    history = rebuild_history(case, state)
    start_turn = len(state["turns"])
    done_turns += start_turn
    print(f"{prefix} {case['id']}: resume at turn {start_turn + 1}/{len(pairs)}", flush=True)
    image_cursor = sum(question["value"].count("<image>") for question, _ in pairs[:start_turn])

    for turn_index in range(start_turn, len(pairs)):
        question, reference = pairs[turn_index]
        image_count = question["value"].count("<image>")
        turn_images = case.get("images", [])[image_cursor : image_cursor + image_count]
        if len(turn_images) != image_count:
            raise ValueError(f"{case['id']}: image references do not match turn {turn_index}")
        absolute_images = [str(args.dataset.parent / image) for image in turn_images]
        for image in absolute_images:
            if not Path(image).is_file():
                raise FileNotFoundError(f"{case['id']}: missing image {image}")

        if prompt_bundle is None:
            current_system = rollout_system_context(patient_context, history)
        else:
            task, _ = prompt_for_question(prompt_bundle, question["value"])
            context = rollout_system_context(patient_context, history, prompt_bundle["history_heading"])
            current_system = question_system(prompt_bundle, question["value"], context)
            print(f"{prefix} question_type={task} rule_blocks={','.join(prompt_bundle['tasks'][task]['blocks'])}", flush=True)
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
                "decoding": decoding,
            }
        )
        state["completed_turns"] = len(state["turns"])
        write_json_atomic(path, state)
        done_turns += 1
        print(
            f"{prefix}[turns {done_turns}/{total_turns}] {case['id']} turn {turn_index + 1}/{len(pairs)} "
            f"prompt={response.prompt_length} generated={response.response_length} {response.finish_reason} "
            f"{state['turns'][-1]['seconds']}s",
            flush=True,
        )
        image_cursor += image_count

    state["status"] = "completed"
    write_json_atomic(path, state)
    write_answers(answers_path, state)
    return done_turns


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
    from common.question_prompts import bind_prompt_state, load_question_prompts, prompt_for_question

    prompt_bundle = load_question_prompts(args.question_prompts) if args.question_prompts else None
    if prompt_bundle is not None:
        for case in cases:
            for question, _ in conversation_pairs(case):
                prompt_for_question(prompt_bundle, question["value"])
    states = []
    for case in cases:
        state_path, _ = patient_paths(args.output_root, args.output_model, case["id"])
        state = load_state(state_path, case, args.overwrite)
        bind_prompt_state(state, prompt_bundle)
        states.append(state)

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
    total_turns = sum(len(conversation_pairs(case)) for case in cases)
    shard = f"[{args.output_model} shard {args.shard_index + 1}/{args.num_shards}]"
    print(f"{shard} selected {len(cases)} cases with {total_turns} question-answer turns.", flush=True)
    done_turns = 0
    for case_index, (case, state) in enumerate(zip(cases, states), start=1):
        prefix = f"{shard}[case {case_index}/{len(cases)}]"
        done_turns = run_case(args, model, case, state, prompt_bundle, prefix, done_turns, total_turns)

    print(f"{shard} finished; answers are under {args.output_root}/<group>/<name>/evaluation/{args.output_model}/", flush=True)


if __name__ == "__main__":
    main()
