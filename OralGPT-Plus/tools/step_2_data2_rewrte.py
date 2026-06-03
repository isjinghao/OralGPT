import os
import re
import json
import time
import http.client
import argparse
from typing import List, Dict, Any, Optional, Union

DISEASE_DEFS = {
    "Abnormal tooth development": (
        "On a panoramic X-ray, abnormal tooth development can be visually identified through a wide range of anomalies affecting tooth number, size, shape, and structure. This includes the presence of extra teeth (supernumerary) or congenitally missing teeth (hypodontia), as well as teeth that are abnormally large (macrodontia) or small (microdontia). X-rays can also reveal unusual shapes, such as sharp bends in a root (dilaceration), a \"tooth within a tooth\" appearance (dens in dente), or the fusion of a tooth directly to the bone (ankylosis), which appears as a loss of the dark line surrounding the root. Furthermore, teeth may be visibly impacted or unerupted within the jawbone, or show signs of structural defects like enlarged pulp chambers (taurodontism), all of which are critical findings for a dental professional to diagnose and manage."
    ),
    "caries": (
        "In a panoramic X-ray, caries appears as a dark, radiolucent area within the tooth structure. The size and appearance vary by severity, with shallow caries showing as a radiolucency in the enamel or outer dentin, moderate caries in the middle third of the dentin, and deep caries extending to the inner dentin, possibly involving the pulp. The exact appearance can be challenging to detect, especially for small or early lesions, and is complicated by overlying anatomy and image quality, making other X-ray types like bitewings often preferred for detailed caries detection."
    ),
    "deep pits and fissures": (
        "Deep pits and fissures are difficult to detect on panoramic X-rays because the X-ray beam can't fully penetrate the deep grooves, making them appear normal or obscured by overlapping structures. While carious lesions typically show as darker, less dense areas, the complex anatomy of pits and fissures, combined with factors like overlapping teeth and anatomical artifacts, can mask early decay. Clinically, these areas are high-risk zones for decay due to plaque accumulation, and a panoramic X-ray may only show the lesion once it has progressed deeper into the dentin."
    ),
    "periapical periodontitis": (
        "In a panoramic X-ray, periapical periodontitis appears as a radiolucent area (a darker area) around the tip of a tooth's root. This can be seen as a widening of the periodontal ligament space or a well-defined lesion. However, it is often hard to see in panoramic images due to overlapping anatomical structures, and sometimes only shows a slight thickening of the periodontal ligament or a loss of the lamina dura."
    ),
    "pulpitis": (
        "Pulpitis's effects can be visualized as signs of the underlying condition. Chronic pulpitis may appear as a widened periodontal ligament space or a discontinuity of the lamina dura, while advanced stages can cause a periapical radiolucency in the bone surrounding the tooth's root. In some cases, the bone may become denser, a condition known as condensing osteitis, which is also visible on the X-ray."
    ),
}

# Lowercase matchers
DISEASE_MATCHERS = {
    "Abnormal tooth development": [
        r"\babnormal (tooth|teeth) development\b",
        r"\bsupernumerary\b",
        r"\bhypodontia\b",
        r"\bmacrodontia\b",
        r"\bmicrodontia\b",
        r"\bdilaceration\b",
        r"\bdens in dente\b",
        r"\bankylosis\b",
        r"\btaurodontism\b",
        r"\bimpacted\b",
        r"\bunerupted\b",
    ],
    "caries": [
        r"\bcaries\b",
        r"\bcarious lesion\b",
        r"\bcavity\b",
        r"\btooth decay\b",
    ],
    "deep pits and fissures": [
        r"\bdeep pits? and fissures?\b",
        r"\bpits? and fissures?\b",
        r"\bfissures?\b",
    ],
    "periapical periodontitis": [
        r"\bperiapical periodontitis\b",
        r"\bapical periodontitis\b",
        r"\bperiapical lesion\b",
        r"\bperiapical radiolucenc(?:y|ies)\b",
    ],
    "pulpitis": [
        r"\bpulpitis\b",
        r"\birreversible pulpitis\b",
        r"\breversible pulpitis\b",
    ],
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite dental panoramic X-ray reasoning dialogue data via GPT API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY", ""),
        help="GPT API key"
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default="jeniya.top",
        help="GPT API host"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GPT_MODEL", "gpt-5-mini-2025-08-07"),
        help="Model name"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/multi_round_output_data2/multi_round_data.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/multi_round_output_data2/multi_round_data2_step_2.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/multi_round_output_data2/checkpoint.json",
        help="Checkpoint file path"
    )

    parser.add_argument(
        "--total-shards",
        type=int,
        default=1,
        help="Total number of shards"
    )
    parser.add_argument(
        "--current-shard",
        type=int,
        default=0,
        help="Current shard index (starting from 0)"
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=2,
        help="Save a checkpoint every N processed samples"
    )
    parser.add_argument(
        "--max-sample-retries",
        type=int,
        default=3,
        help="Max retries per sample"
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help="Max retries per API call"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Max tokens per API call"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="API call timeout in seconds"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="API call temperature"
    )

    return parser.parse_args()


def normalize_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)

def detect_diseases_in_text(text: str) -> List[str]:
    t = (text or "").lower()
    hits = []
    for name, pats in DISEASE_MATCHERS.items():
        if any(re.search(p, t) for p in pats):
            hits.append(name)
    return hits

def build_disease_context_for_turn(content: str) -> Dict[str, str]:
    names = detect_diseases_in_text(content)
    return {n: DISEASE_DEFS[n] for n in names} if names else {}

def has_answer_tag(content: str) -> bool:
    return bool(re.search(r"<\s*answer\s*>", content or "", flags=re.I))

def has_tool_call_tag(content: str) -> bool:
    return bool(re.search(r"<\s*tool_call\s*>", content or "", flags=re.I))

def extract_tool_call_block(content: str) -> Optional[str]:
    """Extract <tool_call> ... </tool_call> verbatim if present."""
    if not content:
        return None
    m = re.search(r"(<\s*tool_call\s*>.*?</\s*tool_call\s*>)", content, flags=re.I | re.S)
    return m.group(1) if m else None

def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def get_response_schema(contains_answer: bool) -> Dict[str, Any]:
    """Build a strict JSON Schema to enforce the required output format."""
    if contains_answer:
        return {
            "type": "object",
            "properties": {
                "think": {
                    "type": "string",
                    "description": "The rewritten <think> block with tags included, as a natural paragraph"
                },
                "answer": {
                    "type": "string",
                    "description": "The rewritten <answer> block with tags included, concise and fluent"
                }
            },
            "required": ["think", "answer"],
            "additionalProperties": False
        }
    else:
        return {
            "type": "object",
            "properties": {
                "think": {
                    "type": "string",
                    "description": "The rewritten <think> block with tags included, as a natural paragraph"
                }
            },
            "required": ["think"],
            "additionalProperties": False
        }

def make_assistant_turn_messages(assistant_content: str,
                                 gpt4_analysis: Optional[str],
                                 disease_ctx: Dict[str, str],
                                 contains_answer: bool,
                                 contains_toolcall: bool) -> List[Dict[str, Any]]:
    """Build the system + user messages to rephrase a single assistant turn.
    If gpt-4-analysis is provided, its key insights should be woven naturally
    into the rewritten <think> (and <answer>) rather than mechanically appended.
    """
    sys_text = (
        "You are a senior dentist. Your task is to REPHRASE a single assistant turn from a multi-round dental panoramic "
        "reasoning dialogue. Rewrite only the `<think>` (and `<answer>` if present). Keep medical accuracy, clarity, and coherence.\n\n"
        "CRITICAL RULES:\n"
        "1) NEVER modify or remove any `<tool_call>` block; it belongs to the NEXT round and must remain untouched.\n"
        "2) The current round `<think>` describes findings from the PREVIOUS round's zoom region (i.e., it must NOT rely on the "
        "   bbox inside the current `<tool_call>`).\n"
        "3) If multiple regions/teeth in THIS round share the same disease, MERGE them into ONE sentence listing the regions/teeth.\n"
        "4) If diseases were detected in this round, you may use the provided scoped disease definitions to improve terminology. "
        "   Do NOT invent new diseases.\n"
        "5) If `<answer>` exists, rewrite it too: consolidate and de-duplicate; keep it concise and fluent.\n"
        "6) Do NOT fabricate bbox, coordinates, teeth IDs, or tools. Do NOT add or remove tags.\n"
        "7) If `gpt_4_analysis` is provided (non-empty), you MUST weave its salient insights into the rewritten `<think>` "
        "   (and into `<answer>` if it helps the final summary). Integrate naturally as part of your reasoning—paraphrase and fuse "
        "   it with the observed radiographic evidence; avoid quoting verbatim, listing bullets, or creating a separate section.\n\n"
        "STYLE:\n"
        "- The `<think>` must be a single, natural paragraph (no bullets or numbering), sounding like a coherent inner monologue.\n"
        "- Keep language consistent with the input (English stays English, Chinese stays Chinese).\n"
        "- INCLUDE the XML tags in your output: <think>...</think> and <answer>...</answer> (if applicable).\n\n"
        "OUTPUT FORMAT:\n"
        "You MUST respond with ONLY a valid JSON object (no other text, no markdown code blocks).\n"
    )

    if contains_answer:
        sys_text += (
            "- Your JSON must have exactly TWO keys: \"think\" and \"answer\".\n"
            "- Example: {\"think\":\"<think>Natural paragraph with integrated analysis...</think>\", "
            "\"answer\":\"<answer>Concise, deduplicated summary (may reflect integrated analysis)...</answer>\"}\n"
        )
    else:
        sys_text += (
            "- Your JSON must have exactly ONE key: \"think\".\n"
            "- Example: {\"think\":\"<think>Natural paragraph with integrated analysis...</think>\"}\n"
        )

    usr_payload = {
        "assistant_turn": {
            "content": assistant_content
        },
        "scoped_disease_definitions": disease_ctx,
        "gpt_4_analysis": gpt4_analysis or "",
        "gpt_4_analysis_present": bool(gpt4_analysis and gpt4_analysis.strip()),
        "contains_answer": contains_answer,
        "contains_tool_call": contains_toolcall,
        "instructions": {
            "merge_same_disease_regions": True,
            "keep_tool_call_verbatim": True,
            "tool_call_bbox_is_for_next_round": True,
            "include_xml_tags_in_output": True,
            "integrate_gpt4_analysis_naturally": True,
            "avoid_verbatim_from_gpt4_analysis": True,
            "no_bullets_in_think": True
        }
    }

    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": compact_json(usr_payload)},
    ]
    return messages

def call_gpt_messages(messages: List[Dict[str, Any]],
                      contains_answer: bool,
                      args,
                      max_tokens: int = None,
                      retries: int = None,
                      timeout: int = None) -> str:
    """Call the API and force a JSON response."""
    if max_tokens is None:
        max_tokens = args.max_tokens
    if retries is None:
        retries = args.max_api_retries
    if timeout is None:
        timeout = args.timeout

    payload_dict = {
        "model": args.model,
        "stream": False,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}
    }

    payload = json.dumps(payload_dict)
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {args.api_key}',
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  API call ({attempt}/{retries})")
            conn = http.client.HTTPSConnection(args.api_host, timeout=timeout)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            resp = conn.getresponse()
            data = resp.read().decode("utf-8", errors="ignore")
            conn.close()

            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status}: {data[:500]}")

            j = json.loads(data)
            choices = j.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            msg = choices[0].get("message", {})
            content = msg.get("content") or msg.get("text")
            if isinstance(content, list):
                content = "\n".join([c if isinstance(c, str) else c.get("text", "") for c in content])
            content = (content or "").strip()
            if not content:
                raise ValueError("Empty response content")

            try:
                parsed = parse_turn_json(content, contains_answer)
                return content
            except Exception as parse_err:
                raise ValueError(f"Response does not match expected JSON format: {parse_err}")

        except Exception as e:
            last_err = e
            print(f"  Failed: {e}")
            if attempt < retries:
                backoff = min(2 ** attempt, 15)
                print(f"  Waiting {backoff}s before retry...")
                time.sleep(backoff)

    raise last_err if last_err else RuntimeError("API call failed")

def parse_turn_json(raw: str, contains_answer: bool) -> Dict[str, str]:
    """Parse and validate the JSON returned by the model."""
    if not raw:
        raise ValueError("Empty response content")

    raw = raw.strip()

    raw = re.sub(r'^```(?:json|JSON)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse as JSON: {e}. Raw content: {raw[:200]}")

    if not isinstance(obj, dict):
        raise ValueError(f"JSON must be an object, not {type(obj).__name__}")

    if "think" not in obj:
        raise ValueError(f"JSON missing required field 'think'. Actual fields: {list(obj.keys())}")

    if contains_answer and "answer" not in obj:
        raise ValueError(f"This turn contains <answer>, JSON must include 'answer'. Actual fields: {list(obj.keys())}")

    if not isinstance(obj["think"], str):
        raise ValueError(f"'think' must be a string, not {type(obj['think']).__name__}")

    if contains_answer and not isinstance(obj["answer"], str):
        raise ValueError(f"'answer' must be a string, not {type(obj['answer']).__name__}")

    think_content = obj["think"].strip()
    if not re.search(r'<\s*think\s*>', think_content, flags=re.I):
        raise ValueError("'think' field must contain a <think> opening tag")
    if not re.search(r'</\s*think\s*>', think_content, flags=re.I):
        raise ValueError("'think' field must contain a </think> closing tag")

    if contains_answer:
        answer_content = obj["answer"].strip()
        if not re.search(r'<\s*answer\s*>', answer_content, flags=re.I):
            raise ValueError("'answer' field must contain an <answer> opening tag")
        if not re.search(r'</\s*answer\s*>', answer_content, flags=re.I):
            raise ValueError("'answer' field must contain an </answer> closing tag")

    expected_keys = {"think", "answer"} if contains_answer else {"think"}
    extra_keys = set(obj.keys()) - expected_keys
    if extra_keys:
        print(f"  Warning: JSON contains extra fields: {extra_keys}")

    return obj

def rewrite_assistant_message_content(content: str, gpt4_analysis: Optional[str], args) -> str:
    """Rewrite only the <think> (and <answer> if present) of the current assistant content.
    The <tool_call> block is kept verbatim and reattached by us.
    """
    contains_answer = has_answer_tag(content)
    contains_toolcall = has_tool_call_tag(content)
    tool_block = extract_tool_call_block(content) if contains_toolcall else None

    disease_ctx = build_disease_context_for_turn(content)

    messages = make_assistant_turn_messages(
        assistant_content=content,
        gpt4_analysis=gpt4_analysis,
        disease_ctx=disease_ctx,
        contains_answer=contains_answer,
        contains_toolcall=contains_toolcall,
    )

    raw = call_gpt_messages(messages, contains_answer=contains_answer, args=args)

    obj = parse_turn_json(raw, contains_answer)

    new_think = obj["think"].strip()

    parts = [new_think]

    if contains_answer:
        new_answer = obj["answer"].strip()
        parts.append(new_answer)
    elif contains_toolcall and tool_block:
        parts.append(tool_block)

    return "\n".join(parts).strip()

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            print(f"Loaded checkpoint: {len(ckpt.get('processed_indices', []))} samples processed")
            return ckpt
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting from scratch")
    return {"processed_indices": [], "results": []}

def save_checkpoint(checkpoint_path: str, processed_indices: List[int], results: List[Dict[str, Any]]):
    ckpt = {
        "processed_indices": processed_indices,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)

def save_output(output_path: str, results: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Output saved to: {output_path}")

def get_shard_indices(total_samples: int, total_shards: int, current_shard: int) -> List[int]:
    if total_shards <= 0 or current_shard < 0 or current_shard >= total_shards:
        raise ValueError(f"Invalid shard: total_shards={total_shards}, current_shard={current_shard}")
    shard_size = total_samples // total_shards
    remainder = total_samples % total_shards
    if current_shard < remainder:
        start = current_shard * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = current_shard * shard_size + remainder
        end = start + shard_size
    return list(range(start, end))

def process_single_sample(sample: Dict[str, Any], idx: int, args) -> Dict[str, Any]:
    """Iterate over sample["messages"]; for each role=assistant message, rewrite only
    its <think> (and <answer> if present), keeping other fields unchanged and the
    <tool_call> block verbatim.
    """
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("sample['messages'] must be a list")

    new_messages = []
    for mi, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            new_messages.append(m)
            continue

        original_content = normalize_str(m.get("content", ""))
        gpt4_analysis = m.get("gpt-4-analysis")

        try:
            print(f"    Processing message {mi}...")
            rewritten = rewrite_assistant_message_content(original_content, gpt4_analysis, args)
            print(f"    Message {mi} rewritten successfully")
        except Exception as e:
            print(f"  Assistant message rewrite failed (sample #{idx}, message {mi}): {e}")
            print(f"    Keeping original text")
            rewritten = original_content

        new_m = dict(m)
        new_m["content"] = rewritten
        new_messages.append(new_m)

    return {
        "messages": new_messages,
        "images": sample.get("images", []),
        "metadata": sample.get("metadata", {}),
    }

def process_all(args):
    print("Starting...")
    print(f"Configuration:")
    print(f"  - API Host: {args.api_host}")
    print(f"  - Model: {args.model}")
    print(f"  - Shard: {args.current_shard + 1}/{args.total_shards}")
    print(f"  - Max sample retries: {args.max_sample_retries}")
    print(f"  - Max API retries: {args.max_api_retries}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max Tokens: {args.max_tokens}")
    print(f"  - Timeout: {args.timeout}s")

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Top-level input JSON must be a list."

    total = len(data)
    print(f"Total samples: {total}")

    shard_indices = get_shard_indices(total, args.total_shards, args.current_shard)
    print(f"Current shard indices: {shard_indices[0]} - {shard_indices[-1]} ({len(shard_indices)} total)")

    ckpt = load_checkpoint(args.checkpoint_path)
    done_set = set(ckpt.get("processed_indices", []))
    results_dict = {i: s for i, s in zip(ckpt.get("processed_indices", []), ckpt.get("results", []))}

    pending = [i for i in shard_indices if i not in done_set]
    print(f"Skipped {len(shard_indices) - len(pending)} already completed samples")
    print(f"Pending {len(pending)} samples")

    processed_count = 0
    failed_samples = []

    for count, idx in enumerate(pending, 1):
        sample = data[idx]
        print(f"\n{'='*60}")
        print(f"[{count}/{len(pending)}] Processing sample #{idx}")
        print(f"{'='*60}")

        last_err = None
        for attempt in range(1, args.max_sample_retries + 1):
            try:
                if attempt > 1:
                    print(f"  Sample-level retry {attempt}/{args.max_sample_retries}")
                fixed = process_single_sample(sample, idx, args)
                results_dict[idx] = fixed
                done_set.add(idx)
                if attempt > 1:
                    print(f"  Sample #{idx} succeeded on retry (attempt {attempt})")
                else:
                    print(f"  Sample #{idx} processed successfully")
                break
            except Exception as e:
                last_err = e
                print(f"  Sample #{idx} failed (attempt {attempt}/{args.max_sample_retries}): {e}")
                if attempt < args.max_sample_retries:
                    backoff = min(2 ** attempt, 20)
                    print(f"  Waiting {backoff}s before retrying sample...")
                    time.sleep(backoff)

        if last_err and idx not in results_dict:
            print(f"Sample #{idx} ultimately failed, skipped")
            failed_samples.append(idx)

        processed_count += 1
        if processed_count % args.save_interval == 0:
            sorted_idx = sorted(done_set)
            sorted_res = [results_dict[i] for i in sorted_idx]
            save_checkpoint(args.checkpoint_path, sorted_idx, sorted_res)
            print(f"\nCheckpoint saved ({len(sorted_idx)}/{len(shard_indices)} completed)")

    sorted_idx = sorted(done_set)
    sorted_res = [results_dict[i] for i in sorted_idx]
    save_checkpoint(args.checkpoint_path, sorted_idx, sorted_res)
    save_output(args.output_path, sorted_res)

    print("\n" + "="*60)
    print("Done")
    print("="*60)
    print(f"Statistics:")
    print(f"  - Successful samples: {len(sorted_res)}/{len(shard_indices)}")
    print(f"  - Failed samples: {len(failed_samples)}")
    if failed_samples:
        print(f"  - Failed sample indices: {failed_samples}")
    print(f"  - Success rate: {len(sorted_res)/len(shard_indices)*100:.2f}%")

if __name__ == "__main__":
    args = parse_args()
    process_all(args)
