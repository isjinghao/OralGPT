import os
import re
import json
import time
import http.client
import argparse
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import mimetypes

DISEASE_DEFS = {
    # From data4
    "Carious lesion": (
        "In a panoramic X-ray, a carious lesion typically appears as a radiolucent (dark) area, indicating demineralization of the tooth structure. The shape can vary, but a common appearance for lesions on proximal (between teeth) surfaces is a triangle with its base at the tooth's surface. Less distinct lesions may appear as dots, lines, bands, or notches, and mild or early lesions can be subtle and easily confused with shadows or optical illusions. "
    ),
    "Apical periodontitis": (
        "In a panoramic X-ray, apical periodontitis appears as a periapical radiolucency, which is a dark or \"lucid\" area at the tip of a tooth root. This area of bone loss can manifest as a widened periodontal ligament or a more defined lesion, and it indicates a chronic inflammatory response to a bacterial infection within the tooth's pulp. However, these lesions can be difficult to see due to superimposed anatomy, and their absence doesn't rule out the condition, so further imaging like Cone-beam computed tomography (CBCT) may be needed for a definitive diagnosis."
    ),
    "Furcation lesion": (
        "In a panoramic X-ray, a furcation lesion appears as a darker, radiolucent area between the roots of a multi-rooted tooth, which can indicate bone loss or a widened periodontal ligament. The lesion may be visible as a widening of the periodontal ligament space or a more pronounced bone defect, but panoramic images have limitations and may not clearly show the extent of the lesion, particularly for early-stage involvement, requiring a dentist's clinical examination for confirmation."
    ),
    "Root resorption": (
        "In a panoramic X-ray, root resorption appears as a radiolucent (dark) area on or within the root. The specific appearance can vary: external resorption often presents with an irregular root outline, while internal resorption may appear as a more symmetrical and uniform enlargement of the pulp canal. Internal replacement resorption can look like a mottled, cloud-like defect within the root."
    ),
    "Root fragment": (
        "A root fragment appears as a distinct, usually irregular-shaped, radiopaque (white) fragment or shadow on a panoramic X-ray. It is often seen separated from the main tooth root and may be embedded in the jawbone. The appearance can vary depending on the size and location of the fragment, and it is often associated with other signs of trauma or a previously damaged tooth."
    ),
    "Bone resorption": (
        "In a panoramic X-ray, bone resorption appears as a loss or thinning of the bone's structure, most noticeably the alveolar bone crest. Other visual signs include a thinner mandibular cortex, a less-defined or less dense appearance of the bone's surface, and sometimes an \"onionskin\" appearance of the cortical bone. These changes can indicate a decreased bone mineral density or other oral pathologies."
    ),
    # From data2
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
    # From data3
    "impacted": (
        "An impacted tooth appears as a tooth that is abnormally positioned, often still within the jawbone, making it visible in a panoramic X-ray as an outline against the surrounding bone. Its appearance can be affected by factors like superimposition of other structures, which can cause a \"ghost image\" or distortion, making precise localization and detailed evaluation difficult. This can also lead to visual artifacts, including blurriness or unclear boundaries between the tooth and surrounding bone."
    ),
    "periapical lesion": (
        "In a panoramic X-ray, a periapical lesion appears as a dark, irregular area (radiolucency) at the root tip of a tooth. However, panoramic images often have limitations, such as overlapping structures and lower resolution, which can make it difficult to see small lesions, sometimes requiring additional imaging like a periapical X-ray or a cone-beam computed tomography (CBCT) scan for a definitive diagnosis."
    ),
}

# Lowercase regex matchers per disease.
DISEASE_MATCHERS = {
    # From data4
    "Carious lesion": [r"\bcarious lesion\b", r"\bcaries\b", r"\bcavity\b"],
    "Apical periodontitis": [r"\bapical periodontitis\b"],
    "Furcation lesion": [r"\bfurcation lesion\b"],
    "Root resorption": [r"\broot resorption\b", r"\bresorption of root\b"],
    "Root fragment": [r"\broot fragment\b", r"\bretained root\b"],
    "Bone resorption": [r"\bbone resorption\b", r"\bbone resorbtion\b"],
    # From data2
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
        r"\bcavities\b",
        r"\btooth decay\b",
        r"\bdecay\b",
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
    # From data3
    "impacted": [r"\bimpacted\b", r"\bimpacted tooth\b", r"\bimpacted teeth\b"],
    "periapical lesion": [r"\bperiapical lesion\b", r"\bperiapical\b", r"\bapical lesion\b"],
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite dental panoramic X-ray reasoning dialogue data via the GPT API.",
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
        help="Model name to use"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/final_data/mmoral_r1_codestart_mmoral_15k.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Optional output JSON file path (if empty, path is generated from n)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/final_data/checkpoint.json",
        help="Checkpoint file path"
    )

    parser.add_argument(
        "--total-shards",
        type=int,
        default=5,
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
        help="Maximum retries per sample"
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help="Maximum retries per API call"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum tokens per API call"
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
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of parallel worker threads"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples to rewrite this run (0 means no limit)"
    )
    parser.add_argument(
        "--max-images-per-sample",
        type=int,
        default=1,
        help="Maximum images attached per sample"
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
    """Extract the <tool_call> ... </tool_call> block verbatim if present."""
    if not content:
        return None
    m = re.search(r"(<\s*tool_call\s*>.*?</\s*tool_call\s*>)", content, flags=re.I | re.S)
    return m.group(1) if m else None

def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def get_response_schema() -> Dict[str, Any]:
    """Return a strict JSON schema containing only the think field."""
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
                                 disease_ctx: Dict[str, str],
                                 image_data_urls: List[str]) -> List[Dict[str, Any]]:
    """Build the system + user messages to rephrase one assistant turn.

    Only the <think> ... </think> block is rewritten; images are passed along to the model.
    """
    sys_text = (
        "You are a senior dentist. Your task is to REPHRASE a single assistant turn from a dental panoramic reasoning dialogue.\n"
        "Rewrite ONLY the <think>...</think> block so that it naturally narrates the visual reasoning and smoothly leads to and corroborates the existing <answer> in this turn.\n"
        "The goal is to produce two or three natural sentences that helps another model learn how the image-based reasoning supports the final answer (without changing the <answer> itself).\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT modify anything outside <think>...</think> (keep <answer>, any other tags, and text untouched).\n"
        "- Use ONLY the provided scoped disease definitions (scoped_disease_definitions) if any; do NOT perform your own disease matching and do NOT invent new diseases.\n"
        "- If no scoped disease definitions are provided, avoid introducing specific disease names; stick to general radiographic observations.\n"
        "- The <think> must be ONE natural sentence (no bullets/numbering), coherent, medically accurate, and consistent with the existing <answer> (no contradictions).\n"
        "- Keep the original language (English stays English, Chinese stays Chinese).\n\n"
        "OUTPUT FORMAT:\n"
        "Respond with ONLY a JSON object with exactly one key: think. Include the <think>...</think> tags in the value."
    )

    usr_text = compact_json({
        "assistant_turn": {"content": assistant_content},
        "scoped_disease_definitions": disease_ctx,
        "instructions": {
            "rewrite_only_think": True,
            "no_bullets_in_think": True,
            "preserve_other_tags": True,
            "align_with_existing_answer": True,
            "use_only_scoped_disease_definitions": True,
            "no_self_matching_or_new_diseases": True
        },
        "response_schema": get_response_schema()
    })

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": usr_text}
    ]
    for url in image_data_urls:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_content},
    ]
    return messages

def call_gpt_messages(messages: List[Dict[str, Any]],
                      args,
                      max_tokens: int = None,
                      retries: int = None,
                      timeout: int = None) -> str:
    """Call the API and require a JSON-formatted response."""
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
                raise ValueError("Response has no choices")
            msg = choices[0].get("message", {})
            content = msg.get("content") or msg.get("text")
            if isinstance(content, list):
                content = "\n".join([c if isinstance(c, str) else c.get("text", "") for c in content])
            content = (content or "").strip()
            if not content:
                raise ValueError("Empty response content")

            # Validate that the content is valid JSON before returning it.
            try:
                _ = parse_turn_json(content)
                return content
            except Exception as parse_err:
                raise ValueError(f"Response content is not valid JSON: {parse_err}")

        except Exception as e:
            last_err = e
            print(f"  Failed: {e}")
            if attempt < retries:
                backoff = min(2 ** attempt, 15)
                print(f"  Waiting {backoff}s before retry...")
                time.sleep(backoff)

    raise last_err if last_err else RuntimeError("API call failed")

def parse_turn_json(raw: str) -> Dict[str, str]:
    """Parse and validate the model's JSON response, checking only the think field."""
    if not raw:
        raise ValueError("Empty response content")

    raw = raw.strip()

    # Strip optional markdown code-block fences.
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
        raise ValueError(f"JSON missing required field 'think'. Got fields: {list(obj.keys())}")

    if not isinstance(obj["think"], str):
        raise ValueError(f"'think' must be a string, not {type(obj['think']).__name__}")

    think_content = obj["think"].strip()
    if not re.search(r'<\s*think\s*>', think_content, flags=re.I):
        raise ValueError("'think' field must contain an opening <think> tag")
    if not re.search(r'</\s*think\s*>', think_content, flags=re.I):
        raise ValueError("'think' field must contain a closing </think> tag")

    expected_keys = {"think"}
    extra_keys = set(obj.keys()) - expected_keys
    if extra_keys:
        print(f"  Warning: JSON contains extra fields: {extra_keys}")

    return obj

def _encode_image_to_data_url(path: str) -> Optional[str]:
    """Read a local image and convert it to a data URL, or None on failure."""
    try:
        if not os.path.exists(path):
            return None
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as _:
        return None

def _replace_think_block(original: str, think_with_tags: str) -> str:
    """Replace the <think>...</think> block in the original, or prepend it if absent."""
    if re.search(r"<\s*think\s*>.*?</\s*think\s*>", original or "", flags=re.I | re.S):
        return re.sub(r"<\s*think\s*>.*?</\s*think\s*>", think_with_tags, original, flags=re.I | re.S)
    return f"{think_with_tags}\n" + (original or "")

def rewrite_assistant_message_content(content: str, images: List[str], args) -> str:
    """Rewrite only the <think> part of the assistant content, leaving the rest intact."""
    disease_ctx = build_disease_context_for_turn(content)

    # Attach images up to the configured per-sample limit.
    image_urls: List[str] = []
    for p in images[: max(0, args.max_images_per_sample) or 0]:
        url = _encode_image_to_data_url(p)
        if url:
            image_urls.append(url)

    messages = make_assistant_turn_messages(
        assistant_content=content,
        disease_ctx=disease_ctx,
        image_data_urls=image_urls,
    )

    raw = call_gpt_messages(messages, args=args)
    obj = parse_turn_json(raw)
    new_think = obj["think"].strip()
    return _replace_think_block(content, new_think).strip()

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
    """Iterate over sample["conversations"] and rewrite only the <think> block of each
    assistant message. All other fields, including <tool_call>, are preserved.
    Uses the schema with conversations / from / value rather than messages / role / content.
    """
    conversations = sample.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("sample['conversations'] must be a list")

    new_conversations = []
    for mi, m in enumerate(conversations):
        if not isinstance(m, dict) or m.get("from") != "assistant":
            new_conversations.append(m)
            continue

        original_value = normalize_str(m.get("value", ""))
        sample_images: List[str] = sample.get("images", []) if isinstance(sample.get("images", []), list) else []

        try:
            print(f"    Processing turn {mi}...")
            rewritten = rewrite_assistant_message_content(original_value, sample_images, args)
            print(f"    Turn {mi} rewritten")
        except Exception as e:
            print(f"  Assistant turn rewrite failed (sample #{idx}, turn {mi}): {e}")
            print(f"    Keeping original text")
            rewritten = original_value

        new_m = dict(m)
        new_m["value"] = rewritten
        new_conversations.append(new_m)

    return {
        "conversations": new_conversations,
        "images": sample.get("images", []),
    }

def process_all(args):
    print("Starting...")
    print(f"Config:")
    print(f"  - API Host: {args.api_host}")
    print(f"  - Model: {args.model}")
    print(f"  - Shard: {args.current_shard + 1}/{args.total_shards}")
    print(f"  - Max sample retries: {args.max_sample_retries}")
    print(f"  - Max API retries: {args.max_api_retries}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Max Tokens: {args.max_tokens}")
    print(f"  - Timeout: {args.timeout}s")
    print(f"  - Threads: {args.num_threads}")
    print(f"  - Max samples this run: {args.max_samples or 'unlimited'}")
    print(f"  - Max images per sample: {args.max_images_per_sample}")

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
    if args.max_samples and args.max_samples > 0:
        pending = pending[: args.max_samples]
    print(f"Skipped {len(shard_indices) - len(pending)} already-completed samples")
    print(f"Pending {len(pending)} samples")

    processed_count = 0
    failed_samples: List[int] = []
    lock = None
    try:
        import threading
        lock = threading.Lock()
    except Exception:
        pass

    def _worker(i: int):
        sample = data[i]
        last_err = None
        for attempt in range(1, args.max_sample_retries + 1):
            try:
                if attempt > 1:
                    print(f"  Sample-level retry {attempt}/{args.max_sample_retries} (sample #{i})")
                fixed = process_single_sample(sample, i, args)
                return (i, fixed, None)
            except Exception as e:
                last_err = e
                print(f"  Sample #{i} failed (attempt {attempt}/{args.max_sample_retries}): {e}")
                if attempt < args.max_sample_retries:
                    backoff = min(2 ** attempt, 20)
                    print(f"  Waiting {backoff}s before retrying sample #{i}...")
                    time.sleep(backoff)
        return (i, None, last_err)

    with ThreadPoolExecutor(max_workers=max(1, args.num_threads)) as ex:
        futures = {ex.submit(_worker, idx): idx for idx in pending}
        for fi, fut in enumerate(as_completed(futures), 1):
            idx = futures[fut]
            try:
                i, fixed, err = fut.result()
                if err is None and fixed is not None:
                    if lock:
                        with lock:
                            results_dict[i] = fixed
                            done_set.add(i)
                            processed_count += 1
                    else:
                        results_dict[i] = fixed
                        done_set.add(i)
                        processed_count += 1
                    print(f"  Sample #{i} processed ({processed_count}/{len(pending)})")
                else:
                    if lock:
                        with lock:
                            failed_samples.append(i)
                    else:
                        failed_samples.append(i)
                    print(f"Sample #{i} permanently failed, skipped")
            except Exception as e:
                if lock:
                    with lock:
                        failed_samples.append(idx)
                else:
                    failed_samples.append(idx)
                print(f"Sample #{idx} exception: {e}")

            # Save a checkpoint periodically.
            if processed_count > 0 and processed_count % args.save_interval == 0:
                sorted_idx = sorted(done_set)
                sorted_res = [results_dict[i] for i in sorted_idx]
                save_checkpoint(args.checkpoint_path, sorted_idx, sorted_res)
                print(f"\nCheckpoint saved (completed {len(sorted_idx)}/{len(pending)})")

    sorted_idx = sorted(done_set)
    sorted_res = [results_dict[i] for i in sorted_idx]
    save_checkpoint(args.checkpoint_path, sorted_idx, sorted_res)

    # Compute the dynamic output path based on n.
    n = len(sorted_res)
    if args.output_path and args.output_path.strip():
        out_path = args.output_path
    else:
        out_dir = "/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/final_data"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"mmoral_r1_codestart_mmoral_{n}.json")
    save_output(out_path, sorted_res)

    # Remove processed samples from the source file.
    if sorted_idx:
        remaining = [s for i, s in enumerate(data) if i not in set(sorted_idx)]
        with open(args.input_path, "w", encoding="utf-8") as f:
            json.dump(remaining, f, ensure_ascii=False, indent=2)
        print(f"Removed {len(sorted_idx)} samples from source, {len(remaining)} remaining: {args.input_path}")

    print("\n" + "="*60)
    print("Done")
    print("="*60)
    print(f"Stats:")
    print(f"  - Succeeded: {len(sorted_res)}/{len(pending)}")
    print(f"  - Failed: {len(failed_samples)}")
    if failed_samples:
        print(f"  - Failed indices: {failed_samples}")
    print(f"  - Success rate: {len(sorted_res)/len(shard_indices)*100:.2f}%")

if __name__ == "__main__":
    args = parse_args()
    process_all(args)
