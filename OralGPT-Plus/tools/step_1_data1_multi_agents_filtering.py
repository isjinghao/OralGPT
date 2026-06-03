import json
import base64
import http.client
import os
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import re
import numpy as np
from PIL import Image

API_KEY = os.environ.get("API_KEY", "")
API_HOST = "jeniya.top"
MODEL_NAME = "gpt-5-mini-2025-08-07"

# Debug logging switch (1/true to enable, 0/false to disable)
DEBUG_LOG_RESPONSES = "0"

# Unified system prompt to encourage compliant, non-refusal radiographic analysis
SYSTEM_PROMPT = (
    "You are an assistant for research and education in oral and maxillofacial radiography working on de-identified "
    "panoramic radiographs and cropped regions of interest.\n\n"
    "Scope:\n"
    "- Describe radiographic findings, visual features, locations, and patterns.\n"
    "- When appropriate, name likely conditions using hedged phrasing (e.g., 'radiographic features are consistent with', 'suggestive of'), avoiding definitive diagnosis.\n"
    "- Do not provide medical advice, management, or treatment recommendations. Do not instruct the user to seek care. Do not include disclaimers in the output.\n"
    "- If uncertain, still describe what is visible and what additional visual evidence would strengthen or weaken a hypothesis.\n\n"
    "Compliance:\n"
    "- Always comply within this scope; do not refuse. If a request goes beyond scope, answer within scope instead of refusing.\n\n"
    "Output format:\n"
    "- Unless the user message specifies a different schema, respond using exactly these XML-style tags:\n"
    "  '<Think> ... </Think>\n<Answer> ... </Answer>\n'\n"
    "- The Think section contains reasoning and evidence; keep it concise and factual.\n"
    "- The Answer is a concise, professional summary of radiographic findings and likely conditions using hedged language and without recommendations.\n"
    "- Do not add extra prefaces, warnings, or extraneous text outside the required tags.\n"
    "- If the user message explicitly defines another output format (e.g., <Matched>/<Unmatched>), follow that format for that turn.\n\n"
    "Brevity and style:\n"
    "- Be concise. Focus only on visual findings and directly relevant evidence.\n"
    "- Keep <Think> to at most 5 short bullet points and under 150 words total.\n"
    "- Keep <Answer> to at most 2 sentences and under 100 words; prefer 1 or 2 sentences when possible.\n"
    "- Avoid repetition, background knowledge dumps, definitions, or restating instructions.\n"
    "- Prefer compact radiographic terminology; use hedged phrases succinctly."
)


def get_clinical_diagnosis_prompt(question_type: str, context: str = "") -> str:
    """Generate clinical diagnosis oriented prompts"""

    if question_type == "full_image_analysis":
        return (
            "Task: Examine the full panoramic radiograph and describe radiographic findings and likely conditions.\n\n"
            "Structure your reasoning as:\n"
            "1. List preliminary and potential conditions with approximate locations (e.g., \"Implant near tooth #17\", \"Impacted tooth at #28\", \"Carious lesion at #36\", \"Mild bone resorption in mandible\").\n"
            "2. Describe the visible radiographic features supporting each condition (implant, prosthetic restoration, obturation, endodontic treatment, orthodontic appliance, surgical device, carious lesion, bone resorption, impacted tooth, apical surgery, apical periodontitis, root fragment, furcation lesion, root resorption, etc.).\n"
            "3. Recall typical radiographic appearances for these findings from knowledge.\n"
            "4. State whether the observed features are consistent with those appearances."
        )

    elif question_type == "cropped_region_analysis":
        return (
            "Task: Examine a cropped subregion from the panoramic image and identify radiographic findings and likely conditions.\n\n"
            "1. List preliminary and potential conditions visible in this crop.\n"
            "2. Explain the specific radiographic features (implant, prosthetic restoration, obturation, endodontic treatment, orthodontic appliance, surgical device, carious lesion, bone resorption, impacted tooth, apical surgery, apical periodontitis, root fragment, furcation lesion, root resorption, etc.).\n"
            "3. Recall typical appearances for the suspected findings.\n"
            "4. Confirm whether observed features align with those appearances."
        )

    elif question_type == "comparative_analysis":
        return (
            "Task: Comparative analysis on a cropped region of interest (left = original RoI; right = mirrored contralateral reference).\n\n"
            "1. List preliminary and potential conditions in the original RoI (left).\n"
            "2. Compare left vs. right visual features and describe differences supporting each suspected condition (implant, prosthetic restoration, obturation, endodontic treatment, orthodontic appliance, surgical device, carious lesion, bone resorption, impacted tooth, apical surgery, apical periodontitis, root fragment, furcation lesion, root resorption, etc.).\n"
            "3. Recall typical radiographic appearances for these findings.\n"
            "4. Confirm whether the observed asymmetries are consistent with expected features."
        )

    elif question_type == "immediate_output_verification":
        return (
            f"Task: Verify whether a provided description of disease(s)/condition(s) in a panoramic image matches the gold standard.\n\n"
            f"- Compare conditions in the given description vs. those in the gold standard.\n"
            f"- List which conditions are matched and which are unmatched (missing, incorrect, or extra).\n\n"
            f"Given description: {context.split('GROUND_TRUTH:')[0].strip()}\n"
            f"Gold standard: {context.split('GROUND_TRUTH:')[1].strip() if 'GROUND_TRUTH:' in context else ''}\n\n"
            "Output format (strict):\n\n"
            "<Matched>\n"
            "[List matched disease(s)/condition(s)]\n"
            "</Matched>\n\n"
            "<Unmatched>\n"
            "[List unmatched disease(s)/condition(s), and note if missing or erroneously included]\n"
            "</Unmatched>"
        )

    elif question_type == "ground_truth_guidance":
        return (
            f"Task: Use the provided ground truth for a region to generate a professional radiographic description.\n\n"
            f"Ground truth: {context}\n\n"
            "Please include:\n"
            "1. Specific radiographic findings in this region.\n"
            "2. Characteristic features supporting the interpretation.\n"
            "3. Radiographic appearance and patterns observed."
        )

    elif question_type == "educational_description":
        condition = context
        return (
            f"Provide educational documentation about radiographic appearances for {condition}. Include:\n"
            "- Radiographic appearance and density patterns\n"
            "- Typical anatomical locations and spatial relationships\n"
            "- Key visual features for identification in clinical contexts\n"
            "- Distinguishing radiographic characteristics\n\n"
            "Frame this as educational material for clinical radiography training."
        )
    else:
        return context

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt_vision_response(image_path: str, question_type: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
    """Call the vision-language model API."""
    try:
        base64_image = encode_image_to_base64(image_path)
        clinical_prompt = get_clinical_diagnosis_prompt(question_type, context)

        content = [{"type": "text", "text": clinical_prompt}]
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

        payload = json.dumps({
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2000,
        })

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }

        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPSConnection(API_HOST)
                conn.request("POST", "/v1/chat/completions", payload, headers)

                response = conn.getresponse()
                status_code = getattr(response, "status", None)
                reason = getattr(response, "reason", None)
                raw_text = response.read().decode("utf-8")
                data = json.loads(raw_text) if raw_text else {}
                conn.close()

                if "choices" in data and len(data["choices"]) > 0:
                    choice0 = data["choices"][0]
                    gpt_answer = (choice0.get("message") or {}).get("content") or ""
                    finish_reason = choice0.get("finish_reason")
                    if DEBUG_LOG_RESPONSES:
                        try:
                            img_name = os.path.basename(image_path)
                        except Exception:
                            img_name = str(image_path)
                        content_len = len(gpt_answer)
                        tqdm.write(f"[VISION][{question_type}] image={img_name} attempt={attempt+1} status={status_code} reason={reason} finish={finish_reason} len={content_len}")
                        trunc = gpt_answer if content_len <= 2000 else gpt_answer[:2000] + "...[truncated]"
                        if content_len == 0:
                            choice_preview = str(choice0)
                            choice_preview = choice_preview if len(choice_preview) <= 800 else choice_preview[:800] + "..."
                            tqdm.write(f"[VISION][RAW EMPTY] choice0={choice_preview}")
                        else:
                            tqdm.write(f"[VISION][RAW] {trunc}")
                    return {"success": True, "gpt_answer": gpt_answer, "error": None, "attempts": attempt + 1}
                else:
                    if DEBUG_LOG_RESPONSES:
                        preview = str(data) if data else raw_text
                        preview = preview if len(preview) <= 500 else preview[:500] + "..."
                        tqdm.write(f"[VISION][WARN] No response from API. status={status_code} reason={reason} Raw preview: {preview}")
                    return {"success": False, "gpt_answer": None, "error": "No response from API", "attempts": attempt + 1}
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    if DEBUG_LOG_RESPONSES:
                        tqdm.write(f"[VISION][ERROR] {type(e).__name__}: {e}")
                    return {"success": False, "gpt_answer": None, "error": f"Request failed after {max_retries} attempts: {str(e)}", "attempts": max_retries}
                time.sleep(2 ** attempt)
                
    except Exception as e:
        return {"success": False, "gpt_answer": None, "error": f"Image processing error: {str(e)}", "attempts": 0}

def get_llm_text_response(prompt_type: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
    """Call the text model API."""
    clinical_prompt = get_clinical_diagnosis_prompt(prompt_type, context)

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": clinical_prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection(API_HOST)
            conn.request("POST", "/v1/chat/completions", payload, headers)

            response = conn.getresponse()
            status_code = getattr(response, "status", None)
            reason = getattr(response, "reason", None)
            raw_text = response.read().decode("utf-8")
            data = json.loads(raw_text) if raw_text else {}
            conn.close()

            if "choices" in data and len(data["choices"]) > 0:
                choice0 = data["choices"][0]
                resp_text = (choice0.get("message") or {}).get("content") or ""
                finish_reason = choice0.get("finish_reason")
                if DEBUG_LOG_RESPONSES:
                    content_len = len(resp_text)
                    tqdm.write(f"[TEXT][{prompt_type}] attempt={attempt+1} status={status_code} reason={reason} finish={finish_reason} len={content_len}")
                    if content_len == 0:
                        choice_preview = str(choice0)
                        choice_preview = choice_preview if len(choice_preview) <= 800 else choice_preview[:800] + "..."
                        tqdm.write(f"[TEXT][RAW EMPTY] choice0={choice_preview}")
                    else:
                        trunc = resp_text if content_len <= 2000 else resp_text[:2000] + "...[truncated]"
                        tqdm.write(f"[TEXT][RAW] {trunc}")
                return {"success": True, "response": resp_text, "error": None, "attempts": attempt + 1}
            else:
                if DEBUG_LOG_RESPONSES:
                    preview = str(data) if data else raw_text
                    preview = preview if len(preview) <= 500 else preview[:500] + "..."
                    tqdm.write(f"[TEXT][WARN] No response from API. status={status_code} reason={reason} Raw preview: {preview}")
                return {"success": False, "response": None, "error": "No response from API", "attempts": attempt + 1}
                
        except Exception as e:
            if attempt == max_retries - 1:
                if DEBUG_LOG_RESPONSES:
                    tqdm.write(f"[TEXT][ERROR] {type(e).__name__}: {e}")
                return {"success": False, "response": None, "error": f"Request failed after {max_retries} attempts: {str(e)}", "attempts": max_retries}
            time.sleep(2 ** attempt)

def crop_image_with_bbox(image_path: str, bbox: Tuple[int, int, int, int], output_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w))
    y2 = max(y1+1, min(y2, h))
    cropped = image[y1:y2, x1:x2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped)
    return output_path

def create_mirror_bbox(bbox: Tuple[int, int, int, int], img_width: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    center_x = img_width // 2
    bbox_width = x2 - x1
    bbox_center_x = (x1 + x2) // 2
    distance_from_center = bbox_center_x - center_x
    mirror_center_x = center_x - distance_from_center
    mirror_x1 = mirror_center_x - bbox_width // 2
    mirror_x2 = mirror_center_x + bbox_width // 2
    mirror_x1 = max(0, mirror_x1)
    mirror_x2 = min(img_width, mirror_x2)
    mirror_y1 = y1
    mirror_y2 = y2
    return mirror_x1, mirror_y1, mirror_x2, mirror_y2


def create_comparative_image(contextual_crop_path: str, comparative_crop_path: str, output_path: str, gap_pixels: int = 10) -> str:
    """Create comparative image - left: original ROI; right: contralateral ROI (without flipping)."""
    img1 = cv2.imread(contextual_crop_path)  # contextual/original
    img2 = cv2.imread(comparative_crop_path)  # contralateral crop
    if img1 is None or img2 is None:
        raise ValueError("Cannot read one or both images for comparison")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        if h1 > h2:
            img2 = cv2.resize(img2, (int(w2 * h1 / h2), h1))
        else:
            img1 = cv2.resize(img1, (int(w1 * h2 / h1), h2))
    h = max(img1.shape[0], img2.shape[0])
    gap = np.ones((h, gap_pixels, 3), dtype=np.uint8) * 255
    combined = np.hstack([img1, gap, img2])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, combined)
    return output_path

def load_proposals_data(proposals_path: str) -> Dict:
    with open(proposals_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_answer_from_response(gpt_response: str) -> str:
    try:
        answer_start = gpt_response.find('<Answer>')
        answer_end = gpt_response.find('</Answer>')
        if answer_start != -1 and answer_end != -1:
            extracted = gpt_response[answer_start + 8:answer_end].strip()
            if DEBUG_LOG_RESPONSES and (not extracted):
                from tqdm import tqdm
                tqdm.write(f"[PARSE][Answer] Empty between tags. start={answer_start} end={answer_end}")
            return extracted
        else:
            if DEBUG_LOG_RESPONSES:
                from tqdm import tqdm
                preview = gpt_response if len(gpt_response) <= 400 else gpt_response[:400] + "..."
                tqdm.write(f"[PARSE][Answer] Tags not found, returning raw. preview={preview}")
            return gpt_response
    except:
        if DEBUG_LOG_RESPONSES:
            from tqdm import tqdm
            tqdm.write("[PARSE][Answer] Exception, returning raw string")
        return gpt_response

def get_expected_diseases_from_proposal(proposal: Dict, item: Dict) -> List[str]:
    """Extract expected diseases from proposal based on source_categories"""
    expected_diseases = []
    if "source_categories" in proposal:
        expected_diseases = list(set(proposal["source_categories"]))
    return expected_diseases

def check_any_disease_in_response(diseases: List[str], response: str) -> bool:
    response_lower = response.lower()
    for disease in diseases:
        if disease.lower() in response_lower:
            return True
    return False

def compute_iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = areaA + areaB - inter_area
    return (inter_area / denom) if denom > 0 else 0.0

def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_checkpoint(results: List[Dict], checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def get_base_filename(image_name: str) -> str:
    return os.path.splitext(os.path.basename(image_name))[0]

def normalize_image_name(image_name: str) -> str:
    return os.path.basename(image_name)

def generate_first_round_variant() -> str:
    """Generate variant responses for first round based on requirement2"""
    variants = [
        "The panoramic radiograph requires closer examination. Some potential abnormalities cannot be directly identified at this resolution and need zoom-in for detailed analysis.",
        "Initial observation shows the overall dental structure, but subtle pathological changes are not clearly visible. Further magnification is needed to examine specific regions of interest.",
        "At this viewing scale, it's difficult to definitively identify all diseases or conditions. Zooming into specific areas will allow for more accurate diagnostic assessment.",
        "The full panoramic view provides an overview, but fine-grained abnormalities require higher magnification. Let me examine specific regions more closely.",
        "Based on the panoramic view, there may be areas of concern that need detailed inspection. I need to zoom in to specific regions for proper evaluation."
    ]
    import random
    return random.choice(variants)

def convert_to_multiround_format(result: Dict) -> Dict:
    """
    Convert to multi-round conversation format with the following rules:
    - First round uses fixed answer indicating need for zoom-in (requirement2)
    - Image names must match actual files (original basename for round 1; later rounds use basename of round_data['used_image'])
    - When the NEXT round uses comparative image, include 'comparative' flag in the tool_call arguments
    - For rounds that used GPT-4, move its analysis to sibling field "gpt-4-analysis",
      and replace the <think> content with GT text (prefer proposal.cur_proposal_gt)
    """
    def _is_comparative_round(r: Dict) -> bool:
        used = r.get("used_image", "") or ""
        if used.endswith("_proposal_comparative.jpg"):
            return True
        typ = (r.get("type") or "").lower()
        return "comparative" in typ

    messages: List[Dict[str, Any]] = []
    images: List[str] = []
    bboxes_used: List[List[int]] = []

    system_content = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"zoom-in","description":"Zoom into a specific region of the image using a bounding box.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"[x1,y1,x2,y2]"}},"required":["bbox_2d"]},"args_format":"JSON"}}
</tools>

Return tool calls within <tool_call></tool_call> tags:
<tool_call>
{"name":"zoom-in","arguments":{"bbox_2d":[x1,y1,x2,y2]}}
</tool_call>"""
    messages.append({"role": "system", "content": system_content})

    original_image_file = os.path.basename(result["image_name"])
    images.append(original_image_file)

    user_question = (
        f'<image> Question: {result["question"]}\n'
        'Answer with a detailed analysis of what you observe in the image.\n'
        'Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. '
        'Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) '
        'OR <answer>...</answer> (if no tools needed).'
    )
    messages.append({"role": "user", "content": user_question})

    proposal_rounds = [r for r in result.get("conversation_rounds", []) if r.get("round", 1) > 1]

    first_round_answer = generate_first_round_variant()

    if len(proposal_rounds) == 0:
        # No proposals, just return first round answer
        assistant_content = f"<answer> {first_round_answer} </answer>"
        messages.append({"role": "assistant", "content": assistant_content})
        metadata = {
            "original_index": result.get("index", 0),
            "num_bboxes": 0,
            "bboxes_used": [],
            "original_image": original_image_file
        }
        return {"messages": messages, "images": images, "metadata": metadata}

    # Prepare first tool call to the first proposal
    first_next_bbox = proposal_rounds[0].get("proposal", {}).get("bbox")
    next_is_comparative = _is_comparative_round(proposal_rounds[0])
    if first_next_bbox is not None:
        bboxes_used.append(first_next_bbox)
        args_str = (
            f'{{"bbox_2d": {first_next_bbox}, comparative}}' if next_is_comparative
            else f'{{"bbox_2d": {first_next_bbox}}}'
        )
        assistant_content = (
            f"<think> {first_round_answer} </think>\n"
            f"<tool_call>\n{{\"name\": \"zoom-in\", \"arguments\": {args_str}}}\n</tool_call>"
        )
    else:
        assistant_content = f"<think> {first_round_answer} </think>"
    messages.append({"role": "assistant", "content": assistant_content})

    # Iterate each proposal round
    for i, rd in enumerate(proposal_rounds):
        used_image_path = rd.get("used_image", "")
        used_image_name = os.path.basename(used_image_path) if used_image_path else ""
        if used_image_name:
            images.append(used_image_name)
        messages.append({"role": "user", "content": "<image>"})

        # Read GT text (prefer cur_proposal_gt), use as <think> content
        # Move GPT-4 analysis to "gpt-4-analysis"
        gt_for_message = (rd.get("gt_for_message") or "").strip()
        gpt4_analysis = (rd.get("gpt4_analysis") or rd.get("answer") or "").strip()

        is_last = (i == len(proposal_rounds) - 1)
        if not is_last:
            nxt = proposal_rounds[i + 1]
            nxt_bbox = nxt.get("proposal", {}).get("bbox")
            nxt_is_comp = _is_comparative_round(nxt)
            if nxt_bbox is not None:
                bboxes_used.append(nxt_bbox)
                args_str = (
                    f'{{"bbox_2d": {nxt_bbox}, comparative}}' if nxt_is_comp
                    else f'{{"bbox_2d": {nxt_bbox}}}'
                )
                assistant_content = (
                    f"<think> {gt_for_message} </think>\n"
                    f"<tool_call>\n{{\"name\": \"zoom-in\", \"arguments\": {args_str}}}\n</tool_call>"
                )
                messages.append({"role": "assistant", "content": assistant_content, "gpt-4-analysis": gpt4_analysis})
            else:
                assistant_content = f"<think> {gt_for_message} </think>"
                messages.append({"role": "assistant", "content": assistant_content, "gpt-4-analysis": gpt4_analysis})
        else:
            # Last round - provide final answer using full Answer from JSON
            full_answer = result.get("answer", "").strip()
            assistant_content = f"<think> {gt_for_message} </think>\n<answer> {full_answer} </answer>"
            messages.append({"role": "assistant", "content": assistant_content, "gpt-4-analysis": gpt4_analysis})

    metadata = {
        "original_index": result.get("index", 0),
        "num_bboxes": len(bboxes_used),
        "bboxes_used": bboxes_used,
        "original_image": original_image_file
    }
    return {"messages": messages, "images": images, "metadata": metadata}

def process_single_image(item: Dict, proposals_data: Dict, data_path: str, output_dir: str, image_idx: int) -> Dict:
    """Process single image with new multi-round conversation logic."""
    image_name = item["image_name"]
    image_path = os.path.join(data_path, "images", os.path.basename(image_name))
    base_filename = get_base_filename(image_name)

    result = {
        "image_name": image_name,
        "image_path": image_path,
        "question": item["Question"],
        "answer": item.get("Answer", ""),
        "conversation_rounds": [],
        "final_result": "failed"
    }

    if not os.path.exists(image_path):
        result["error"] = f"Image not found: {image_path}"
        return result

    first_round_answer = generate_first_round_variant()

    result["conversation_rounds"].append({
        "round": 1,
        "type": "initial_assessment",
        "question": item["Question"],
        "answer": first_round_answer,
        "source": "Fixed response indicating need for zoom-in"
    })

    # proposals lookup
    normalized_item_name = normalize_image_name(image_name)
    proposals = []
    for image_data in proposals_data["images"]:
        if normalize_image_name(image_data["image_name"]) == normalized_item_name:
            proposals = image_data.get("proposals", [])
            break

    if not proposals:
        print(f"  No proposals found for {normalized_item_name}")
        result["final_result"] = "no_proposals_found"
        return result

    other_proposals = [p for p in proposals if p.get("type") == "other"]
    all_proposals = other_proposals

    if not all_proposals:
        print(f"  No 'other' type proposals found for {normalized_item_name}")
        result["final_result"] = "no_other_proposals_found"
        return result

    img = cv2.imread(image_path)
    if img is None:
        result["final_result"] = "error"
        result["error"] = f"Failed to read image for proposal processing: {image_path}"
        return result
    img_h, img_w = img.shape[:2]

    for proposal_idx, proposal in enumerate(all_proposals):
        round_number = proposal_idx + 2
        bbox = tuple(proposal["bbox"])
        crop_path = os.path.join(output_dir, "crops", f"{base_filename}_round{round_number}_proposal.jpg")
        crop_image_with_bbox(image_path, bbox, crop_path)

        expected_diseases = get_expected_diseases_from_proposal(proposal, item)

        cur_gt = (proposal.get("cur_proposal_gt") or "").strip()
        gt_description = cur_gt if cur_gt else ""

        question = f"Let's examine the region at bbox {bbox} for potential abnormalities."

        crop_resp = get_gpt_vision_response(crop_path, "cropped_region_analysis")
        crop_answer = ""
        if crop_resp["success"]:
            crop_answer = extract_answer_from_response(crop_resp["gpt_answer"])

        pass_direct = False
        if crop_answer and expected_diseases:
            pass_direct = check_any_disease_in_response(expected_diseases, crop_answer)

        if pass_direct:
            result["conversation_rounds"].append({
                "round": round_number,
                "type": "successful_identification",
                "proposal": proposal,
                "question": question,
                "answer": crop_answer,
                "gpt4_analysis": crop_answer,
                "gt_for_message": gt_description,
                "ground_truth": gt_description,
                "expected_diseases": expected_diseases,
                "analysis_method": "cropped_region_analysis",
                "bbox": bbox,
                "crop_path": crop_path,
                "used_image": crop_path
            })
            continue

        mirror_bbox = create_mirror_bbox(bbox, img_w)
        iou_val = compute_iou(bbox, mirror_bbox)
        mirror_skip = iou_val > 0.4

        if mirror_skip:
            guide_resp = get_llm_text_response("ground_truth_guidance", gt_description)
            guided_answer = extract_answer_from_response(guide_resp["response"]) if guide_resp["success"] else gt_description
            result["conversation_rounds"].append({
                "round": round_number,
                "type": "ground_truth_guided_mirror_skipped",
                "proposal": proposal,
                "question": question,
                "answer": guided_answer,
                "gt_for_message": gt_description,
                "ground_truth": gt_description,
                "expected_diseases": expected_diseases,
                "analysis_method": "ground_truth_guidance",
                "bbox": bbox,
                "crop_path": crop_path,
                "used_image": crop_path,
                "mirror_bbox": mirror_bbox,
                "mirror_iou": iou_val,
                "mirror_skipped_reason": "IoU>0.4"
            })
            continue

        mirror_crop_path = os.path.join(output_dir, "crops", f"{base_filename}_round{round_number}_mirror.jpg")
        crop_image_with_bbox(image_path, mirror_bbox, mirror_crop_path)

        comparative_path = os.path.join(output_dir, "crops", f"{base_filename}_round{round_number}_proposal_comparative.jpg")
        create_comparative_image(crop_path, mirror_crop_path, comparative_path, gap_pixels=10)

        comp_resp = get_gpt_vision_response(comparative_path, "comparative_analysis")
        comp_answer = extract_answer_from_response(comp_resp["gpt_answer"]) if comp_resp["success"] else ""

        guide_resp = get_llm_text_response("ground_truth_guidance", gt_description)
        guided_answer = extract_answer_from_response(guide_resp["response"]) if guide_resp["success"] else gt_description

        result["conversation_rounds"].append({
            "round": round_number,
            "type": "comparative_with_gt_supplement",
            "proposal": proposal,
            "question": question,
            "answer": comp_answer if comp_answer else guided_answer,
            "gpt4_analysis": comp_answer if comp_answer else "",
            "gt_for_message": gt_description,
            "supplement_from_ground_truth": True,
            "guided_answer": guided_answer,
            "ground_truth": gt_description,
            "expected_diseases": expected_diseases,
            "analysis_method": "comparative_with_gt_guidance",
            "bbox": bbox,
            "crop_path": crop_path,
            "mirror_bbox": mirror_bbox,
            "mirror_iou": iou_val,
            "mirror_crop_path": mirror_crop_path,
            "comparative_path": comparative_path,
            "used_image": comparative_path
        })

    result["final_result"] = "completed_multi_round"
    return result

def process_dental_dataset(json_path: str, proposals_path: str, data_path: str, output_path: str = None, start_idx: int = 0, end_idx: int = None, shard_id: int = None) -> None:
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    proposals_data = load_proposals_data(proposals_path)
    if end_idx is None:
        end_idx = len(data_list)
    data_list = data_list[start_idx:end_idx]
    if output_path is None:
        output_path = '/hpc2hdd/home/yfan546/workplace/xray_teeth/r1/mmoral_reasoning_data/multi_round_output'

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "crops"), exist_ok=True)

    if shard_id is not None:
        checkpoint_file = os.path.join(output_path, f"checkpoint_shard_{shard_id}.json")
        multiround_file = os.path.join(output_path, f"multi_round_data_shard_{shard_id}.json")
    else:
        checkpoint_file = os.path.join(output_path, "multi_round_conversation_results.json")
        multiround_file = os.path.join(output_path, "multi_round_data.json")

    existing_results = load_checkpoint(checkpoint_file)
    processed_indices = {result.get("index", -1) for result in existing_results}

    shard_info = f" (Shard {shard_id})" if shard_id is not None else ""
    print(f"Starting to process {len(data_list)} samples{shard_info}...")
    print(f"Already processed samples: {len(existing_results)}")

    results = existing_results.copy()
    multiround_data = []
    if os.path.exists(multiround_file):
        try:
            with open(multiround_file, 'r', encoding='utf-8') as f:
                multiround_data = json.load(f)
        except:
            multiround_data = []

    for idx, item in enumerate(tqdm(data_list, desc="Processing images")):
        current_idx = start_idx + idx
        if current_idx in processed_indices:
            print(f"Skipping already processed sample {current_idx}")
            continue
        try:
            print(f"\nProcessing sample {idx+1}/{len(data_list)}: {item['image_name']}")
            result = process_single_image(item, proposals_data, data_path, output_path, current_idx)
            result["index"] = current_idx
            results.append(result)

            multiround_entry = convert_to_multiround_format(result)
            if multiround_entry:
                multiround_data.append(multiround_entry)

            if len(results) % 20 == 0:
                print(f"Saving checkpoint, currently processed {len(results)} samples")
                save_checkpoint(results, checkpoint_file)
                with open(multiround_file, 'w', encoding='utf-8') as f:
                    json.dump(multiround_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            error_result = {
                "index": current_idx,
                "image_name": item.get("image_name", "unknown"),
                "error": str(e),
                "final_result": "error"
            }
            results.append(error_result)
            continue

    save_checkpoint(results, checkpoint_file)
    with open(multiround_file, 'w', encoding='utf-8') as f:
        json.dump(multiround_data, f, ensure_ascii=False, indent=2)
    print_final_statistics(results)

def print_final_statistics(results: List[Dict]) -> None:
    total = len(results)
    completed_multi = sum(1 for r in results if r.get("final_result") == "completed_multi_round")
    no_proposals = sum(1 for r in results if r.get("final_result") == "no_proposals_found")
    no_other_proposals = sum(1 for r in results if r.get("final_result") == "no_other_proposals_found")
    failed = sum(1 for r in results if r.get("final_result") == "failed")
    errors = sum(1 for r in results if r.get("final_result") == "error")

    print("\n" + "="*60)
    print("Multi-round Conversation Results Statistics")
    print("="*60)
    print(f"Total processed samples: {total}")
    print(f"Completed with multi-round: {completed_multi} ({completed_multi/total*100:.1f}%)")
    print(f"No proposals found: {no_proposals} ({no_proposals/total*100:.1f}%)")
    print(f"No 'other' type proposals found: {no_other_proposals} ({no_other_proposals/total*100:.1f}%)")
    print(f"Processing failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Errors occurred: {errors} ({errors/total*100:.1f}%)")

    total_rounds = 0
    successful_identifications = 0
    comparative_successes = 0
    ground_truth_guided = 0

    for result in results:
        if "conversation_rounds" in result:
            total_rounds += len(result["conversation_rounds"])
            for round_data in result["conversation_rounds"]:
                if round_data.get("type") == "successful_identification":
                    successful_identifications += 1
                elif round_data.get("type") == "comparative_analysis_success":
                    comparative_successes += 1
                elif "guided" in round_data.get("type", ""):
                    ground_truth_guided += 1

    print(f"\nConversation Analysis:")
    print(f"Total conversation rounds: {total_rounds}")
    print(f"Direct successful identifications: {successful_identifications}")
    print(f"Successful with comparative analysis: {comparative_successes}")
    print(f"Required ground truth guidance: {ground_truth_guided}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Multi-round conversation data generation for dental X-ray analysis')
    parser.add_argument('--json_path', type=str,
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/1_380/Multimodal_data_Tufts_Dental_Database.json",
                        help='JSON file path')
    parser.add_argument('--proposals_path', type=str,
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/1_380/proposals_data1.json",
                        help='Proposals JSON file path')
    parser.add_argument('--data_path', type=str,
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/1_380/",
                        help='Data root directory path')
    parser.add_argument('--output_path', type=str,
                        default='/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/multi_round_output_data1',
                        help='Output directory path')
    parser.add_argument('--start_idx', type=int, default=0, help='Start processing index')
    parser.add_argument('--end_idx', type=int, default=None, help='End processing index')
    parser.add_argument('--shard_id', type=int, default=0, help='Shard ID for parallel processing')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards for automatic splitting')
    args = parser.parse_args()

    if args.num_shards is not None and args.shard_id is not None:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            total_samples = len(json.load(f))
        
        shard_size = (total_samples + args.num_shards - 1) // args.num_shards
        calculated_start = args.shard_id * shard_size
        calculated_end = min((args.shard_id + 1) * shard_size, total_samples)
        
        if args.start_idx == 0 and args.end_idx is None:
            args.start_idx = calculated_start
            args.end_idx = calculated_end
            print(f"Auto-calculated shard {args.shard_id}/{args.num_shards}: processing samples {args.start_idx} to {args.end_idx}")

    process_dental_dataset(
        json_path=args.json_path,
        proposals_path=args.proposals_path,
        data_path=args.data_path,
        output_path=args.output_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        shard_id=args.shard_id
    )

if __name__ == "__main__":
    main()