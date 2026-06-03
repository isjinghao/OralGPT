import requests
import json
import os
import time
import base64
from typing import Dict, Any, List, Tuple
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import re
import numpy as np
from PIL import Image

# API configuration
url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ.get('API_KEY', '')}"
}

# Disease mapping
segment_names_to_labels = [
    ("Implant", 0), ("Prosthetic restoration", 1), ("Obturation", 2), ("Endodontic treatment", 3),
    ("Orthodontic device", 12), ("Surgical device", 13),  # Low risk
    ("Carious lesion", 4), ("Bone resorbtion", 5), ("Impacted tooth", 6), ("Apical surgery", 10),  # Medium risk
    ("Apical periodontitis", 7), ("Root fragment", 8), ("Furcation lesion", 9), ("Root resorption", 11)  # High risk
]

label_to_name = {label: name for name, label in segment_names_to_labels}
name_to_label = {name: label for name, label in segment_names_to_labels}

def get_clinical_diagnosis_prompt(question_type: str, context: str = "") -> str:
    """Generate clinical diagnosis oriented prompts"""

    if question_type == "full_image_analysis":
        return """You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions.

Please examine this radiographic image systematically and identify any present disease(s) or condition(s). You MUST strictly follow this output format, including all opening and closing tags exactly as written:
            "<Think> ... </Think>\n"
            "<Answer> ... </Answer>\n\n"

Instructions for each section:

<Think>
1. Output the preliminary and potential diseases/conditions visible on the panoramic radiograph, with approximate locations. Example: "Implant detected at tooth #17," "Impacted tooth at #28," "Carious lesion at tooth #36," "Mild bone resorption in the lower jaw," etc.

2. Explain the visible radiographic features that demonstrate the characteristics of the identified disease(s)/conditions, such as: implant, prosthetic restoration, obturation, endodontic treatment, orthodontic appliance, surgical device, carious lesion, bone resorption, impacted tooth, apical surgery, apical periodontitis, root fragment, furcation lesion, root resorption, or any other relevant disease(s)/condition(s).

3. Recall the typical radiographic appearance and common symptoms associated with the preliminarily predicted conditions, based on professional knowledge.

4. Confirm whether the observed radiographic findings are consistent with the knowledge-based expected diagnostic features.
</Think>

<Answer>
Summarize the confirmed disease(s)/condition(s) from step 4 into one concise, fluent, and professional diagnostic statement, clearly specifying the findings in the image, without including any treatment recommendations or management advice.
</Answer>"""

    elif question_type == "cropped_region_analysis":
        return """You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions.
For this task, you are not viewing the entire panoramic radiograph. Instead, you are provided with a cropped subregion extracted from the full panoramic image. This subregion allows closer observation of potential areas where a disease or condition may be present.

Please examine this cropped radiographic image systematically and identify any present disease(s) or condition(s). You MUST strictly follow this output format, including all opening and closing tags exactly as written:
            "<Think> ... </Think>\n"
            "<Answer> ... </Answer>\n\n"

Instructions for each section:

<Think>
1. List the preliminary and potential diseases/conditions visible within this cropped panoramic radiograph.

2. Explain the visible radiographic features that demonstrate the characteristics of the identified disease(s)/conditions, such as: implant, prosthetic restoration, obturation, endodontic treatment, orthodontic appliance, surgical device, carious lesion, bone resorption, impacted tooth, apical surgery, apical periodontitis, root fragment, furcation lesion, root resorption, or any other relevant disease(s)/condition(s).

3. Recall the typical radiographic appearance and common symptoms associated with the preliminarily predicted conditions, based on professional knowledge.

4. Confirm whether the observed radiographic findings are consistent with the knowledge-based expected diagnostic features.
</Think>

<Answer>
Summarize the confirmed disease(s)/condition(s) from step 4 into one concise, fluent, and professional diagnostic statement, clearly specifying the findings in the image, without including any treatment recommendations or management advice.
</Answer>"""

    elif question_type == "comparative_analysis":
        return """You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions.

This is a cropped region of interest (RoI) from a panoramic x-ray. The left side of the image shows the original RoI, which may contain potential abnormalities.
The right side of the image shows the mirrored symmetrical region of the oral cavity. The mirrored symmetrical region is provided mainly for visual reference. 

By comparing the two sides, please identify and describe any relatively subtle abnormalities that may be present in the original RoI (left side of the image). This comparative approach is inspired by clinical dental practice, where dentists often examine analogous teeth in opposite quadrants due to the natural symmetry of the dentition.

Please examine this cropped radiographic image systematically and identify any present disease(s) or condition(s). You MUST strictly follow this output format, including all opening and closing tags exactly as written:
            "<Think> ... </Think>\n"
            "<Answer> ... </Answer>\n\n"

Instructions for each section:

<Think>
1. List the preliminary and potential diseases/conditions visible within this cropped panoramic radiograph.

2. By comparing the visual features between the left (original RoI) and the right (mirrored reference) cropped regions, explain the visible radiographic features that demonstrate the characteristics of the identified disease(s)/conditions, such as: implant, prosthetic restoration, obturation, endodontic treatment, orthodontic appliance, surgical device, carious lesion, bone resorption, impacted tooth, apical surgery, apical periodontitis, root fragment, furcation lesion, root resorption, or any other relevant disease(s)/condition(s).

3. Recall the typical radiographic appearance and common symptoms associated with the preliminarily predicted conditions, based on professional knowledge.

4. Confirm whether the observed radiographic findings are consistent with the knowledge-based expected diagnostic features.
</Think>

<Answer>
Summarize the confirmed disease(s)/condition(s) from step 4 into one concise, fluent, and professional diagnostic statement, clearly specifying the findings in the image, without including any treatment recommendations or management advice.
</Answer>"""

    elif question_type == "immediate_output_verification":
        return f"""You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions.

Your task: Evaluate whether a provided description of disease(s)/condition(s) in a panoramic radiograph matches the actual ground truth (gold standard) description of that radiograph.

- Extract and compare the disease(s)/condition(s) listed in the given description against those in the gold standard description.
- Identify which disease(s)/condition(s) are matched (correctly described) and which are unmatched (missing, incorrect, or extra).

Given description: {context.split('GROUND_TRUTH:')[0].strip()}
Ground truth description: {context.split('GROUND_TRUTH:')[1].strip() if 'GROUND_TRUTH:' in context else ''}

Output format (strict):

<Matched>
[List all matched disease(s)/condition(s) here]
</Matched>

<Unmatched>
[List all unmatched disease(s)/condition(s) here, indicating whether they are missing from the given description or erroneously included when compared to the gold standard]
</Unmatched>"""

    elif question_type == "ground_truth_guidance":
        return f"""You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions.

You are provided with the ground truth answer for a specific region in a panoramic radiograph. Based on this ground truth, please generate a professional diagnostic description that explains what can be observed in the image.

Ground truth answer: {context}

Please generate a comprehensive description that includes:
1. The specific radiographic findings visible in this region
2. The characteristic features that support the diagnosis
3. The radiographic appearance and patterns observed

You MUST strictly follow this output format:
<Answer>
[Your comprehensive diagnostic description based on the ground truth, explaining the radiographic findings in a professional manner]
</Answer>"""

    elif question_type == "educational_description":
        condition = context
        return f"""You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions.

Provide educational documentation about radiographic appearances for {condition}. Include:
- Radiographic appearance and density patterns
- Typical anatomical locations and spatial relationships
- Key visual features for identification in clinical contexts
- Distinguishing radiographic characteristics

Frame this as educational material for clinical radiography training."""
    else:
        return context

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_gpt_vision_response(image_path: str, question_type: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
    try:
        base64_image = encode_image_to_base64(image_path)
        clinical_prompt = get_clinical_diagnosis_prompt(question_type, context)

        messages = [
            {
                "role": "system",
                "content": "You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions. Provide precise clinical diagnostic analysis."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": clinical_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

        data = {"model": "gpt-4", "messages": messages, "temperature": 0.1, "max_tokens": 1000}

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
                response.raise_for_status()
                response_json = response.json()
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    gpt_answer = response_json["choices"][0]["message"]["content"]
                    return {"success": True, "gpt_answer": gpt_answer, "error": None, "attempts": attempt + 1}
                else:
                    return {"success": False, "gpt_answer": None, "error": "No response from GPT", "attempts": attempt + 1}
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {"success": False, "gpt_answer": None, "error": f"Request failed after {max_retries} attempts: {str(e)}", "attempts": max_retries}
                time.sleep(2 ** attempt)
            except Exception as e:
                return {"success": False, "gpt_answer": None, "error": f"Unexpected error: {str(e)}", "attempts": attempt + 1}
    except Exception as e:
        return {"success": False, "gpt_answer": None, "error": f"Image processing error: {str(e)}", "attempts": 0}

def get_llm_text_response(prompt_type: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
    clinical_prompt = get_clinical_diagnosis_prompt(prompt_type, context)
    messages = [
        {"role": "system", "content": "You are an oral and maxillofacial radiology expert, specialized in analyzing panoramic radiographs to identify various diseases and conditions."},
        {"role": "user", "content": clinical_prompt}
    ]
    data = {"model": "gpt-4", "messages": messages, "temperature": 0.1, "max_tokens": 1000}
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            response.raise_for_status()
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return {"success": True, "response": response_json["choices"][0]["message"]["content"], "error": None, "attempts": attempt + 1}
            else:
                return {"success": False, "response": None, "error": "No response from GPT", "attempts": attempt + 1}
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {"success": False, "response": None, "error": f"Request failed after {max_retries} attempts: {str(e)}", "attempts": max_retries}
            time.sleep(2 ** attempt)
        except Exception as e:
            return {"success": False, "response": None, "error": f"Unexpected error: {str(e)}", "attempts": attempt + 1}

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
    """Create comparative image - left: original ROI; right: mirrored (h-flipped) contralateral ROI."""
    img1 = cv2.imread(contextual_crop_path)  # contextual/original
    img2 = cv2.imread(comparative_crop_path)  # contralateral crop
    if img1 is None or img2 is None:
        raise ValueError("Cannot read one or both images for comparison")

    img2 = cv2.flip(img2, 1)

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

def find_proposals_for_image(proposals_data: Dict, image_name: str) -> List[Dict]:
    for image_data in proposals_data["images"]:
        if image_data["image_name"] == image_name:
            return image_data["proposals"]
    return []

def get_description_by_lesion_indices(full_answer: str, lesion_indices: List[int]) -> str:
    parts = [p.strip() for p in re.split(r'\.\s*', full_answer) if p.strip()]
    picked = []
    for idx in lesion_indices:
        if 0 <= idx < len(parts):
            picked.append(parts[idx])
    return ('. '.join(picked) + '.') if picked else ''

def extract_answer_from_response(gpt_response: str) -> str:
    try:
        answer_start = gpt_response.find('<Answer>')
        answer_end = gpt_response.find('</Answer>')
        if answer_start != -1 and answer_end != -1:
            return gpt_response[answer_start + 8:answer_end].strip()
        else:
            return gpt_response
    except:
        return gpt_response

def check_disease_in_response(disease: str, response: str) -> bool:
    response_lower = response.lower()
    disease_lower = disease.lower()
    return disease_lower in response_lower

def get_expected_diseases_from_proposal(proposal: Dict, item: Dict) -> List[str]:
    expected_diseases = []
    if "lesion_indices" in proposal and "Category" in item:
        categories = item["Category"]
        lesion_indices = proposal["lesion_indices"]
        for idx in lesion_indices:
            if idx < len(categories):
                disease_name = categories[idx]
                if disease_name not in expected_diseases:
                    expected_diseases.append(disease_name)
    if not expected_diseases and "source_categories" in proposal:
        expected_diseases = list(set(proposal["source_categories"]))
    return expected_diseases

def check_any_disease_in_response(diseases: List[str], response: str) -> bool:
    response_lower = response.lower()
    for disease in diseases:
        if disease.lower() in response_lower:
            return True
    return False

def parse_matched_unmatched(verification_response: str) -> Tuple[List[str], List[str]]:
    matched_features = []
    unmatched_features = []
    try:
        matched_start = verification_response.find('<Matched>')
        matched_end = verification_response.find('</Matched>')
        unmatched_start = verification_response.find('<Unmatched>')
        unmatched_end = verification_response.find('</Unmatched>')
        if matched_start != -1 and matched_end != -1:
            matched_content = verification_response[matched_start + 9:matched_end].strip()
            if matched_content and matched_content.lower() != '[list all matched disease(s)/condition(s) here]':
                for feature_name, _ in segment_names_to_labels:
                    if feature_name.lower() in matched_content.lower():
                        matched_features.append(feature_name)
        if unmatched_start != -1 and unmatched_end != -1:
            unmatched_content = verification_response[unmatched_start + 11:unmatched_end].strip()
            if unmatched_content and unmatched_content.lower() != '[list all unmatched disease(s)/condition(s) here, indicating whether they are missing from the given description or erroneously included when compared to the gold standard]':
                for feature_name, _ in segment_names_to_labels:
                    if feature_name.lower() in unmatched_content.lower():
                        unmatched_features.append(feature_name)
    except Exception as e:
        print(f"Error parsing verification response: {e}")
    return matched_features, unmatched_features

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

def convert_to_multiround_format(result: Dict) -> Dict:
    """
    Convert to multi-round conversation format with the following rules:
    - If second_round_answer is empty -> first round ONLY <answer></answer>, no <think>, no tool_call.
    - Image names must match actual files (original basename for round 1; later rounds use basename of round_data['used_image']).
    - When the NEXT round uses comparative image, include 'comparative' flag in the tool_call arguments:
      {"bbox_2d": [bbox], comparative}
    - For rounds that used GPT-4, move its analysis to sibling field "gpt-4-analysis",
      and replace the <think> content in assistant message with GT text (prefer proposal.cur_proposal_gt).
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
    first_round_answer = (result.get("first_round_answer") or "").strip()
    second_round_answer = (result.get("second_round_answer") or "").strip()
    sr_empty = (second_round_answer == "")

    if sr_empty or len(proposal_rounds) == 0:
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
            f"<think> {first_round_answer} There are some potential fine-grained diseases/conditions that I need to zoom in for further examination. "
            f"Areas that may contain potential diseases/conditions include: multiple proposals【zoom-in tool + parameters】 </think>\n"
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
            fr = (result.get("first_round_answer") or "").strip()
            sr = (result.get("second_round_answer") or "").strip()
            full_answer = (fr + (" " + sr if sr else "")).strip()
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
        "first_round_answer": item.get("First Round Answer", ""),
        "second_round_answer": item.get("Second Round Answer", ""),
        "conversation_rounds": [],
        "final_result": "failed"
    }

    if not os.path.exists(image_path):
        result["error"] = f"Image not found: {image_path}"
        return result

    # Round 1 (rule-based)
    # print(f"Round 1: Rule-based analysis for {os.path.basename(image_name)}")
    first_round_answer = result["first_round_answer"]
    has_more = bool(item.get("Full Answer", "").strip())
    if has_more:
        first_round_response = (first_round_answer.strip() +
            (" There are some potential fine-grained diseases/conditions that I need to zoom in for further examination. "
             "Areas that may contain potential diseases/conditions include: multiple proposals【zoom-in tool + parameters】"
             if first_round_answer.strip() else ""))
    else:
        first_round_response = first_round_answer

    result["conversation_rounds"].append({
        "round": 1,
        "type": "rule_based",
        "question": item["Question"],
        "answer": first_round_response,
        "source": "First Round Answer from new_json"
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

    # Priority
    bone_resorbtion_proposals = [p for p in proposals if p.get("type") == "bone_resorbtion"]
    other_proposals = [p for p in proposals if p.get("type") != "bone_resorbtion"]
    all_proposals = bone_resorbtion_proposals + other_proposals

    full_answer_text = item.get("Full Answer", "")
    categories_list = item.get("Category", [])

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
        if cur_gt:
            gt_description = cur_gt
        else:
            gt_description = get_description_by_lesion_indices(full_answer_text, proposal.get("lesion_indices", []))

        question = f"Let's first look at the <proposal bbox {bbox}> position and call GPT to see what diseases can be identified [cropped_region_analysis]"

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

def process_dental_dataset(json_path: str, proposals_path: str, data_path: str, output_path: str = None, start_idx: int = 0, end_idx: int = None) -> None:
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

    checkpoint_file = os.path.join(output_path, "multi_round_conversation_results.json")
    multiround_file = os.path.join(output_path, "multi_round_data.json")

    existing_results = load_checkpoint(checkpoint_file)
    processed_indices = {result.get("index", -1) for result in existing_results}

    print(f"Starting to process {len(data_list)} samples...")
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
    # print(f"\nMulti-round data saved to: {multiround_file}")
    # print(f"Total multi-round entries: {len(multiround_data)}")

def print_final_statistics(results: List[Dict]) -> None:
    total = len(results)
    completed_first = sum(1 for r in results if r.get("final_result") == "completed_first_round")
    completed_multi = sum(1 for r in results if r.get("final_result") == "completed_multi_round")
    no_proposals = sum(1 for r in results if r.get("final_result") == "no_proposals_found")
    failed = sum(1 for r in results if r.get("final_result") == "failed")
    errors = sum(1 for r in results if r.get("final_result") == "error")

    print("\n" + "="*60)
    print("Multi-round Conversation Results Statistics")
    print("="*60)
    print(f"Total processed samples: {total}")
    print(f"Completed in first round: {completed_first} ({completed_first/total*100:.1f}%)")
    print(f"Completed with multi-round: {completed_multi} ({completed_multi/total*100:.1f}%)")
    print(f"No proposals found: {no_proposals} ({no_proposals/total*100:.1f}%)")
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
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/4_1600/new_Multimodal_data_Dental_Conditions_Detection_2025_Romania.json",
                        help='JSON file path')
    parser.add_argument('--proposals_path', type=str,
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/4_1600/proposals_data4.json",
                        help='Proposals JSON file path')
    parser.add_argument('--data_path', type=str,
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/4_1600/all",
                        help='Data root directory path')
    parser.add_argument('--output_path', type=str,
                        default='/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/mmoral_reasoning_data/multi_round_output',
                        help='Output directory path')
    parser.add_argument('--start_idx', type=int, default=0, help='Start processing index')
    parser.add_argument('--end_idx', type=int, default=None, help='End processing index')
    args = parser.parse_args()

    process_dental_dataset(
        json_path=args.json_path,
        proposals_path=args.proposals_path,
        data_path=args.data_path,
        output_path=args.output_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )

if __name__ == "__main__":
    main()