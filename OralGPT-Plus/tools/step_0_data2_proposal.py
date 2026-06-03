#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np

# --- Optional (for visualization) ---
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False

# -----------------------
# Utilities: geometry
# -----------------------
def split_full_answer_sentences(text: str) -> List[str]:
    """
    Split Full Answer into clauses by '.', stripping whitespace and dropping empties.
    """
    if not isinstance(text, str):
        return []
    parts = [p.strip() for p in re.split(r'\.\s*', text) if p.strip()]
    return parts

def build_cur_proposal_gt(full_answer: str, lesion_indices: List[int]) -> str:
    """
    Pick clauses from Full Answer by lesion_indices and join them with spaces.
    """
    sents = split_full_answer_sentences(full_answer)
    picked = []
    for i in sorted(set(lesion_indices)):
        if 0 <= i < len(sents):
            picked.append(sents[i])
    return ' '.join([s + '.' for s in picked])


def box_area(b: List[int]) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_center(b: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def box_union(boxes: List[List[int]]) -> List[int]:
    if not boxes:
        return [0, 0, 0, 0]
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    
    x1, y1 = int(min(xs1)), int(min(ys1))
    x2, y2 = int(max(xs2)), int(max(ys2))

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    return [x1, y1, x2, y2]

def iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0

# Intersection area & IoM>=thr check
def inter_area(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return iw * ih

def iom_ge(a: List[int], b: List[int], thr: float) -> bool:
    ia = inter_area(a, b)
    aa = box_area(a)
    ab = box_area(b)
    if aa <= 0 or ab <= 0:
        return False
    return (ia / aa >= thr) or (ia / ab >= thr)

def boxes_overlap_horizontal(a: List[int], b: List[int],
                             min_h_overlap_ratio: float = 0.0,
                             min_v_overlap_ratio: float = 0.0) -> bool:
    """
    Return True only when boxes overlap both horizontally and vertically.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)

    if iw <= 0 or ih <= 0:
        return False

    h_overlap_ratio = iw / min(aw, bw)
    v_overlap_ratio = ih / min(ah, bh)

    return (h_overlap_ratio >= min_h_overlap_ratio) and (v_overlap_ratio >= min_v_overlap_ratio)

# -----------------------
# Category matching
# -----------------------
def normalize_category(cat: str) -> str:
    c = cat.strip().lower()
    c = c.replace('resorption', 'resorbtion')
    return c

OTHER_SET = {
    'abnormal tooth development',
    'deep pits and fissures',
    'caries',
    'periapical periodontitis',
    'pulpitis'
}

COLOR_PRECISE = {
    'abnormal tooth development': (255, 165, 0),
    'deep pits and fissures': (0, 200, 0),
    'caries': (0, 128, 255),
    'periapical periodontitis': (255, 0, 255),
    'pulpitis': (128, 0, 128),
    'other': (255, 255, 255),
}
COLOR_PROPOSAL = {
    'other': (255, 255, 0),
}

def is_other_target(cat: str) -> bool:
    c = normalize_category(cat)
    return any(c.startswith(k) for k in OTHER_SET)

# -----------------------
# Helpers
# -----------------------
def extract_image_id_from_path(image_name: str) -> Optional[str]:
    basename = os.path.basename(image_name)
    if basename:
        return os.path.splitext(basename)[0]
    return None

# -----------------------
# Simple KMeans (numpy)
# -----------------------
def kmeans(points: np.ndarray, k: int, max_iter: int=50, seed: int=42) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 2
    n = points.shape[0]
    rng = np.random.default_rng(seed)
    centers = np.empty((k, 2), dtype=np.float64)
    idx = rng.integers(0, n)
    centers[0] = points[idx]
    dists = np.full(n, np.inf)
    for ci in range(1, k):
        dists = np.minimum(dists, np.sum((points - centers[ci-1])**2, axis=1))
        probs = dists / dists.sum() if dists.sum() > 0 else np.full(n, 1.0/n)
        idx = rng.choice(n, p=probs)
        centers[ci] = points[idx]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        d = ((points[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = points[mask].mean(axis=0)
            else:
                centers[j] = points[rng.integers(0, n)]
    return labels

def choose_k_for_other(n: int) -> int:
    if n <= 3: return n
    if n <= 7: return 3
    if n <= 12: return 4
    return 5

# -----------------------
# Mapping helpers
# -----------------------
def get_ctx_or_precise(i: int,
                       precise: List[List[int]],
                       contextual: List[List[int]]) -> Optional[List[int]]:
    """Prefer contextual; fallback to precise."""
    if i < len(contextual) and contextual[i]:
        return [int(x) for x in contextual[i]]
    if i < len(precise) and precise[i]:
        return [int(x) for x in precise[i]]
    return None

# -----------------------
# Proposal merging for 'other' type
# -----------------------
def merge_overlapping_other_proposals(proposals: List[Dict[str, Any]], 
                                      logger: Optional[logging.Logger] = None,
                                      min_h_overlap_ratio: float = 0.0,
                                      min_v_overlap_ratio: float = 0.0) -> List[Dict[str, Any]]:
    other_proposals = [p for p in proposals if p.get('type') == 'other']
    non_other_proposals = [p for p in proposals if p.get('type') != 'other']
    
    if len(other_proposals) <= 1:
        return proposals
    
    n = len(other_proposals)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i in range(n):
        for j in range(i + 1, n):
            bbox_i = other_proposals[i]['bbox']
            bbox_j = other_proposals[j]['bbox']
            if boxes_overlap_horizontal(bbox_i, bbox_j,
                                        min_h_overlap_ratio=min_h_overlap_ratio,
                                        min_v_overlap_ratio=min_v_overlap_ratio):
                union(i, j)
                if logger:
                    logger.debug(f"Merging other proposals {i} and {j} due to horizontal+vertical overlap")
    
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    
    merged_other_proposals = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            merged_other_proposals.append(other_proposals[group_indices[0]])
        else:
            proposals_to_merge = [other_proposals[i] for i in group_indices]
            merged_proposal = merge_proposals(proposals_to_merge)
            merged_other_proposals.append(merged_proposal)
            if logger:
                logger.info(f"Merged {len(group_indices)} other proposals into one (H+V overlap)")
    
    return non_other_proposals + merged_other_proposals

def merge_proposals(proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple 'other' proposals into one.
    """
    if not proposals:
        return {}
    if len(proposals) == 1:
        return proposals[0]

    all_bboxes = [p['bbox'] for p in proposals]
    merged_bbox = box_union(all_bboxes)

    merged_source_categories = []
    merged_lesion_indices = []
    cluster_ids = []

    for p in proposals:
        merged_source_categories.extend(p.get('source_categories', []))
        merged_lesion_indices.extend(p.get('lesion_indices', []))
        if p.get('cluster_id') is not None:
            cluster_ids.append(p.get('cluster_id'))

    merged_lesion_indices = sorted(list(set(merged_lesion_indices)))

    if cluster_ids:
        new_cluster_id = min(cluster_ids) if all(isinstance(x, int) for x in cluster_ids) else "merged"
        if len(set(cluster_ids)) > 1:
            new_cluster_id = f"merged_{new_cluster_id}" if isinstance(new_cluster_id, int) else "merged"
    else:
        new_cluster_id = "merged"

    # aggregate cur_proposal_gt
    merged_gt_chunks = []
    for p in proposals:
        gt = (p.get('cur_proposal_gt') or '').strip()
        if gt:
            merged_gt_chunks.append(gt)
    seen = set()
    merged_gt_unique = []
    for t in merged_gt_chunks:
        if t not in seen:
            seen.add(t)
            merged_gt_unique.append(t)
    merged_gt_text = ' '.join(merged_gt_unique).strip()
    
    return {
        "type": "other",
        "cluster_id": new_cluster_id,
        "source_categories": merged_source_categories,
        "lesion_indices": merged_lesion_indices,
        "bbox": [int(x) for x in merged_bbox],
        "merged_from": len(proposals),
        "cur_proposal_gt": merged_gt_text,
    }

# Generic merger for IoM70
def merge_proposals_generic(proposals: List[Dict[str, Any]], ptype: str, note: str) -> Dict[str, Any]:
    if not proposals:
        return {}
    if len(proposals) == 1:
        out = dict(proposals[0])
        out["merge_note"] = note
        out["merged_from"] = 1
        return out

    all_bboxes = [p['bbox'] for p in proposals]
    merged_bbox = box_union(all_bboxes)

    merged_source_categories = []
    merged_lesion_indices = []
    cluster_ids = []

    for p in proposals:
        merged_source_categories.extend(p.get('source_categories', []))
        merged_lesion_indices.extend(p.get('lesion_indices', []))
        if p.get('cluster_id') is not None:
            cluster_ids.append(p.get('cluster_id'))

    merged_lesion_indices = sorted(list(set(merged_lesion_indices)))

    if cluster_ids:
        new_cluster_id = min([c for c in cluster_ids if isinstance(c, int)], default="merged")
        if len(set(map(str, cluster_ids))) > 1:
            new_cluster_id = f"merged_{new_cluster_id}" if isinstance(new_cluster_id, int) else "merged"
    else:
        new_cluster_id = "merged"

    # aggregate cur_proposal_gt
    merged_gt_chunks = []
    for p in proposals:
        gt = (p.get('cur_proposal_gt') or '').strip()
        if gt:
            merged_gt_chunks.append(gt)
    seen = set()
    merged_gt_unique = []
    for t in merged_gt_chunks:
        if t not in seen:
            seen.add(t)
            merged_gt_unique.append(t)
    merged_gt_text = ' '.join(merged_gt_unique).strip()

    return {
        "type": ptype,
        "cluster_id": new_cluster_id,
        "source_categories": merged_source_categories,
        "lesion_indices": merged_lesion_indices,
        "bbox": [int(x) for x in merged_bbox],
        "merged_from": len(proposals),
        "merge_note": note,
        "cur_proposal_gt": merged_gt_text,
    }

def merge_by_iom_per_type(proposals: List[Dict[str, Any]],
                          thr: float = 0.7,
                          logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Post-process: within each type, merge two proposals when their intersection
    covers >= 70% of either one.
    """
    out: List[Dict[str, Any]] = []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for p in proposals:
        by_type.setdefault(p.get('type', 'other'), []).append(p)

    for ptype, plist in by_type.items():
        if len(plist) <= 1:
            out.extend(plist)
            continue

        n = len(plist)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i+1, n):
                bi, bj = plist[i]['bbox'], plist[j]['bbox']
                if iom_ge(bi, bj, thr):
                    union(i, j)
                    if logger:
                        logger.debug(f"IoM{int(thr*100)} merge: type={ptype}, i={i}, j={j}")

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        for gidxs in groups.values():
            if len(gidxs) == 1:
                out.append(plist[gidxs[0]])
            else:
                merged = merge_proposals_generic([plist[i] for i in gidxs], ptype, note=f"IoM{int(thr*100)}")
                if logger:
                    logger.info(f"IoM{int(thr*100)} merged {len(gidxs)} proposals for type={ptype}")
                out.append(merged)

    return out

# -----------------------
# Visualization helpers
# -----------------------

def _draw_rect_border(draw: "ImageDraw.ImageDraw", box: List[int], color: Tuple[int,int,int], width: int=3):
    x1, y1, x2, y2 = box

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    if x2 - x1 < 1 or y2 - y1 < 1:
        return

    for w in range(width):
        left = x1 - w
        top = y1 - w
        right = x2 + w
        bottom = y2 + w

        if left < right and top < bottom:
            draw.rectangle([left, top, right, bottom], outline=color, width=1)

def _get_font(font_size: int = 20):
    try:
        import platform
        system = platform.system()
        if system == "Windows":
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/tahoma.ttf"
            ]
        elif system == "Darwin":
            font_paths = [
                "/System/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf"
            ]
        else:
            font_paths = [
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/arial.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
            ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except Exception:
                    continue
        return ImageFont.load_default()
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None

def _put_text(draw: "ImageDraw.ImageDraw", xy: Tuple[int,int], text: str, color=(255,255,255), bg=(0,0,0), font_size: int = 20):
    font = _get_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
    except AttributeError:
        try:
            bw, bh = draw.textsize(text, font=font)
        except AttributeError:
            bw, bh = len(text) * (font_size // 2), font_size
    
    x, y = xy
    pad = 4
    draw.rectangle([x, y, x+bw+pad*2, y+bh+pad*2], fill=bg, outline=color, width=1)
    draw.text((x+pad, y+pad), text, fill=color, font=font)

def categorize_for_precise(cat: str) -> str:
    c = normalize_category(cat)
    for k in COLOR_PRECISE.keys():
        if k == 'other': continue
        if c.startswith(k):
            return k
    return 'other'

def visualize_sample(image_root: str,
                     sample: Dict[str, Any],
                     proposals: List[Dict[str, Any]],
                     out_dir: str,
                     logger: Optional[logging.Logger]=None):
    if not PIL_OK:
        if logger: logger.warning("Pillow not available; skip visualization.")
        return
    os.makedirs(out_dir, exist_ok=True)
    image_name = sample.get('image_name', '')
    base = os.path.basename(image_name)
    img_path = os.path.join(image_root, base)
    if not os.path.isfile(img_path):
        if logger: logger.warning(f"Image not found for vis: {img_path}")
        return

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        if logger: logger.warning(f"Open image failed: {img_path}, {e}")
        return

    draw = ImageDraw.Draw(img)
    precise = sample.get('Precise Grounding Position', []) or []
    cats = sample.get('Category', []) or []

    img_w, img_h = img.size
    base_font_size = max(16, min(img_w, img_h) // 50)

    # draw precise boxes (only for 'other' categories)
    for i, b in enumerate(precise):
        if i >= len(cats): continue
        if is_other_target(cats[i]):
            tag = categorize_for_precise(cats[i])
            color = COLOR_PRECISE.get(tag, (255,255,255))
            _draw_rect_border(draw, [int(x) for x in b], color, width=3)
            x1,y1,_,_ = [int(x) for x in b]
            _put_text(draw, (x1, max(0, y1-30)), tag, font_size=base_font_size)

    # draw proposals
    for p in proposals:
        b = p['bbox']
        ptype = p.get('type', 'other')
        color = COLOR_PROPOSAL.get(ptype, (255,255,255))
        _draw_rect_border(draw, [int(x) for x in b], color, width=6)
        
        label = f"proposal:{ptype}"
        if p.get('cluster_id', None) is not None:
            label += f"/c{p['cluster_id']}"
        if p.get('merge_note'):
            label += f" ({p['merge_note']} from {p.get('merged_from', '?')})"
        elif p.get('merged_from'):
            label += f" (H-merged from {p['merged_from']})"
        
        x1,y1,_,_ = [int(x) for x in b]
        _put_text(draw, (x1, max(0, y1-40)), label, color=(0,0,0), bg=color, font_size=base_font_size+2)

    img_id = extract_image_id_from_path(image_name) or os.path.splitext(base)[0]
    out_path = os.path.join(out_dir, f"{img_id}_vis.jpg")
    try:
        img.save(out_path, quality=95)
    except Exception as e:
        if logger: logger.warning(f"Save vis failed: {out_path}, {e}")

# -----------------------
# Main per-image logic
# -----------------------
def process_image_sample(sample: Dict[str, Any],
                         logger: Optional[logging.Logger]=None) -> Dict[str, Any]:
    image_name = sample.get('image_name', '')
    img_w = int(sample.get('image_width', 0))
    img_h = int(sample.get('image_height', 0))
    categories = sample.get('Category', [])
    precise = sample.get('Precise Grounding Position', [])
    contextual = sample.get('Contextual Grounding Position', [])
    full_answer_text = sample.get('Annotations', '') or sample.get('Annotations', '') or ''

    if not isinstance(categories, list) or not isinstance(precise, list):
        return {
            "image_name": image_name,
            "image_width": img_w,
            "image_height": img_h,
            "proposals": []
        }

    if not isinstance(contextual, list):
        contextual = []

    # Collect indices for 'other' categories only
    other_idx: List[int] = []
    for i, cat in enumerate(categories):
        if i >= len(precise):
            continue
        if is_other_target(cat):
            other_idx.append(i)

    proposals: List[Dict[str, Any]] = []

    # ---------- OTHERS (clustering directly on bboxes) ----------
    if len(other_idx) > 0:
        pts = []
        items = []  # (cx,cy), lesion_index, bbox
        for i_lesion in other_idx:
            ctx_box = get_ctx_or_precise(i_lesion, precise, contextual)
            if ctx_box is None:
                continue
            cx, cy = box_center(ctx_box)
            pts.append([cx, cy])
            items.append(((cx, cy), i_lesion, ctx_box))

        if items:
            pts = np.array(pts, dtype=np.float64)
            K = choose_k_for_other(len(items))
            labels = np.zeros(len(items), dtype=np.int32) if K <= 1 else kmeans(pts, min(K, len(items)))

            for cl in range(labels.max()+1):
                lesion_indices = [items[k][1] for k in range(len(items)) if labels[k]==cl]
                cluster_boxes = [items[k][2] for k in range(len(items)) if labels[k]==cl]

                bbox = box_union(cluster_boxes) if cluster_boxes else [0, 0, img_w, img_h]

                proposals.append({
                    "type": "other",
                    "cluster_id": int(cl),
                    "source_categories": [categories[i] for i in lesion_indices],
                    "lesion_indices": lesion_indices,
                    "bbox": [int(x) for x in bbox],
                    "cur_proposal_gt": build_cur_proposal_gt(full_answer_text, lesion_indices),
                })

    # horizontal + vertical merge
    proposals = merge_overlapping_other_proposals(proposals, logger, 0.4, 0.8)

    # IoM >= 0.7 merge
    proposals = merge_by_iom_per_type(proposals, thr=0.7, logger=logger)

    return {
        "image_name": image_name,
        "image_width": img_w,
        "image_height": img_h,
        "proposals": proposals
    }

# -----------------------
# IO & main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Generate proposals for dental panoramic X-rays based on bbox clustering.")
    parser.add_argument("--new-json", type=str, default='/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/2_400/Multimodal_data_Sichuan_Uni_Children.json',
                        help="Path to the aggregated new_Multimodal_data JSON")
    parser.add_argument("--out", type=str, default='/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/2_400/proposals_data2.json',
                        help="Output path for proposals JSON")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")

    # visualization options
    parser.add_argument("--visualize", action="store_true", help="If set, output per-image visualization.")
    parser.add_argument("--image-root", type=str,
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/2_400/images",
                        help="Root directory containing original images")
    parser.add_argument("--vis-out", type=str, default='proposal_output_test', help="Directory to save visualization images")

    # sample limiting
    parser.add_argument("--max-samples", type=int, default=0, help="Process at most N samples (0 = no limit)")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip first M samples before processing")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("proposal_gen")

    with open(args.new_json, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    assert isinstance(new_data, list), "new_json must be a list of samples"

    out_dir_for_vis = args.vis_out
    if args.visualize:
        if not PIL_OK:
            logger.warning("Pillow is not installed; visualization will be skipped.")
        if out_dir_for_vis is None:
            out_dir_for_vis = os.path.join(os.path.dirname(args.out) or ".", "vis")
        os.makedirs(out_dir_for_vis, exist_ok=True)

    results = {"images": []}
    total = len(new_data)
    start = min(max(0, args.sample_offset), total)
    end = total if args.max_samples <= 0 else min(total, start + args.max_samples)

    logger.info(f"Total samples: {total} | Processing range: [{start}, {end})")
    for idx in range(start, end):
        sample = new_data[idx]
        if (idx - start) % 100 == 0:
            logger.info(f"Processing {idx+1-start}/{end-start} (global idx {idx}) ...")
        try:
            out = process_image_sample(sample, logger)
            results["images"].append(out)
            if args.visualize and PIL_OK:
                if out.get("proposals"):
                    visualize_sample(args.image_root, sample, out["proposals"], out_dir_for_vis, logger)
        except Exception as e:
            logger.exception(f"Failed on sample #{idx} ({sample.get('image_name','?')}): {e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved proposals to {args.out}")
    if args.visualize:
        logger.info(f"Visualization saved to {out_dir_for_vis}")

if __name__ == "__main__":
    main()