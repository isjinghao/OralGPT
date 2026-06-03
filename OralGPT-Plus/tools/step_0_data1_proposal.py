#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional
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
def box_area(b: List[int]) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_center(b: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def expand_box(bbox: List[int], expansion: int, img_w: int, img_h: int) -> List[int]:
    """
    Expand a bounding box, clamped within image bounds.
    Args:
        bbox: [x1, y1, x2, y2]
        expansion: expansion in pixels
        img_w: image width
        img_h: image height
    Returns:
        expanded bbox
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - expansion)
    y1 = max(0, y1 - expansion)
    x2 = min(img_w, x2 + expansion)
    y2 = min(img_h, y2 + expansion)
    return [int(x1), int(y1), int(x2), int(y2)]

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

# -----------------------
# Proposal merging
# -----------------------
def merge_overlapping_proposals(proposals: List[Dict[str, Any]], 
                                logger: Optional[logging.Logger] = None,
                                min_h_overlap_ratio: float = 0.0,
                                min_v_overlap_ratio: float = 0.0) -> List[Dict[str, Any]]:
    """Merge overlapping proposals (horizontal + vertical)."""
    if len(proposals) <= 1:
        return proposals
    
    n = len(proposals)
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
            bbox_i = proposals[i]['bbox']
            bbox_j = proposals[j]['bbox']
            if boxes_overlap_horizontal(bbox_i, bbox_j,
                                        min_h_overlap_ratio=min_h_overlap_ratio,
                                        min_v_overlap_ratio=min_v_overlap_ratio):
                union(i, j)
                if logger:
                    logger.debug(f"Merging proposals {i} and {j} due to horizontal+vertical overlap")
    
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    
    merged_proposals = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            merged_proposals.append(proposals[group_indices[0]])
        else:
            proposals_to_merge = [proposals[i] for i in group_indices]
            merged_proposal = merge_proposals(proposals_to_merge)
            merged_proposals.append(merged_proposal)
            if logger:
                logger.info(f"Merged {len(group_indices)} proposals into one (H+V overlap)")
    
    return merged_proposals

def merge_proposals(proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple proposals into one."""
    if not proposals:
        return {}
    if len(proposals) == 1:
        return proposals[0]
    
    all_bboxes = [p['bbox'] for p in proposals]
    merged_bbox = box_union(all_bboxes)
    
    cluster_ids = []
    box_indices = []
    
    for p in proposals:
        if p.get('cluster_id') is not None:
            cluster_ids.append(p.get('cluster_id'))
        box_indices.extend(p.get('box_indices', []))
    
    box_indices = sorted(list(set(box_indices)))
    
    if cluster_ids:
        new_cluster_id = min(cluster_ids) if all(isinstance(x, int) for x in cluster_ids) else "merged"
        if len(set(cluster_ids)) > 1:
            new_cluster_id = f"merged_{new_cluster_id}" if isinstance(new_cluster_id, int) else "merged"
    else:
        new_cluster_id = "merged"
    
    return {
        "cluster_id": new_cluster_id,
        "box_indices": box_indices,
        "bbox": [int(x) for x in merged_bbox],
        "merged_from": len(proposals),
        "is_expanded": False,
    }

def merge_by_iom(proposals: List[Dict[str, Any]],
                 thr: float = 0.7,
                 logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """Merge proposals with IoM >= thr."""
    if len(proposals) <= 1:
        return proposals

    n = len(proposals)
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
            bi, bj = proposals[i]['bbox'], proposals[j]['bbox']
            if iom_ge(bi, bj, thr):
                union(i, j)
                if logger:
                    logger.debug(f"IoM{int(thr*100)} merge: i={i}, j={j}")

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    out = []
    for gidxs in groups.values():
        if len(gidxs) == 1:
            out.append(proposals[gidxs[0]])
        else:
            merged = merge_proposals([proposals[i] for i in gidxs])
            merged['merge_note'] = f"IoM{int(thr*100)}"
            if logger:
                logger.info(f"IoM{int(thr*100)} merged {len(gidxs)} proposals")
            out.append(merged)

    return out

def expand_all_proposals(proposals: List[Dict[str, Any]], 
                        expansion: int,
                        img_w: int, 
                        img_h: int,
                        area_threshold_ratio: float = 0.15,
                        logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Expand all final proposals, skipping proposals whose area is too large.
    Args:
        proposals: list of proposals
        expansion: expansion in pixels
        img_w: image width
        img_h: image height
        area_threshold_ratio: area ratio threshold (default 0.2)
        logger: logger
    Returns:
        list of expanded proposals
    """
    img_area = img_w * img_h
    area_threshold = img_area * area_threshold_ratio
    
    expanded_proposals = []
    skipped_count = 0
    expanded_count = 0
    
    for p in proposals:
        bbox = p['bbox']
        bbox_area = box_area(bbox)

        if bbox_area > area_threshold:
            expanded_p = p.copy()
            expanded_p['is_expanded'] = False
            expanded_p['skip_reason'] = f"area_too_large ({bbox_area:.0f} > {area_threshold:.0f})"
            expanded_proposals.append(expanded_p)
            skipped_count += 1
            
            if logger:
                logger.debug(f"Skipped expansion for proposal {p.get('cluster_id', '?')}: "
                           f"area {bbox_area:.0f} > threshold {area_threshold:.0f} "
                           f"({area_threshold_ratio*100:.1f}% of image)")
        else:
            expanded_bbox = expand_box(bbox, expansion, img_w, img_h)

            expanded_p = p.copy()
            expanded_p['bbox'] = expanded_bbox
            expanded_p['is_expanded'] = True
            expanded_p['original_bbox'] = bbox
            expanded_p['original_area'] = bbox_area
            
            expanded_proposals.append(expanded_p)
            expanded_count += 1
            
            if logger:
                logger.debug(f"Expanded proposal {p.get('cluster_id', '?')}: "
                           f"{bbox} -> {expanded_bbox} (area: {bbox_area:.0f})")
    
    if logger and (skipped_count > 0 or expanded_count > 0):
        logger.info(f"Expansion summary: expanded={expanded_count}, skipped={skipped_count} "
                   f"(threshold: {area_threshold_ratio*100:.1f}% of image area)")
    
    return expanded_proposals

# -----------------------
# Visualization helpers
# -----------------------
COLOR_PRECISE = (255, 165, 0)      # precise grounding box
COLOR_EXPANDED = (0, 255, 0)       # expanded proposal
COLOR_NOT_EXPANDED = (255, 255, 0) # proposal not expanded (area too large)

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

    img_w, img_h = img.size
    base_font_size = max(16, min(img_w, img_h) // 50)

    # draw precise grounding boxes
    for i, b in enumerate(precise):
        _draw_rect_border(draw, [int(x) for x in b], COLOR_PRECISE, width=3)
        x1, y1, _, _ = [int(x) for x in b]
        _put_text(draw, (x1, max(0, y1-30)), f"Box {i}", font_size=base_font_size)

    # draw proposals
    for p in proposals:
        b = p['bbox']
        is_expanded = p.get('is_expanded', False)

        color = COLOR_EXPANDED if is_expanded else COLOR_NOT_EXPANDED
        _draw_rect_border(draw, [int(x) for x in b], color, width=6)

        label = f"Cluster {p['cluster_id']}"
        if is_expanded:
            label += " (Expanded)"
        else:
            skip_reason = p.get('skip_reason', '')
            if skip_reason:
                label += " (Too Large)"
        
        if p.get('merge_note'):
            label += f" ({p['merge_note']} from {p.get('merged_from', '?')})"
        elif p.get('merged_from'):
            label += f" (merged from {p['merged_from']})"
        
        x1, y1, _, _ = [int(x) for x in b]
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
                         expansion_size: int = 100,
                         area_threshold_ratio: float = 0.2,
                         logger: Optional[logging.Logger]=None) -> Dict[str, Any]:
    """
    Process a single image sample.
    Args:
        sample: image sample data
        expansion_size: expansion in pixels (applied to proposals below threshold)
        area_threshold_ratio: area ratio threshold (default 0.2)
        logger: logger
    """
    image_name = sample.get('image_name', '')
    img_w = int(sample.get('image_width', 0))
    img_h = int(sample.get('image_height', 0))
    precise = sample.get('Precise Grounding Position', [])

    if not isinstance(precise, list) or len(precise) == 0:
        return {
            "image_name": image_name,
            "image_width": img_w,
            "image_height": img_h,
            "proposals": []
        }

    # collect bbox centers
    pts = []
    boxes = []
    for i, bbox in enumerate(precise):
        if bbox and len(bbox) == 4:
            cx, cy = box_center([int(x) for x in bbox])
            pts.append([cx, cy])
            boxes.append(([int(x) for x in bbox], i))

    if not boxes:
        return {
            "image_name": image_name,
            "image_width": img_w,
            "image_height": img_h,
            "proposals": []
        }

    # clustering
    pts = np.array(pts, dtype=np.float64)
    K = 1
    labels = np.zeros(len(boxes), dtype=np.int32) if K <= 1 else kmeans(pts, min(K, len(boxes)))

    # build a proposal per cluster (not expanded yet)
    proposals = []
    for cl in range(labels.max() + 1):
        cluster_boxes = [boxes[k][0] for k in range(len(boxes)) if labels[k] == cl]
        box_indices = [boxes[k][1] for k in range(len(boxes)) if labels[k] == cl]

        bbox = box_union(cluster_boxes)
        
        if logger:
            logger.debug(f"Image {image_name}, Cluster {cl}: {len(cluster_boxes)} boxes, bbox before expansion: {bbox}")
        
        proposals.append({
            "cluster_id": int(cl),
            "box_indices": box_indices,
            "bbox": [int(x) for x in bbox],
            "is_expanded": False,
        })

    # horizontal + vertical merge
    proposals = merge_overlapping_proposals(proposals, logger, min_h_overlap_ratio=0.4, min_v_overlap_ratio=0.8)

    # IoM >= 0.7 merge
    proposals = merge_by_iom(proposals, thr=0.7, logger=logger)

    # finally expand all proposals (skipping ones with too large area)
    proposals = expand_all_proposals(proposals, expansion_size, img_w, img_h,
                                    area_threshold_ratio=area_threshold_ratio, 
                                    logger=logger)
    
    expanded_count = sum(1 for p in proposals if p.get('is_expanded', False))
    skipped_count = len(proposals) - expanded_count
    
    if logger:
        logger.info(f"Image {image_name}: Generated {len(proposals)} proposals, "
                   f"expanded={expanded_count}, skipped={skipped_count}")

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
    parser.add_argument("--new-json", type=str, default='/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/1_380/Multimodal_data_Tufts_Dental_Database.json',
                        help="Path to the aggregated new_Multimodal_data JSON")
    parser.add_argument("--out", type=str, default='/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/1_380/proposals_data1.json',
                        help="Output path for proposals JSON")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")

    # expansion parameter
    parser.add_argument("--expansion", type=int, default=100, 
                        help="Expansion size in pixels for proposals (default: 100)")
    parser.add_argument("--area-threshold", type=float, default=0.15,
                        help="Area threshold ratio (0-1). Proposals larger than this ratio of image area won't be expanded (default: 0.2 = 1/5)")

    # visualization options
    parser.add_argument("--visualize", action="store_true", help="If set, output per-image visualization.")
    parser.add_argument("--image-root", type=str,
                        default="/hpc2hdd/home/yfan546/workplace/mllm_playground/mllm_2/r1/source_data/1_380/images",
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
    logger.info(f"Expansion size: {args.expansion} pixels")
    logger.info(f"Area threshold: {args.area_threshold*100:.1f}% of image area (proposals larger than this won't be expanded)")
    
    for idx in range(start, end):
        sample = new_data[idx]
        if (idx - start) % 100 == 0:
            logger.info(f"Processing {idx+1-start}/{end-start} (global idx {idx}) ...")
        try:
            out = process_image_sample(sample, 
                                      expansion_size=args.expansion,
                                      area_threshold_ratio=args.area_threshold,
                                      logger=logger)
            results["images"].append(out)
            if args.visualize and PIL_OK:
                if out.get("proposals"):
                    visualize_sample(args.image_root, sample, out["proposals"], out_dir_for_vis, logger)
        except Exception as e:
            logger.exception(f"Failed on sample #{idx} ({sample.get('image_name','?')}): {e}")

    # summary stats
    total_proposals = sum(len(img["proposals"]) for img in results["images"])
    expanded_proposals = sum(1 for img in results["images"] 
                            for p in img["proposals"] 
                            if p.get("is_expanded", False))
    skipped_proposals = total_proposals - expanded_proposals
    
    logger.info(f"=" * 60)
    logger.info(f"Processing complete!")
    logger.info(f"Total proposals generated: {total_proposals}")
    logger.info(f"Expanded proposals: {expanded_proposals} ({expanded_proposals/total_proposals*100:.1f}%)")
    logger.info(f"Skipped proposals (too large): {skipped_proposals} ({skipped_proposals/total_proposals*100:.1f}%)")
    logger.info(f"=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved proposals to {args.out}")
    if args.visualize:
        logger.info(f"Visualization saved to {out_dir_for_vis}")

if __name__ == "__main__":
    main()