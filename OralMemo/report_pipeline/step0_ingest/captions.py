from __future__ import annotations

import re

# 通用的图注抽取
CAPTION_RE = re.compile(
    r"(?:Figure|Fig\.?)\s*(\d+)\s*[:\.\-]\s*(.+?)"
    r"(?=(?:Figure|Fig\.?)\s*\d+\s*[:\.\-]|Patient Perspective|Discussion|Conclusions?\b"
    r"|Clinical Outcomes|Timeline and Follow|Abbreviations|Acknowledge|References|Note\*"
    r"|ARTICLE IN PRESS|===== PAGE|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_figure_captions(fulltext: str) -> list[dict]:
    """从全文文本中通用地抽取图注
    返回按图号排序、去重的 [{"figure": "Figure N", "caption": "..."}]。
    """
    seen: dict[int, str] = {}
    for m in CAPTION_RE.finditer(fulltext):
        num = int(m.group(1))
        caption = re.sub(r"\s+", " ", m.group(2)).strip()
        if len(caption) < 8:
            continue
        if num not in seen or len(caption) > len(seen[num]):
            seen[num] = caption
    return [{"figure": f"Figure {n}", "caption": seen[n]} for n in sorted(seen)]


def build_images_map(kept_images: list[str], captions: list[dict]) -> dict:
    # 把有序的图注与有序的有效图片按顺序对齐
    amap: dict = {}
    n = min(len(kept_images), len(captions))
    for i in range(n):
        fig = captions[i]["figure"]
        amap[fig] = {"image": kept_images[i], "caption": captions[i]["caption"]}
    amap["unmapped_images"] = kept_images[n:]
    return amap
