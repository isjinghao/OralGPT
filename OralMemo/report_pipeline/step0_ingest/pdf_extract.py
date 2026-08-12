"""step0 PDF 摄取: 用 MinerU(pipeline 后端) 解析 PDF, 产出:
  - fulltext.json : {"pages": [{"page","text"}]}(按阅读顺序聚合文本)
  - tables.json   : [{"page","caption","html"}](MinerU 表格结构识别; 表格即图片也识别为表)
  - images/*.jpg  : 图片区域
  - 返回 images_map(图注↔图片, 以 "Figure N" 为权威身份)

MinerU 由 `pip install "mineru[core]"` 提供; 模型源用 ModelScope(可用 MINERU_MODEL_SOURCE 覆盖)。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

FIG_RE = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)
# 图注式: "Figure N." / "Figure N:" (数字后跟句点或冒号), 用于识别真正的图注、排除正文中的 "(Figure 4)" 等引用
CAP_FIG_RE = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)\s*[.:]", re.IGNORECASE)


def build_fulltext(pages: list[dict]) -> str:
    """把按页存的 fulltext.json 拼回带分页标记的整段文本(供 LLM 抽取/校验使用)。"""
    return "".join(f"\n===== PAGE {p['page']} =====\n{p['text']}" for p in pages)


def read_fulltext(raw_dir: Path) -> str:
    data = json.loads((raw_dir / "fulltext.json").read_text(encoding="utf-8"))
    return build_fulltext(data["pages"])


def _run_mineru(pdf_path: Path, work_dir: Path) -> tuple[list, Path]:
    """调用 MinerU CLI 解析 PDF, 返回 (content_list, auto_dir)。"""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    subprocess.run(
        ["mineru", "-p", str(pdf_path), "-o", str(work_dir), "-b", "pipeline", "-m", "auto"],
        check=True, env=env,
    )
    stem = pdf_path.stem
    auto = work_dir / stem / "auto"
    content = json.loads((auto / f"{stem}_content_list.json").read_text(encoding="utf-8"))
    return content, auto


def _block_text(blk: dict) -> str:
    t = blk.get("type")
    if t in ("text", "header", "page_footnote"):
        return (blk.get("text") or "").strip()
    if t == "list":
        items = blk.get("list_items") or []
        return "\n".join(str(x) for x in items).strip()
    if t == "table":
        return " ".join(blk.get("table_caption") or []).strip()
    if t == "image":
        return " ".join(c for c in (blk.get("image_caption") or []) if FIG_RE.search(str(c))).strip()
    return ""


def extract_pdf(pdf_path: Path, out_dir: Path, images_dir: Path, rel_base: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        shutil.rmtree(images_dir)  # 清理历史残留图片
    images_dir.mkdir(parents=True, exist_ok=True)

    content, auto = _run_mineru(pdf_path, out_dir / "_mineru")

    # (1) 逐页文本(阅读顺序)
    pages_map: dict[int, list[str]] = {}
    for blk in content:
        page = int(blk.get("page_idx", 0)) + 1
        txt = _block_text(blk)
        if txt:
            pages_map.setdefault(page, []).append(txt)
    pages = [{"page": p, "text": "\n".join(pages_map[p])} for p in sorted(pages_map)]
    (out_dir / "fulltext.json").write_text(
        json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")

    # (2) 表格(MinerU 结构识别; 表格即图片也在此)
    tables = []
    for blk in content:
        if blk.get("type") == "table":
            html = (blk.get("table_body") or "").strip()
            if html:
                tables.append({
                    "page": int(blk.get("page_idx", 0)) + 1,
                    "caption": " ".join(blk.get("table_caption") or []).strip(),
                    "html": html,
                })
    (out_dir / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")

    # (3) 图片 + 图注
    #   - 先从"图注式" (Figure N. / Figure N:) 统计全篇 figure(含 text 块里的图注/图注页)。
    #   - 若 图块数 == figure 数: 按阅读顺序 1:1 分配(每张图一个独立 figure)。
    #   - 否则(图块多于 figure, 说明存在多面板图): 按页纵向聚类, 每个 figure 的 image 为其全部子图列表。
    img_blocks = [b for b in content if b.get("type") == "image"]

    cap_by_fig: dict[int, str] = {}
    # 图片块自带的图注(最干净, 每条一个 figure)
    for b in img_blocks:
        for cap in (b.get("image_caption") or []):
            m = CAP_FIG_RE.search(str(cap))
            if m:
                cap_by_fig.setdefault(int(m.group(1)), str(cap).strip())
    # 补充: text/header 块里的图注(某些 figure 的 caption 是独立文本块)
    for b in content:
        if b.get("type") in ("text", "header"):
            txt = b.get("text") or ""
            for m in CAP_FIG_RE.finditer(txt):
                cap_by_fig.setdefault(int(m.group(1)), txt[m.start():m.start() + 300].strip())
    all_figs = sorted(cap_by_fig)

    def _y(b):
        return (b.get("bbox") or [0, 0, 0, 0])[1]

    def _cluster(blocks: list, k: int) -> list[list]:
        # 按 y_top 排序, 在最大的 (k-1) 个纵向间隔处切分成 k 组(组内为一个 figure 的多个子图)
        blocks = sorted(blocks, key=_y)
        if k <= 1 or len(blocks) <= 1:
            return [blocks]
        ys = [_y(b) for b in blocks]
        cuts = set(sorted(range(1, len(blocks)), key=lambda i: -(ys[i] - ys[i - 1]))[:k - 1])
        groups, cur = [], []
        for i, b in enumerate(blocks):
            if i in cuts:
                groups.append(cur)
                cur = []
            cur.append(b)
        groups.append(cur)
        return groups

    kept = 0

    def _copy(b) -> str | None:
        nonlocal kept
        src = auto / (b.get("img_path") or "")
        if not src.is_file():
            return None
        ext = src.suffix.lstrip(".") or "jpg"
        fname = f"p{int(b.get('page_idx', 0)) + 1:02d}_img{kept:03d}.{ext}"
        shutil.copyfile(src, images_dir / fname)
        kept += 1
        return (images_dir / fname).relative_to(rel_base).as_posix()

    images_map: dict = {}
    if img_blocks and len(img_blocks) == len(all_figs):
        # 图块数 == figure 数 → 按阅读顺序 1:1(每张图独立一个 figure)
        for b, fig in zip(img_blocks, all_figs):
            p = _copy(b)
            if p:
                images_map[f"Figure {fig}"] = {"images": [p], "caption": cap_by_fig[fig]}
    else:
        # 多面板图: 按页纵向聚类, figure 按编号升序 <-> 聚类自上而下
        by_page: dict[int, list] = {}
        for b in img_blocks:
            by_page.setdefault(int(b.get("page_idx", 0)), []).append(b)
        for page in sorted(by_page):
            blocks = by_page[page]
            figs = sorted({int(m.group(1)) for b in blocks for c in (b.get("image_caption") or [])
                           if (m := CAP_FIG_RE.search(str(c)))})
            if not figs:
                continue
            clusters = _cluster(blocks, len(figs))
            for fig, cluster in zip(figs, clusters):
                figkey = f"Figure {fig}"
                if figkey in images_map:
                    continue
                paths = [p for b in cluster if (p := _copy(b))]
                if paths:
                    images_map[figkey] = {"images": paths, "caption": cap_by_fig.get(fig, "")}

    # 清理 MinerU 中间产物(已提取所需内容)
    shutil.rmtree(out_dir / "_mineru", ignore_errors=True)

    return {
        "n_pages": len(pages),
        "n_images_kept": kept,
        "n_tables": len(tables),
        "images_map": images_map,
    }
