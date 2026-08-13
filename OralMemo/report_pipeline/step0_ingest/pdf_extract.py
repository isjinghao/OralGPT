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
from html import unescape
from pathlib import Path

HSPACE = r"[ \t\u00a0]"
NORMAL_FIG_PREFIX = r"(?:Figure|Fig\.?|Fi\.?)"
OCR_FIG_PREFIX = rf"F{HSPACE}+I{HSPACE}+G{HSPACE}+U{HSPACE}+R{HSPACE}+E"
FIG_PATTERN = rf"(?:{NORMAL_FIG_PREFIX}{HSPACE}*(\d+)|{OCR_FIG_PREFIX}{HSPACE}*(\d(?:{HSPACE}+\d)+|\d+))"
FIG_RE = re.compile(rf"\b{FIG_PATTERN}[A-Za-z]?", re.IGNORECASE)
TABLE_PREFIX = rf"(?:Table|T{HSPACE}*A{HSPACE}*B{HSPACE}*L{HSPACE}*E)"
TABLE_RE = re.compile(rf"^{TABLE_PREFIX}{HSPACE}*\d+", re.IGNORECASE)
# 图注须从行首开始；兼容不换行空格、OCR 字符间空格、Fi 缩写及简单 HTML 标点。
CAP_FIG_RE = re.compile(
    rf"^{HSPACE}*{FIG_PATTERN}[A-Za-z]?(?:<[^>]+>[.:]</[^>]+>)?(?:{HSPACE}*[.:]{HSPACE}*|{HSPACE}+|{HSPACE}*$)",
    re.IGNORECASE | re.MULTILINE,
)


def _figure_number(match: re.Match) -> int:
    return int(re.sub(HSPACE, "", match.group(1) or match.group(2)))


def _plain_caption(value: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", str(value)))


def _caption_figures(value: str) -> list[int]:
    text = _plain_caption(value)
    if not CAP_FIG_RE.search(text):
        return []
    return [_figure_number(match) for match in FIG_RE.finditer(text)]


def _pdf_captions(pdf_path: Path) -> dict[int, list[tuple[int, str]]]:
    from pypdf import PdfReader

    captions = {}
    for page, pdf_page in enumerate(PdfReader(pdf_path).pages):
        text = pdf_page.extract_text() or ""
        matches = [
            (_figure_number(match), text[match.start():].splitlines()[0].strip())
            for match in CAP_FIG_RE.finditer(text)
        ]
        if matches:
            captions[page] = matches
    return captions


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


def _visual_captions(blk: dict) -> list[str]:
    key = {
        "chart": "chart_caption",
        "table": "table_caption",
    }.get(blk.get("type"), "image_caption")
    return blk.get(key) or []


def _block_text(blk: dict) -> str:
    t = blk.get("type")
    if t in ("text", "header", "page_footnote"):
        return (blk.get("text") or "").strip()
    if t == "list":
        items = blk.get("list_items") or []
        return "\n".join(str(x) for x in items).strip()
    if t == "table":
        return " ".join(blk.get("table_caption") or []).strip()
    if t in ("image", "chart"):
        return " ".join(c for c in _visual_captions(blk) if _caption_figures(c)).strip()
    return ""


def _markdown_captions(auto_dir: Path) -> dict[str, list[str]]:
    lines = next(auto_dir.glob("*.md")).read_text(encoding="utf-8").splitlines()
    captions = {}
    for index, line in enumerate(lines):
        match = re.search(r"!\[[^]]*\]\(([^)]+)\)", line)
        if not match:
            continue
        following_captions = []
        for following in lines[index + 1:]:
            following = following.strip()
            if not following:
                continue
            if following.startswith("![") or not _caption_figures(following):
                break
            following_captions.append(following)
        if following_captions:
            captions[Path(match.group(1)).name] = following_captions
    return captions


def _markdown_table_captions(auto_dir: Path) -> dict[str, str]:
    lines = next(auto_dir.glob("*.md")).read_text(encoding="utf-8").splitlines()
    captions = {}
    for index, line in enumerate(lines):
        if not line.strip().startswith("<table"):
            continue
        previous = next((item.strip() for item in reversed(lines[:index]) if item.strip()), "")
        if TABLE_RE.match(previous):
            captions[line.strip()] = previous
    return captions


def _table_caption(captions: list[str], body: str, markdown_caption: str = "") -> str:
    caption = " ".join(captions).strip() or markdown_caption
    if caption:
        return caption
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.IGNORECASE | re.DOTALL)[:2]
    text = [re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", row))).strip() for row in rows]
    return " ".join(text) if text and TABLE_RE.match(text[0]) else ""


def extract_pdf(pdf_path: Path, out_dir: Path, images_dir: Path, rel_base: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        shutil.rmtree(images_dir)  # 清理历史残留图片
    images_dir.mkdir(parents=True, exist_ok=True)

    content, auto = _run_mineru(pdf_path, out_dir / "_mineru")
    markdown_captions = _markdown_captions(auto)
    markdown_table_captions = _markdown_table_captions(auto)
    pdf_captions = _pdf_captions(pdf_path)

    # (1) 逐页文本(阅读顺序)
    pages_map: dict[int, list[str]] = {}
    used_pdf_caption_pages = set()
    for blk in content:
        page_idx = int(blk.get("page_idx", 0))
        page = page_idx + 1
        txt = _block_text(blk)
        if blk.get("type") in ("image", "chart") and not txt:
            txt = " ".join(markdown_captions.get(Path(blk.get("img_path") or "").name, []))
            if not txt and len(pdf_captions.get(page_idx, [])) == 1 and page_idx not in used_pdf_caption_pages:
                txt = pdf_captions[page_idx][0][1]
                used_pdf_caption_pages.add(page_idx)
        if txt:
            pages_map.setdefault(page, []).append(txt)
    pages = [{"page": p, "text": "\n".join(pages_map[p])} for p in sorted(pages_map)]
    (out_dir / "fulltext.json").write_text(
        json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")

    # (2) 表格(MinerU 结构识别; 表格即图片也在此)
    tables = []
    for blk in content:
        if blk.get("type") == "table":
            captions = blk.get("table_caption") or []
            figure_captions = [caption for caption in captions if _caption_figures(caption)]
            table_captions = [caption for caption in captions if TABLE_RE.match(str(caption).strip())]
            if figure_captions and not table_captions:
                continue
            html = (blk.get("table_body") or "").strip()
            if html:
                tables.append({
                    "page": int(blk.get("page_idx", 0)) + 1,
                    "caption": _table_caption(
                        table_captions or captions,
                        html,
                        markdown_table_captions.get(html, ""),
                    ),
                    "html": html,
                })
    (out_dir / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")

    # (3) 图片 + 图注
    #   - 先从 Figure/Fig 图注统计全篇 figure(含 text 块里的图注/图注页)。
    #   - 若 图块数 == figure 数: 按阅读顺序 1:1 分配(每张图一个独立 figure)。
    #   - 否则(图块多于 figure, 说明存在多面板图): 按页纵向聚类, 每个 figure 的 image 为其全部子图列表。
    img_blocks = [
        block for block in content
        if block.get("type") in ("image", "chart")
        or (
            block.get("type") == "table"
            and any(_caption_figures(caption) for caption in _visual_captions(block))
        )
    ]

    cap_by_fig: dict[int, str] = {}
    figs_by_page: dict[int, set[int]] = {}
    figs_by_block: dict[int, set[int]] = {}
    # 图片块自带的图注(最干净, 也可能一张复合图对应多个 figure)
    for b in img_blocks:
        page = int(b.get("page_idx", 0))
        captions = list(_visual_captions(b))
        captions.extend(markdown_captions.get(Path(b.get("img_path") or "").name, []))
        for cap in captions:
            for fig in _caption_figures(cap):
                cap_by_fig.setdefault(fig, str(cap).strip())
                figs_by_page.setdefault(page, set()).add(fig)
                figs_by_block.setdefault(id(b), set()).add(fig)
    # 补充: text/header 块里的图注(某些 figure 的 caption 是独立文本块)
    for b in content:
        if b.get("type") in ("text", "header"):
            txt = _plain_caption(b.get("text") or "")
            for m in CAP_FIG_RE.finditer(txt):
                fig = _figure_number(m)
                cap_by_fig.setdefault(fig, txt[m.start():m.start() + 300].strip())
                figs_by_page.setdefault(int(b.get("page_idx", 0)), set()).add(fig)
    visual_pages = {int(block.get("page_idx", 0)) for block in img_blocks}
    for page, captions in pdf_captions.items():
        missing = [(fig, caption) for fig, caption in captions if fig not in cap_by_fig]
        if page in visual_pages and len(missing) == 1:
            fig, caption = missing[0]
            cap_by_fig[fig] = caption
            figs_by_page.setdefault(page, set()).add(fig)
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
    combined_blocks = {block_id for block_id, figs in figs_by_block.items() if len(figs) > 1}
    for block in img_blocks:
        if id(block) not in combined_blocks:
            continue
        path = _copy(block)
        if path:
            for fig in sorted(figs_by_block[id(block)]):
                images_map[f"Figure {fig}"] = {
                    "images": [path],
                    "caption": cap_by_fig.get(fig, ""),
                }

    remaining_blocks = [block for block in img_blocks if id(block) not in combined_blocks]
    remaining_figs = [fig for fig in all_figs if f"Figure {fig}" not in images_map]
    if remaining_blocks and len(remaining_blocks) == len(remaining_figs):
        # 图块数 == figure 数 → 按阅读顺序 1:1(每张图独立一个 figure)
        for b, fig in zip(remaining_blocks, remaining_figs):
            p = _copy(b)
            if p:
                images_map[f"Figure {fig}"] = {"images": [p], "caption": cap_by_fig[fig]}
    else:
        # 多面板图: 按页纵向聚类, figure 按编号升序 <-> 聚类自上而下
        by_page: dict[int, list] = {}
        for b in remaining_blocks:
            by_page.setdefault(int(b.get("page_idx", 0)), []).append(b)
        for page in sorted(by_page):
            blocks = by_page[page]
            figs = [
                fig for fig in sorted(figs_by_page.get(page, set()))
                if f"Figure {fig}" not in images_map
            ]
            if not figs:
                continue
            clusters = _cluster(blocks, len(figs))
            for fig, cluster in zip(figs, clusters):
                paths = [p for b in cluster if (p := _copy(b))]
                if paths:
                    images_map[f"Figure {fig}"] = {
                        "images": paths,
                        "caption": cap_by_fig.get(fig, ""),
                    }

    images_map = dict(sorted(
        images_map.items(),
        key=lambda item: int(item[0].split()[1]),
    ))

    # 清理 MinerU 中间产物(已提取所需内容)
    shutil.rmtree(out_dir / "_mineru", ignore_errors=True)

    return {
        "n_pages": len(pages),
        "n_images_kept": kept,
        "n_tables": len(tables),
        "images_map": images_map,
    }
