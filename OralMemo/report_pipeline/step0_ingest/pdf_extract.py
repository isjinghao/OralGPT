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
from collections import Counter
from html import unescape
from pathlib import Path

HSPACE = r"[ \t\u00a0]"
NORMAL_FIG_PREFIX = r"(?:Figure|Fig\.?|Fi\.?)"
OCR_FIG_PREFIX = rf"F{HSPACE}*I{HSPACE}*G{HSPACE}*U{HSPACE}*R{HSPACE}*E"
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
    match = CAP_FIG_RE.search(text)
    if not match:
        return []
    figures = [_figure_number(match)]
    tail = text[match.end():]
    combined = re.match(rf"(?:and|&|,){HSPACE}*{FIG_PATTERN}", tail, re.IGNORECASE)
    if combined:
        figures.append(_figure_number(combined))
    return figures


def _caption_panel_count(value: str) -> int:
    letters = []
    for group in re.findall(r"\(([^)]{1,30})\)", value.casefold()):
        letters.extend(re.findall(r"(?<![a-z])([a-t])(?![a-z])", group))
    return max((ord(letter) - ord("a") + 1 for letter in letters), default=1)


def _pdf_page_texts(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(pdf_path).pages]


def _pdf_caption(text: str, match: re.Match) -> str:
    lines = text[match.start():].splitlines()
    caption = lines[0].strip()
    label_only = match.end() - match.start() == len(caption)
    for line in lines[1:4]:
        line = line.strip()
        if not line or CAP_FIG_RE.match(line) or TABLE_RE.match(line):
            break
        if not label_only and re.search(r"[.!?][\"')\]]?$", caption):
            break
        caption = f"{caption} {line}"
        label_only = False
        if len(caption) >= 500:
            break
    return re.sub(r"\s+", " ", caption).strip()


def _pdf_captions(page_texts: list[str]) -> dict[int, list[tuple[int, str]]]:
    captions = {}
    for page, text in enumerate(page_texts):
        matches = [
            (_figure_number(match), _pdf_caption(text, match))
            for match in CAP_FIG_RE.finditer(text)
        ]
        if matches:
            captions[page] = matches
    return captions


def _caption_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _plain_caption(value).casefold())


def _more_complete_caption(current: str, pdf_caption: str) -> str:
    current_tokens = _caption_tokens(current)
    pdf_tokens = _caption_tokens(pdf_caption)
    if not current_tokens or len(pdf_tokens) >= len(current_tokens) + 2:
        return pdf_caption
    return current


def _should_use_pdf_text(pdf_text: str, mineru_text: str) -> bool:
    pdf_tokens = re.findall(r"[a-z0-9]+", pdf_text.casefold())
    mineru_tokens = re.findall(r"[a-z0-9]+", mineru_text.casefold())
    if not pdf_tokens or not mineru_tokens:
        return bool(pdf_tokens) and not mineru_tokens

    pdf_counts = Counter(pdf_tokens)
    mineru_counts = Counter(mineru_tokens)
    matched = sum((pdf_counts & mineru_counts).values())
    token_recall = matched / len(pdf_tokens)
    missing = len(pdf_tokens) - matched
    return (
        (len(pdf_text) >= 500 or len(pdf_tokens) >= 100)
        and missing >= max(20, int(len(pdf_tokens) * 0.05))
        and token_recall < 0.92
    )


def _pdf_table_captions(page_texts: list[str]) -> dict[int, list[str]]:
    captions = {}
    label_only = re.compile(rf"^{TABLE_PREFIX}{HSPACE}*\d+[.:]?$", re.IGNORECASE)
    for page, text in enumerate(page_texts):
        lines = [line.strip() for line in text.splitlines()]
        matches = []
        for index, line in enumerate(lines):
            if not TABLE_RE.match(line):
                continue
            following = [item for item in lines[index + 1:] if item][:2]
            if label_only.match(line) and following:
                title = following[0]
                if len(following) > 1 and not title.endswith((".", ":")) and following[1][:1].islower():
                    title = f"{title} {following[1]}"
                matches.append(f"{line} {title}")
            else:
                matches.append(line)
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
    return [*(blk.get(key) or []), *(blk.get("image_footnote") or [])]


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
        suffix = []
        for item in [item.strip() for item in reversed(lines[:index]) if item.strip()][:3]:
            if item.startswith("<table"):
                continue
            if TABLE_RE.match(item):
                captions[line.strip()] = " ".join([item, *reversed(suffix)])
                break
            suffix.append(item)
    return captions


def _table_number(value: str) -> int | None:
    match = TABLE_RE.match(value.strip())
    return int(re.search(r"\d+", match.group()).group()) if match else None


def _table_caption(captions: list[str], body: str, markdown_caption: str = "") -> str:
    caption = " ".join(captions).strip()
    if TABLE_RE.match(caption):
        return caption
    if markdown_caption:
        return markdown_caption
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
    pdf_page_texts = _pdf_page_texts(pdf_path)
    pdf_captions = _pdf_captions(pdf_page_texts)
    pdf_caption_by_page = {
        page: {figure: caption for figure, caption in captions}
        for page, captions in pdf_captions.items()
    }
    pdf_table_captions = _pdf_table_captions(pdf_page_texts)

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
        figures = _caption_figures(txt)
        if len(figures) == 1 and figures[0] in pdf_caption_by_page.get(page_idx, {}):
            txt = _more_complete_caption(txt, pdf_caption_by_page[page_idx][figures[0]])
        if txt:
            page_blocks = pages_map.setdefault(page, [])
            if txt not in page_blocks:
                page_blocks.append(txt)
    for page, pdf_text in enumerate(pdf_page_texts, 1):
        mineru_text = "\n".join(pages_map.get(page, []))
        pdf_words = pdf_text.split()
        pdf_tokens = re.findall(r"[a-z0-9]+", pdf_text.casefold())
        mineru_tokens = set(re.findall(r"[a-z0-9]+", mineru_text.casefold()))
        pdf_token_set = set(pdf_tokens)
        token_recall = len(pdf_token_set & mineru_tokens) / len(pdf_token_set) if pdf_token_set else 1
        leading_tokens = set(pdf_tokens[:len(pdf_tokens) // 2])
        leading_recall = len(leading_tokens & mineru_tokens) / len(leading_tokens) if leading_tokens else 1
        if _should_use_pdf_text(pdf_text, mineru_text) or (
            (len(pdf_words) >= 50 or len(pdf_tokens) >= 100 or len(pdf_text) >= 500)
            and (
                len(mineru_text.split()) < len(pdf_words) * 0.6
                or token_recall < 0.5
                or (len(pdf_tokens) >= 300 and leading_recall < 0.5)
            )
        ):
            pages_map[page] = [pdf_text.strip()]
    pages = [{"page": p, "text": "\n".join(pages_map[p])} for p in sorted(pages_map)]
    (out_dir / "fulltext.json").write_text(
        json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")

    # (2) 表格(MinerU 结构识别; 表格即图片也在此)
    tables = []
    table_index_by_page: dict[int, int] = {}
    for blk in content:
        if blk.get("type") == "table":
            captions = blk.get("table_caption") or []
            figure_captions = [caption for caption in captions if _caption_figures(caption)]
            table_captions = [caption for caption in captions if TABLE_RE.match(str(caption).strip())]
            if figure_captions and not table_captions:
                continue
            html = (blk.get("table_body") or "").strip()
            if html:
                page_idx = int(blk.get("page_idx", 0))
                table_index = table_index_by_page.get(page_idx, 0)
                pdf_page_captions = pdf_table_captions.get(page_idx, [])
                pdf_caption = pdf_page_captions[table_index] if table_index < len(pdf_page_captions) else ""
                table_index_by_page[page_idx] = table_index + 1
                table = {
                    "page": page_idx + 1,
                    "caption": _table_caption(
                        table_captions or captions,
                        html,
                        pdf_caption or markdown_table_captions.get(html, ""),
                    ),
                    "html": html,
                }
                table_number = _table_number(table["caption"])
                if (
                    table_number is not None
                    and tables
                    and tables[-1]["page"] == table["page"]
                    and _table_number(tables[-1]["caption"]) == table_number
                ):
                    tables[-1]["html"] += "\n" + html
                else:
                    tables.append(table)
    (out_dir / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")

    # (3) 图片 + 图注：只采用图片自身的单一明确图注，或同页空间邻近关系。
    img_blocks = [
        block for block in content
        if block.get("type") in ("image", "chart")
        or (
            block.get("type") == "table"
            and any(_caption_figures(caption) for caption in _visual_captions(block))
            and not any(TABLE_RE.match(_plain_caption(caption)) for caption in _visual_captions(block))
        )
    ]

    cap_by_fig: dict[int, str] = {}
    figs_by_page: dict[int, set[int]] = {}
    figs_by_block: dict[int, set[int]] = {}
    caption_y_by_fig: dict[int, list[float]] = {}
    for block in img_blocks:
        page = int(block.get("page_idx", 0))
        captions = list(_visual_captions(block))
        captions.extend(markdown_captions.get(Path(block.get("img_path") or "").name, []))
        for caption in captions:
            for figure in _caption_figures(caption):
                cap_by_fig[figure] = _more_complete_caption(
                    cap_by_fig.get(figure, ""), str(caption).strip()
                )
                figs_by_page.setdefault(page, set()).add(figure)
                figs_by_block.setdefault(id(block), set()).add(figure)

    for block in content:
        if block.get("type") not in ("text", "header"):
            continue
        text = _plain_caption(block.get("text") or "")
        bbox = block.get("bbox") or []
        for match in CAP_FIG_RE.finditer(text):
            figure = _figure_number(match)
            page = int(block.get("page_idx", 0))
            cap_by_fig[figure] = _more_complete_caption(
                cap_by_fig.get(figure, ""), text[match.start():match.start() + 300].strip()
            )
            figs_by_page.setdefault(page, set()).add(figure)
            if len(bbox) >= 2:
                caption_y_by_fig.setdefault(figure, []).append(float(bbox[1]))

    visual_pages = {int(block.get("page_idx", 0)) for block in img_blocks}
    for page, captions in pdf_captions.items():
        if page not in visual_pages:
            continue
        for figure, caption in captions:
            cap_by_fig[figure] = _more_complete_caption(cap_by_fig.get(figure, ""), caption)
            figs_by_page.setdefault(page, set()).add(figure)

    kept = 0
    path_by_block: dict[int, str] = {}
    for block in img_blocks:
        src = auto / (block.get("img_path") or "")
        if not src.is_file():
            continue
        ext = src.suffix.lstrip(".") or "jpg"
        filename = f"p{int(block.get('page_idx', 0)) + 1:02d}_img{kept:03d}.{ext}"
        shutil.copyfile(src, images_dir / filename)
        path_by_block[id(block)] = (images_dir / filename).relative_to(rel_base).as_posix()
        kept += 1

    blocks_by_fig: dict[int, list[dict]] = {}
    assigned_blocks: set[int] = set()
    for block in img_blocks:
        figures = figs_by_block.get(id(block), set())
        if len(figures) == 1 and id(block) in path_by_block:
            figure = next(iter(figures))
            blocks_by_fig.setdefault(figure, []).append(block)
            assigned_blocks.add(id(block))

    for page in visual_pages:
        page_blocks = sorted(
            (
                block for block in img_blocks
                if int(block.get("page_idx", 0)) == page
                and id(block) in path_by_block
                and len(block.get("bbox") or []) >= 4
            ),
            key=lambda block: (block["bbox"][1], block["bbox"][0]),
        )
        rows: list[list[dict]] = []
        for block in page_blocks:
            if rows and float(block["bbox"][1]) < max(float(item["bbox"][3]) for item in rows[-1]):
                rows[-1].append(block)
            else:
                rows.append([block])
        for block in page_blocks:
            caption_figures = [_caption_figures(caption) for caption in _visual_captions(block)]
            if not caption_figures or any(len(figures) != 1 for figures in caption_figures):
                continue
            figures = [items[0] for items in caption_figures]
            row = next((items for items in rows if any(id(item) == id(block) for item in items)), [])
            if len(row) != len(figures) or len(set(figures)) != len(figures):
                continue
            pairs = list(zip(figures, row))
            if any(
                figure in blocks_by_fig
                and not any(id(existing) == id(item) for existing in blocks_by_fig[figure])
                for figure, item in pairs
            ):
                continue
            for figure, item in pairs:
                blocks_by_fig.setdefault(figure, [item])
                assigned_blocks.add(id(item))

        ambiguous_figures = set()
        for block in img_blocks:
            figures = figs_by_block.get(id(block), set())
            if int(block.get("page_idx", 0)) == page and len(figures) > 1:
                ambiguous_figures.update(figures)
        if not ambiguous_figures:
            continue
        blocks = sorted(
            (
                block for block in img_blocks
                if int(block.get("page_idx", 0)) == page and id(block) in path_by_block
            ),
            key=lambda block: ((block.get("bbox") or [0, 0])[1], (block.get("bbox") or [0, 0])[0]),
        )
        counts = [(figure, _caption_panel_count(cap_by_fig[figure])) for figure in sorted(ambiguous_figures)]
        if any(figure in blocks_by_fig for figure in ambiguous_figures) or sum(count for _, count in counts) != len(blocks):
            continue
        start = 0
        for figure, count in counts:
            blocks_by_fig[figure] = blocks[start:start + count]
            assigned_blocks.update(id(block) for block in blocks_by_fig[figure])
            start += count

    for page in visual_pages:
        if any(figure not in blocks_by_fig for figure in figs_by_page.get(page, set())):
            continue
        pending = []
        for block in (
            item for item in img_blocks
            if int(item.get("page_idx", 0)) == page and id(item) in path_by_block
        ):
            if id(block) not in assigned_blocks:
                pending.append(block)
                continue
            figures = [
                figure for figure, figure_blocks in blocks_by_fig.items()
                if any(id(block) == id(item) for item in figure_blocks)
            ]
            if len(figures) == 1:
                group = [block]
                matched = []
                for candidate in reversed(pending):
                    bbox = candidate.get("bbox") or []
                    if len(bbox) < 4:
                        break
                    gaps = []
                    for item in group:
                        item_bbox = item.get("bbox") or []
                        if len(item_bbox) < 4:
                            continue
                        dx = max(0, max(bbox[0], item_bbox[0]) - min(bbox[2], item_bbox[2]))
                        dy = max(0, max(bbox[1], item_bbox[1]) - min(bbox[3], item_bbox[3]))
                        gaps.append(max(dx, dy))
                    if not gaps or min(gaps) > 80:
                        break
                    matched.append(candidate)
                    group.append(candidate)
                if matched:
                    figure = figures[0]
                    blocks_by_fig[figure] = [*reversed(matched), *blocks_by_fig[figure]]
                    assigned_blocks.update(id(item) for item in matched)
            pending = []

    for block in img_blocks:
        block_id = id(block)
        bbox = block.get("bbox") or []
        if block_id in assigned_blocks or block_id not in path_by_block or len(bbox) < 4:
            continue
        page = int(block.get("page_idx", 0))
        candidates = []
        for figure in figs_by_page.get(page, set()):
            for caption_y in caption_y_by_fig.get(figure, []):
                distance = caption_y - float(bbox[3])
                score = distance if distance >= 0 else abs(distance) + 1_000_000
                candidates.append((score, figure))
        candidates.sort()
        if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
            continue
        figure = candidates[0][1]
        blocks_by_fig.setdefault(figure, []).append(block)
        assigned_blocks.add(block_id)

    for page in visual_pages:
        blocks = sorted(
            (
                block for block in img_blocks
                if int(block.get("page_idx", 0)) == page
                and id(block) in path_by_block
                and len(block.get("bbox") or []) >= 4
            ),
            key=lambda block: (block["bbox"][1], block["bbox"][0]),
        )
        rows: list[list[dict]] = []
        for block in blocks:
            if rows and float(block["bbox"][1]) < max(float(item["bbox"][3]) for item in rows[-1]):
                rows[-1].append(block)
            else:
                rows.append([block])

        page_figures = sorted(figs_by_page.get(page, set()))
        anchors = [
            block for figure in page_figures for block in blocks_by_fig.get(figure, [])
            if int(block.get("page_idx", 0)) == page
        ]
        if len(rows) == len(page_figures) and anchors:
            consistent = all(
                all(any(id(block) == id(row_block) for row_block in row) for block in blocks_by_fig.get(figure, []))
                for figure, row in zip(page_figures, rows)
            )
            if consistent:
                for figure, row in zip(page_figures, rows):
                    blocks_by_fig[figure] = row
                    assigned_blocks.update(id(block) for block in row)

        if any(figure not in blocks_by_fig for figure in page_figures):
            continue
        for row in rows:
            figures = {
                figure for figure, figure_blocks in blocks_by_fig.items()
                if any(id(block) == id(row_block) for block in figure_blocks for row_block in row)
            }
            unassigned = [block for block in row if id(block) not in assigned_blocks]
            if len(figures) == 1 and unassigned:
                figure = next(iter(figures))
                blocks_by_fig[figure].extend(unassigned)
                assigned_blocks.update(id(block) for block in unassigned)

        unassigned = [block for block in blocks if id(block) not in assigned_blocks]
        pdf_page_figures = {figure for figure, _ in pdf_captions.get(page, [])}
        if len(pdf_page_figures) == 1 and pdf_page_figures == figs_by_page.get(page, set()):
            figure = next(iter(pdf_page_figures))
            blocks_by_fig[figure].extend(unassigned)
            assigned_blocks.update(id(block) for block in unassigned)
            continue

        assigned_on_page = [block for block in blocks if id(block) in assigned_blocks]
        for block in unassigned:
            bbox = block["bbox"]
            candidates = []
            for assigned in assigned_on_page:
                assigned_bbox = assigned["bbox"]
                width = min(bbox[2] - bbox[0], assigned_bbox[2] - assigned_bbox[0])
                height = min(bbox[3] - bbox[1], assigned_bbox[3] - assigned_bbox[1])
                x_overlap = max(0, min(bbox[2], assigned_bbox[2]) - max(bbox[0], assigned_bbox[0]))
                y_overlap = max(0, min(bbox[3], assigned_bbox[3]) - max(bbox[1], assigned_bbox[1]))
                vertical_gap = max(0, max(bbox[1], assigned_bbox[1]) - min(bbox[3], assigned_bbox[3]))
                horizontal_gap = max(0, max(bbox[0], assigned_bbox[0]) - min(bbox[2], assigned_bbox[2]))
                vertical_neighbor = x_overlap / width >= 0.8 and vertical_gap <= height * 0.25
                horizontal_neighbor = y_overlap / height >= 0.8 and horizontal_gap <= width * 0.25
                if not vertical_neighbor and not horizontal_neighbor:
                    continue
                gap = min(
                    vertical_gap if vertical_neighbor else float("inf"),
                    horizontal_gap if horizontal_neighbor else float("inf"),
                )
                figures = [
                    figure for figure, figure_blocks in blocks_by_fig.items()
                    if any(id(assigned) == id(item) for item in figure_blocks)
                ]
                if len(figures) == 1:
                    candidates.append((gap, figures[0]))
            best_by_figure = {}
            for gap, figure in candidates:
                best_by_figure[figure] = min(gap, best_by_figure.get(figure, gap))
            candidates = sorted((gap, figure) for figure, gap in best_by_figure.items())
            if candidates and (len(candidates) == 1 or candidates[0][0] != candidates[1][0]):
                figure = candidates[0][1]
                blocks_by_fig[figure].append(block)
                assigned_blocks.add(id(block))

    for page in visual_pages:
        figures = sorted(figure for figure in figs_by_page.get(page, set()) if figure not in blocks_by_fig)
        blocks = sorted(
            (
                block for block in img_blocks
                if int(block.get("page_idx", 0)) == page
                and id(block) in path_by_block
                and id(block) not in assigned_blocks
                and len(block.get("bbox") or []) >= 4
            ),
            key=lambda block: (block["bbox"][1], block["bbox"][0]),
        )
        rows: list[list[dict]] = []
        for block in blocks:
            if rows and float(block["bbox"][1]) < max(float(item["bbox"][3]) for item in rows[-1]):
                rows[-1].append(block)
            else:
                rows.append([block])
        if len(rows) == len(figures):
            for figure, row in zip(figures, rows):
                blocks_by_fig[figure] = row
                assigned_blocks.update(id(block) for block in row)

    images_map = {
        f"Figure {figure}": {
            "images": [path_by_block[id(block)] for block in blocks],
            "caption": cap_by_fig.get(figure, ""),
        }
        for figure, blocks in sorted(blocks_by_fig.items())
    }
    unmapped_images = [
        path_by_block[id(block)]
        for block in img_blocks
        if id(block) in path_by_block and id(block) not in assigned_blocks
    ]

    # 清理 MinerU 中间产物(已提取所需内容)
    shutil.rmtree(out_dir / "_mineru", ignore_errors=True)

    return {
        "n_pages": len(pages),
        "n_images_kept": kept,
        "n_tables": len(tables),
        "images_map": images_map,
        "unmapped_images": unmapped_images,
    }
