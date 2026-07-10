from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_fulltext(pages: list[dict]) -> str:
    """把按页存的 fulltext.json 拼回带分页标记的整段文本(供 LLM 抽取/校验/图注解析使用)。"""
    return "".join(f"\n===== PAGE {p['page']} =====\n{p['text']}" for p in pages)


def read_fulltext(raw_dir: Path) -> str:
    """读取 raw/fulltext.json 并拼成整段文本。"""
    data = json.loads((raw_dir / "fulltext.json").read_text(encoding="utf-8"))
    return build_fulltext(data["pages"])


def extract_pdf(
    pdf_path: Path,
    out_dir: Path,
    images_dir: Path,
    rel_base: Path | None = None,
    min_side: int = 180,
    min_area: int = 90_000,
) -> dict:
    # 抽取PDF内容
    import fitz, pdfplumber

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    rel_base = rel_base or images_dir

    def _rel(p: Path) -> str:
        try:
            return p.relative_to(rel_base).as_posix()
        except ValueError:
            return p.as_posix()

    # (1) 文本
    pages: list[dict] = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]
    (out_dir / "fulltext.json").write_text(
        json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_pages = len(pages)

    # (2) 图片
    seen_hashes: dict[str, str] = {}
    kept_images: list[str] = []
    for i, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            width = base.get("width", 0)
            height = base.get("height", 0)
            data = base["image"]
            ext = base.get("ext", "png")
            digest = hashlib.md5(data).hexdigest()
            significant = min(width, height) >= min_side and width * height >= min_area
            if not significant or digest in seen_hashes:
                continue
            kept = len(kept_images)
            fname = f"p{i + 1:02d}_img{kept:03d}.{ext if ext in ('png', 'jpg', 'jpeg') else 'png'}"
            (images_dir / fname).write_bytes(data)
            seen_hashes[digest] = fname
            kept_images.append(_rel(images_dir / fname))

    doc.close()

    # (3) 表格
    tables: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            for j, table in enumerate(page.extract_tables() or []):
                tables.append({"page": i + 1, "index": j, "rows": table})
    (out_dir / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "n_pages": n_pages,
        "n_images_kept": len(kept_images),
        "kept_images": kept_images,
        "n_tables": len(tables),
    }
