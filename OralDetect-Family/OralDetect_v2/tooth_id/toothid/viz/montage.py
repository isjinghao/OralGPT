"""Contact sheet: many overlays on one image, each cell letterboxed and captioned with its file stem."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .overlay import _font


def montage(paths, out, cols: int = 4, cell=(960, 720), title: str | None = None, strip: str = "") -> Path:
    paths = [Path(p) for p in paths]
    rows, (cw, ch), cap, pad = -(-len(paths) // cols), cell, 34, 8
    head = 60 if title else 0
    sheet = Image.new("RGB", (cols * (cw + pad) + pad, head + rows * (ch + cap + pad) + pad), "#111111")
    dr = ImageDraw.Draw(sheet)
    if title:
        dr.text((pad, 10), title, fill="white", font=_font(36))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        s = min(cw / im.width, ch / im.height)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        x0 = pad + (i % cols) * (cw + pad)
        y0 = head + pad + (i // cols) * (ch + cap + pad)
        sheet.paste(im, (x0 + (cw - im.width) // 2, y0 + cap + (ch - im.height) // 2))
        dr.text((x0 + 4, y0 + 4), p.stem.replace(strip, "")[:60], fill="white", font=_font(22))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out
