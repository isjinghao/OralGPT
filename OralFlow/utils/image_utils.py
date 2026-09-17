from __future__ import annotations

import base64
import mimetypes
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


def image_data_url(path: Path) -> str | None:
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def resolve_image_path(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else root / path


def crop_white_border(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, background).convert("L")
    mask = diff.point(lambda value: 255 if value > 14 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return rgb
    left, top, right, bottom = bbox
    margin = 8
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(rgb.width, right + margin)
    bottom = min(rgb.height, bottom + margin)
    if (right - left) * (bottom - top) < 0.25 * rgb.width * rgb.height:
        return rgb
    return rgb.crop((left, top, right, bottom))


def grayscale_image_data_url(path: Path, max_side: int = 768) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = crop_white_border(image)
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image = ImageOps.grayscale(image).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True, progressive=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_data_urls(root: Path, image_paths: list[str]) -> list[str]:
    paths = [resolve_image_path(root, image_path) for image_path in image_paths]
    return [url for path in paths if path.is_file() and (url := image_data_url(path))]


def grayscale_image_data_urls(root: Path, image_paths: list[str]) -> list[str]:
    paths = [resolve_image_path(root, image_path) for image_path in image_paths]
    return [url for path in paths if path.is_file() and (url := grayscale_image_data_url(path))]
