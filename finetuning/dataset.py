from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image


@dataclass(frozen=True)
class EditPair:
    root: str
    start_path: Path
    end_path: Path
    caption: Optional[str] = None


def iter_generated_interiors_pairs(
    generated_interiors_dir: os.PathLike,
    *,
    start_filename: str = "empty_plan.png",
    end_filename: str = "2D_plan.png",
    caption: Optional[str] = None,
) -> Iterable[EditPair]:
    base = Path(generated_interiors_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        start_path = child / start_filename
        end_path = child / end_filename
        if not start_path.exists() or not end_path.exists():
            continue

        yield EditPair(root=child.name, start_path=start_path, end_path=end_path, caption=caption)


def build_flux_edit_zip(
    pairs: Iterable[EditPair],
    *,
    output_zip: os.PathLike,
) -> Path:
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pair in pairs:
            zf.writestr(f"{pair.root}_start.png", _to_rgb_png_bytes(pair.start_path))
            zf.writestr(f"{pair.root}_end.png", _to_rgb_png_bytes(pair.end_path))

            if pair.caption is not None:
                zf.writestr(f"{pair.root}.txt", pair.caption)

    return output_zip


def _to_rgb_png_bytes(image_path: Path) -> bytes:
    img = Image.open(image_path)
    if img.mode != "RGB":
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode in ("RGBA", "LA", "PA"):
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img.convert("RGB"))
        img = background
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
