from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class InferenceItem:
    root: str
    image_path: Path
    prompt_path: Path


def iter_inference_data_dir(
    data_dir: Path,
    *,
    image_suffix: str = "_start",
) -> Iterator[InferenceItem]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    exts = (".png", ".jpg", ".jpeg", ".webp")
    for img in sorted(data_dir.iterdir()):
        if not img.is_file():
            continue
        if img.suffix.lower() not in exts:
            continue
        if not img.stem.endswith(image_suffix):
            continue

        root = img.stem[: -len(image_suffix)]
        prompt_path = data_dir / f"{root}.txt"
        if not prompt_path.exists():
            continue

        yield InferenceItem(root=root, image_path=img, prompt_path=prompt_path)


def read_prompt(prompt_path: Path) -> str:
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return text


def ensure_output_dir(base_output_dir: Path, root: str) -> Path:
    out_dir = base_output_dir / root
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
