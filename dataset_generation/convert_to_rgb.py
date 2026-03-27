#!/usr/bin/env python
"""Convert all RGBA images in a directory to RGB (white background).

Creates a new directory under dataset_generation/data/ with all files copied,
converting any RGBA/LA/PA images to RGB in the process.

Usage:
    python dataset_generation/convert_to_rgb.py <source_dir> [--output-name NAME]

Examples:
    python dataset_generation/convert_to_rgb.py dataset_generation/data/my_dataset
    python dataset_generation/convert_to_rgb.py /path/to/images --output-name dataset_rgb
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def convert_directory(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in source_dir.rglob("*") if f.is_file())

    converted = 0
    copied = 0
    skipped = 0

    for src_file in files:
        rel = src_file.relative_to(source_dir)
        dst_file = output_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.suffix.lower() in IMAGE_SUFFIXES:
            img = Image.open(src_file)
            if img.mode != "RGB":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode in ("RGBA", "LA", "PA"):
                    bg.paste(img, mask=img.split()[-1])
                else:
                    bg.paste(img.convert("RGB"))
                img = bg
                dst_png = dst_file.with_suffix(".png")
                img.save(dst_png, format="PNG")
                print(f"  converted  {rel}  ({_mode(src_file)} → RGB)")
                converted += 1
            else:
                shutil.copy2(src_file, dst_file)
                print(f"  ok         {rel}")
                copied += 1
        else:
            shutil.copy2(src_file, dst_file)
            skipped += 1

    print(f"\nDone → {output_dir}")
    print(f"  {converted} converted (RGBA→RGB), {copied} already RGB, {skipped} non-image files copied")


def _mode(path: Path) -> str:
    try:
        return Image.open(path).mode
    except Exception:
        return "?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RGBA images to RGB in a directory.")
    parser.add_argument("source_dir", help="Source directory to scan")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Name for the output directory under dataset_generation/data/ (default: <source_name>_rgb)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Full output directory path (overrides --output-name)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        return 1
    if not source_dir.is_dir():
        print(f"Error: not a directory: {source_dir}", file=sys.stderr)
        return 1

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_name = args.output_name or f"{source_dir.name}_rgb"
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "dataset_generation" / "data" / output_name

    if output_dir.exists():
        print(f"Error: output directory already exists: {output_dir}", file=sys.stderr)
        print("Remove it first or use --output-name to specify a different name.", file=sys.stderr)
        return 1

    print(f"Source : {source_dir}")
    print(f"Output : {output_dir}")
    print()

    convert_directory(source_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
