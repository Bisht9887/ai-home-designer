"""
Run Part 1 of the training pipeline: OpenCV preprocessing on a dataset.

Reads layout images from each variation folder, runs opencv_preprocess,
and writes processed images (e.g. layout_processed.png) into the same folder
or to an output directory.

Usage:
    python run_preprocess.py <dataset_root> [--output-dir DIR] [--limit N]

Example:
    python run_preprocess.py "C:\\ai_interior_design\\Openrouter\\D01_120_variations"
    python run_preprocess.py "C:\\ai_interior_design\\Openrouter\\D01_120_variations" --limit 5
"""

import argparse
import sys
from pathlib import Path

# Allow importing from same package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from opencv_preprocess import (
    find_layout_image,
    preprocess_layout,
    save_processed,
    DEFAULT_TARGET_SIZE,
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess layout images with OpenCV (pipeline step 1).")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root folder containing variation subfolders (e.g. D01_120_variations).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write processed images here (with variation subfolder names). Else write layout_processed.png into each variation folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of variation folders to process (default: all).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=list(DEFAULT_TARGET_SIZE),
        metavar=("W", "H"),
        help="Target width and height for resize (default: 512 512).",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Skip edge detection; output grayscale (or denoised color) only.",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    target_size = tuple(args.target_size)
    folders = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        folders = folders[: args.limit]

    done = 0
    skipped = 0
    for folder in folders:
        layout_path = find_layout_image(folder)
        if layout_path is None:
            skipped += 1
            continue
        try:
            processed = preprocess_layout(
                layout_path,
                target_size=target_size,
                to_gray=True,
                denoise_first=True,
                edge_detection=not args.no_edges,
                keep_aspect=False,
            )
            if args.output_dir is not None:
                out_dir = args.output_dir / folder.name
                out_path = out_dir / "layout_processed.png"
            else:
                out_path = folder / "layout_processed.png"
            save_processed(out_path, processed)
            done += 1
        except Exception as e:
            print(f"Error processing {folder.name}: {e}", file=sys.stderr)

    print(f"Done: {done} processed, {skipped} skipped (no layout image).")


if __name__ == "__main__":
    main()
