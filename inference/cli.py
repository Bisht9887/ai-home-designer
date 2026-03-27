from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from inference.fal_inference import (
    edit_with_lora_flux2,
    image_to_image_base,
    text_to_image_base,
)
from inference.dataset_runner import ensure_output_dir, iter_inference_data_dir, read_prompt
from inference.io_utils import download_file


def _cmd_t2i(args: argparse.Namespace) -> int:
    images = text_to_image_base(
        prompt=args.prompt,
        num_images=args.num_images,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_format=args.output_format,
    )

    out_dir = Path(args.output_dir)
    for i, img in enumerate(images, start=1):
        download_file(img.url, out_dir / f"t2i_{i}.{args.output_format}")
    return 0


def _cmd_i2i(args: argparse.Namespace) -> int:
    images = image_to_image_base(
        image_path=Path(args.image),
        prompt=args.prompt,
        strength=args.strength,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_format=args.output_format,
    )

    out_dir = Path(args.output_dir)
    for i, img in enumerate(images, start=1):
        download_file(img.url, out_dir / f"i2i_{i}.{args.output_format}")
    return 0


def _cmd_i2i_lora(args: argparse.Namespace) -> int:
    images = edit_with_lora_flux2(
        image_path=Path(args.image),
        prompt=args.prompt,
        lora_path_or_url=args.lora,
        lora_scale=args.lora_scale,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_format=args.output_format,
        enable_prompt_expansion=args.enable_prompt_expansion,
        acceleration=args.acceleration,
    )

    out_dir = Path(args.output_dir)
    for i, img in enumerate(images, start=1):
        download_file(img.url, out_dir / f"i2i_lora_{i}.{args.output_format}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    base_out_dir = Path(args.output_dir)

    items = list(iter_inference_data_dir(data_dir, image_suffix=args.image_suffix))
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No valid items found in {data_dir}. Expected ROOT{args.image_suffix}.png + ROOT.txt")

    for item in items:
        prompt = read_prompt(item.prompt_path)
        out_dir = ensure_output_dir(base_out_dir, item.root)

        if args.lora is None:
            images = image_to_image_base(
                image_path=item.image_path,
                prompt=prompt,
                strength=args.strength,
                num_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                output_format=args.output_format,
            )
            for i, img in enumerate(images, start=1):
                download_file(img.url, out_dir / f"base_{i}.{args.output_format}")
        else:
            images = edit_with_lora_flux2(
                image_path=item.image_path,
                prompt=prompt,
                lora_path_or_url=args.lora,
                lora_scale=args.lora_scale,
                num_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                output_format=args.output_format,
                enable_prompt_expansion=args.enable_prompt_expansion,
                acceleration=args.acceleration,
            )
            for i, img in enumerate(images, start=1):
                download_file(img.url, out_dir / f"lora_{i}.{args.output_format}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_t2i = sub.add_parser("t2i", help="Text-to-image with base FLUX")
    p_t2i.add_argument("--prompt", required=True)
    p_t2i.add_argument("--output-dir", default=str(Path("inference") / "outputs"))
    p_t2i.add_argument("--num-images", type=int, default=1)
    p_t2i.add_argument("--image-size", default="landscape_4_3")
    p_t2i.add_argument("--num-inference-steps", type=int, default=28)
    p_t2i.add_argument("--guidance-scale", type=float, default=3.5)
    p_t2i.add_argument("--seed", type=int, default=None)
    p_t2i.add_argument("--output-format", choices=["png", "jpeg"], default="png")
    p_t2i.set_defaults(func=_cmd_t2i)

    p_i2i = sub.add_parser("i2i", help="Image-to-image with base FLUX")
    p_i2i.add_argument("--image", required=True, help="Local path to input image")
    p_i2i.add_argument("--prompt", required=True)
    p_i2i.add_argument("--output-dir", default=str(Path("inference") / "outputs"))
    p_i2i.add_argument("--strength", type=float, default=0.95)
    p_i2i.add_argument("--num-images", type=int, default=1)
    p_i2i.add_argument("--num-inference-steps", type=int, default=40)
    p_i2i.add_argument("--guidance-scale", type=float, default=3.5)
    p_i2i.add_argument("--seed", type=int, default=None)
    p_i2i.add_argument("--output-format", choices=["png", "jpeg"], default="png")
    p_i2i.set_defaults(func=_cmd_i2i)

    p_i2i_lora = sub.add_parser("i2i-lora", help="Image edit with FLUX.2 + LoRA adapter")
    p_i2i_lora.add_argument("--image", required=True, help="Local path to input image")
    p_i2i_lora.add_argument("--prompt", required=True)
    p_i2i_lora.add_argument("--lora", required=True, help="Local path or URL to LoRA weights (.safetensors)")
    p_i2i_lora.add_argument("--lora-scale", type=float, default=1.0)
    p_i2i_lora.add_argument("--output-dir", default=str(Path("inference") / "outputs"))
    p_i2i_lora.add_argument("--num-images", type=int, default=1)
    p_i2i_lora.add_argument("--num-inference-steps", type=int, default=28)
    p_i2i_lora.add_argument("--guidance-scale", type=float, default=2.5)
    p_i2i_lora.add_argument("--seed", type=int, default=None)
    p_i2i_lora.add_argument("--output-format", choices=["png", "jpeg", "webp"], default="png")
    p_i2i_lora.add_argument("--enable-prompt-expansion", action="store_true")
    p_i2i_lora.add_argument("--acceleration", choices=["none", "regular", "high"], default="regular")
    p_i2i_lora.set_defaults(func=_cmd_i2i_lora)

    p_ds = sub.add_parser("dataset", help="Run batch inference on inference/data (ROOT_start.png + ROOT.txt)")
    p_ds.add_argument("--data-dir", default=str(Path("inference") / "data"))
    p_ds.add_argument("--output-dir", default=str(Path("inference") / "outputs"))
    p_ds.add_argument("--image-suffix", default="_start", help="Suffix of image files before extension")
    p_ds.add_argument("--limit", type=int, default=None)
    p_ds.add_argument("--num-images", type=int, default=1)
    p_ds.add_argument("--num-inference-steps", type=int, default=40)
    p_ds.add_argument("--guidance-scale", type=float, default=3.5)
    p_ds.add_argument("--seed", type=int, default=None)
    p_ds.add_argument("--output-format", choices=["png", "jpeg", "webp"], default="png")
    p_ds.add_argument("--strength", type=float, default=0.95, help="Only used for base image-to-image")
    p_ds.add_argument("--lora", default=None, help="If set, runs LoRA edit mode using this local path or URL")
    p_ds.add_argument("--lora-scale", type=float, default=1.0)
    p_ds.add_argument("--enable-prompt-expansion", action="store_true")
    p_ds.add_argument("--acceleration", choices=["none", "regular", "high"], default="regular")
    p_ds.set_defaults(func=_cmd_dataset)

    return p


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
