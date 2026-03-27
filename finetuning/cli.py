from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from finetuning.dataset import build_flux_edit_zip, iter_generated_interiors_pairs
from finetuning.fal_flux_trainer import (
    download_file,
    poll_until_complete,
    submit_edit_trainer,
    upload_training_zip,
)


def _cmd_prepare(args: argparse.Namespace) -> int:
    pairs = list(
        iter_generated_interiors_pairs(
            args.generated_interiors,
            caption=args.caption,
            start_filename=args.start_filename,
            end_filename=args.end_filename,
        )
    )
    if not pairs:
        raise SystemExit("No valid pairs found. Check paths and filenames.")

    zip_path = build_flux_edit_zip(pairs, output_zip=args.output_zip)
    print(str(zip_path))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    zip_url = upload_training_zip(Path(args.zip_path))

    network_weights = None
    if args.network_weights is not None:
        if args.network_weights.startswith("http://") or args.network_weights.startswith("https://"):
            network_weights = args.network_weights
        else:
            from finetuning.fal_flux_trainer import upload_file

            network_weights = upload_file(Path(args.network_weights))

    req = submit_edit_trainer(
        image_data_url=zip_url,
        steps=args.steps,
        learning_rate=args.learning_rate,
        default_caption=args.default_caption,
        network_weights=network_weights,
        output_lora_format=args.output_lora_format,
    )
    print(req.request_id)
    return 0


def _cmd_wait(args: argparse.Namespace) -> int:
    print(f"Waiting for request_id={args.request_id}", flush=True)
    try:
        result = poll_until_complete(
            request_id_to_request(args.request_id),
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except KeyboardInterrupt:
        print("Interrupted", flush=True)
        return 130

    out_dir = Path(args.output_dir)
    print(f"LoRA URL: {result.diffusers_lora_url}", flush=True)
    print(f"Config URL: {result.config_url}", flush=True)
    print(f"Downloading to: {out_dir}", flush=True)
    lora_path = download_file(result.diffusers_lora_url, out_dir / "diffusers_lora.safetensors")
    cfg_path = download_file(result.config_url, out_dir / "config.json")

    print(str(lora_path), flush=True)
    print(str(cfg_path), flush=True)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    import fal_client

    status = fal_client.status("fal-ai/flux-2-trainer/edit", args.request_id, with_logs=args.logs)

    status_str = None
    queue_pos = None
    logs = None

    if isinstance(status, dict):
        status_str = status.get("status")
        queue_pos = status.get("queue_position")
        logs = status.get("logs")
    else:
        status_str = getattr(status, "status", None)
        queue_pos = getattr(status, "queue_position", None)
        logs = getattr(status, "logs", None)

        if status_str is None and hasattr(status, "__dict__"):
            d = status.__dict__
            status_str = d.get("status") or d.get("state")
            queue_pos = queue_pos if queue_pos is not None else d.get("queue_position")
            logs = logs if logs is not None else d.get("logs")

    if status_str is None:
        print(repr(status))
        return 0

    if queue_pos is not None:
        print(f"{status_str} queue_position={queue_pos}")
    else:
        print(f"{status_str}")

    if args.logs and logs:
        print(logs)

    return 0


def _cmd_cancel(args: argparse.Namespace) -> int:
    import fal_client

    fal_client.cancel("fal-ai/flux-2-trainer/edit", args.request_id)
    print(f"Cancel request sent for request_id={args.request_id}", flush=True)
    return 0


def request_id_to_request(request_id: str):
    from finetuning.fal_flux_trainer import TrainRequest

    return TrainRequest(request_id=request_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="finetuning")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Build training zip from generated_interiors")
    p_prep.add_argument(
        "--generated-interiors",
        default=str(Path("dataset_generation") / "generated_interiors"),
        help="Path to dataset_generation/generated_interiors",
    )
    p_prep.add_argument("--output-zip", default=str(Path("finetuning") / "artifacts" / "train.zip"))
    p_prep.add_argument("--caption", default=None)
    p_prep.add_argument("--start-filename", default="empty_plan.png")
    p_prep.add_argument("--end-filename", default="2D_plan.png")
    p_prep.set_defaults(func=_cmd_prepare)

    p_train = sub.add_parser("train", help="Upload zip and submit flux edit trainer")
    p_train.add_argument("zip_path", help="Path to the zip created by prepare")
    p_train.add_argument("--steps", type=int, default=1000)
    p_train.add_argument("--learning-rate", type=float, default=5e-5)
    p_train.add_argument("--default-caption", default=None)
    p_train.add_argument(
        "--network-weights",
        default=None,
        help="Optional URL or local path to an existing LoRA (.safetensors) to continue training from",
    )
    p_train.add_argument("--output-lora-format", choices=["fal", "comfy"], default="fal")
    p_train.set_defaults(func=_cmd_train)

    p_wait = sub.add_parser("wait", help="Poll status, then download artifacts")
    p_wait.add_argument("request_id")
    p_wait.add_argument("--output-dir", default=str(Path("finetuning") / "runs"))
    p_wait.add_argument("--poll-seconds", type=float, default=10.0)
    p_wait.add_argument("--timeout-seconds", type=float, default=None)
    p_wait.set_defaults(func=_cmd_wait)

    p_status = sub.add_parser("status", help="Check queue status for a request_id")
    p_status.add_argument("request_id")
    p_status.add_argument("--logs", action="store_true", help="Include logs in the status response")
    p_status.set_defaults(func=_cmd_status)

    p_cancel = sub.add_parser("cancel", help="Cancel a queued or in-progress training request")
    p_cancel.add_argument("request_id")
    p_cancel.set_defaults(func=_cmd_cancel)

    return p


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
