"""
Run evaluation metrics on generated vs ground-truth image pairs.

Usage:
    python -m evaluation.run_eval \
      --generated-dir inference/outputs \
      --ground-truth-dir inference/data \
      --generated-filename base_1.png

Expects:
- Generated: <generated-dir>/<ROOT>/<generated-filename>  (e.g. base_1.png or lora_1.png)
- Ground truth: <ground-truth-dir>/<ROOT>_end.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.metrics import compute_metrics


def iter_pairs(
    generated_dir: Path,
    ground_truth_dir: Path,
    generated_filename: str = "base_1.png",
) -> list[tuple[Path, Path, str]]:
    """Yield (generated_path, gt_path, root) for nested layout: ROOT/sub/base_1.png + ROOT_end.png."""
    pairs = []
    for sub in sorted(generated_dir.iterdir()):
        if not sub.is_dir():
            continue
        root = sub.name
        gen_path = sub / generated_filename
        gt_path = ground_truth_dir / f"{root}_end.png"

        if not gen_path.exists():
            continue
        if not gt_path.exists():
            continue

        pairs.append((gen_path, gt_path, root))
    return pairs


def iter_pairs_flat(
    flat_dir: Path,
    generated_suffix: str = "_lora",
) -> list[tuple[Path, Path, str]]:
    """Yield (generated_path, gt_path, root) for flat layout: ROOT_base.png, ROOT_end.png, ROOT_lora.png."""
    pairs = []
    for f in sorted(flat_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != ".png":
            continue
        stem = f.stem
        if not stem.endswith("_end"):
            continue
        root = stem[: -len("_end")]
        gen_path = flat_dir / f"{root}{generated_suffix}.png"
        gt_path = f

        if not gen_path.exists():
            continue

        pairs.append((gen_path, gt_path, root))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated images vs ground truth")
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("inference/outputs"),
        help="Directory containing ROOT/subdirs with generated images",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("inference/data"),
        help="Directory containing ROOT_end.png files",
    )
    parser.add_argument(
        "--generated-filename",
        default="base_1.png",
        help="Generated image filename (e.g. base_1.png or lora_1.png)",
    )
    parser.add_argument(
        "--flat-dir",
        type=Path,
        default=None,
        help="Flat mode: single dir with ROOT_base.png, ROOT_end.png, ROOT_lora.png. Compares _lora or _base vs _end.",
    )
    parser.add_argument(
        "--generated-suffix",
        default="_lora",
        choices=["_base", "_lora"],
        help="When using --flat-dir: compare _lora or _base against _end (ground truth)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save per-sample and aggregate metrics to JSON file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="With --flat-dir: run both _lora and _base, show %% improvement of LoRA over Base",
    )
    args = parser.parse_args()

    if args.flat_dir is not None:
        flat_dir = args.flat_dir.resolve()
        if not flat_dir.exists():
            print(f"Error: flat dir not found: {flat_dir}", file=sys.stderr)
            return 1
        if args.compare:
            pairs_lora = iter_pairs_flat(flat_dir, generated_suffix="_lora")
            pairs_base = iter_pairs_flat(flat_dir, generated_suffix="_base")
            roots_lora = {r for (_, _, r) in pairs_lora}
            roots_base = {r for (_, _, r) in pairs_base}
            common = roots_lora & roots_base
            if not common:
                print("Error: --compare requires both ROOT_lora.png and ROOT_base.png", file=sys.stderr)
                return 1
            pairs_lora = [(g, gt, r) for (g, gt, r) in pairs_lora if r in common]
            pairs_base = [(g, gt, r) for (g, gt, r) in pairs_base if r in common]
            pairs_lora.sort(key=lambda x: x[2])
            pairs_base.sort(key=lambda x: x[2])

            # Run compare flow: LoRA vs Base, both vs ground truth
            compare_results = []
            for (gl, gt, r), (gb, _, _) in zip(pairs_lora, pairs_base):
                try:
                    ml = compute_metrics(gl, gt)
                    mb = compute_metrics(gb, gt)
                    # LPIPS, MSE: lower better, improvement = (base - lora) / base * 100
                    # SSIM: higher better, improvement = (lora - base) / base * 100
                    lpips_imp = 100 * (mb["lpips"] - ml["lpips"]) / mb["lpips"] if mb["lpips"] else 0
                    ssim_imp = 100 * (ml["ssim"] - mb["ssim"]) / mb["ssim"] if mb["ssim"] else 0
                    mse_imp = 100 * (mb["mse"] - ml["mse"]) / mb["mse"] if mb["mse"] else 0
                    compare_results.append({
                        "root": r,
                        "lora": ml,
                        "base": mb,
                        "lpips_improvement_pct": lpips_imp,
                        "ssim_improvement_pct": ssim_imp,
                        "mse_improvement_pct": mse_imp,
                    })
                    print(f"{r}: LPIPS {ml['lpips']:.4f} vs {mb['lpips']:.4f} ({lpips_imp:+.1f}%)  "
                          f"SSIM {ml['ssim']:.4f} vs {mb['ssim']:.4f} ({ssim_imp:+.1f}%)  "
                          f"MSE {ml['mse']:.4f} vs {mb['mse']:.4f} ({mse_imp:+.1f}%)")
                except Exception as e:
                    print(f"{r}: ERROR {e}", file=sys.stderr)

            if not compare_results:
                return 1

            n = len(compare_results)
            lpips_imp_avg = sum(r["lpips_improvement_pct"] for r in compare_results) / n
            ssim_imp_avg = sum(r["ssim_improvement_pct"] for r in compare_results) / n
            mse_imp_avg = sum(r["mse_improvement_pct"] for r in compare_results) / n
            lora_agg = {
                "lpips": sum(r["lora"]["lpips"] for r in compare_results) / n,
                "ssim": sum(r["lora"]["ssim"] for r in compare_results) / n,
                "mse": sum(r["lora"]["mse"] for r in compare_results) / n,
            }
            base_agg = {
                "lpips": sum(r["base"]["lpips"] for r in compare_results) / n,
                "ssim": sum(r["base"]["ssim"] for r in compare_results) / n,
                "mse": sum(r["base"]["mse"] for r in compare_results) / n,
            }
            print("\n--- LoRA vs Base: % Improvement (positive = LoRA better) ---")
            print(f"LPIPS: {lpips_imp_avg:+.1f}% (lower is better, + = LoRA reduced LPIPS)")
            print(f"SSIM:  {ssim_imp_avg:+.1f}% (higher is better, + = LoRA increased SSIM)")
            print(f"MSE:   {mse_imp_avg:+.1f}% (lower is better, + = LoRA reduced MSE)")
            print(f"N:     {n}")

            if args.output_json:
                out = {
                    "lora_aggregate": lora_agg,
                    "base_aggregate": base_agg,
                    "improvement_pct": {"lpips": lpips_imp_avg, "ssim": ssim_imp_avg, "mse": mse_imp_avg},
                    "n_samples": n,
                    "per_sample": compare_results,
                }
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
                print(f"\nSaved to {args.output_json}")
            return 0
        else:
            pairs = iter_pairs_flat(flat_dir, generated_suffix=args.generated_suffix)
            if not pairs:
                print(
                    f"No valid pairs in flat mode. Need ROOT{args.generated_suffix}.png and ROOT_end.png in {flat_dir}",
                    file=sys.stderr,
                )
                return 1
    else:
        gen_dir = args.generated_dir.resolve()
        gt_dir = args.ground_truth_dir.resolve()
        if not gen_dir.exists():
            print(f"Error: generated dir not found: {gen_dir}", file=sys.stderr)
            return 1
        if not gt_dir.exists():
            print(f"Error: ground-truth dir not found: {gt_dir}", file=sys.stderr)
            return 1
        pairs = iter_pairs(gen_dir, gt_dir, args.generated_filename)
        if not pairs:
            print(
                f"No valid pairs found. Check {gen_dir}/ROOT/{args.generated_filename} and {gt_dir}/ROOT_end.png",
                file=sys.stderr,
            )
            return 1

    results = []
    for gen_path, gt_path, root in pairs:
        try:
            m = compute_metrics(gen_path, gt_path)
            results.append({"root": root, **m})
            print(f"{root}: LPIPS={m['lpips']:.4f} SSIM={m['ssim']:.4f} MSE={m['mse']:.4f}")
        except Exception as e:
            print(f"{root}: ERROR {e}", file=sys.stderr)

    if not results:
        return 1

    n = len(results)
    lpips_mean = sum(r["lpips"] for r in results) / n
    ssim_mean = sum(r["ssim"] for r in results) / n
    mse_mean = sum(r["mse"] for r in results) / n
    lpips_std = (sum((r["lpips"] - lpips_mean) ** 2 for r in results) / n) ** 0.5
    ssim_std = (sum((r["ssim"] - ssim_mean) ** 2 for r in results) / n) ** 0.5
    mse_std = (sum((r["mse"] - mse_mean) ** 2 for r in results) / n) ** 0.5
    agg = {
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "n_samples": n,
    }

    print("\n--- Aggregate ---")
    print(f"LPIPS: {agg['lpips_mean']:.4f} ± {agg['lpips_std']:.4f} (lower = better)")
    print(f"SSIM:  {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f} (higher = better)")
    print(f"MSE:   {agg['mse_mean']:.4f} ± {agg['mse_std']:.4f} (lower = better)")
    print(f"N:     {agg['n_samples']}")

    if args.output_json:
        out = {"per_sample": results, "aggregate": agg}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
