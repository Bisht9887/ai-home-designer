# Evaluation

Compute metrics between generated images and ground truth.

## Metrics

- **LPIPS** – Perceptual similarity (lower = more similar)
- **SSIM** – Structural similarity (0–1, higher = more similar)
- **MSE** – Mean Squared Error, pixel-wise (lower = more similar)

## Setup

```bash
pip install -r evaluation/requirements.txt
```

## Usage

### Flat layout (all files in one folder)

Files: `ROOT_base.png`, `ROOT_end.png`, `ROOT_lora.png` (ground truth = `_end`).

```bash
# Compare LoRA vs ground truth (default)
python -m evaluation.run_eval --flat-dir evaluation/eval_1

# Compare base vs ground truth
python -m evaluation.run_eval --flat-dir evaluation/eval_1 --generated-suffix _base

# Save to JSON
python -m evaluation.run_eval --flat-dir evaluation/eval_1 --output-json evaluation/results.json

# Compare LoRA vs Base and show % improvement
python -m evaluation.run_eval --flat-dir evaluation/eval_1 --compare

# Save for UI display (frontend reads evaluation/results.json)
python -m evaluation.run_eval --flat-dir evaluation/eval_1 --compare --output-json evaluation/results.json
```

### Nested layout (inference outputs)

- **Generated:** `inference/outputs/<ROOT>/base_1.png` (or `lora_1.png`)
- **Ground truth:** `inference/data/<ROOT>_end.png`

```bash
python -m evaluation.run_eval --generated-dir inference/outputs --ground-truth-dir inference/data
```
