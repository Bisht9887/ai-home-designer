# Training pipeline

This folder contains **step 1** of the training pipeline: **OpenCV preprocessing** for layout images (floor plans, room plans, etc.).

## What it does

- **Input**
  - A dataset root folder that contains **one subfolder per sample/variation** (e.g. `D01_R06_0120/`)
  - Each subfolder should contain **one layout image** (`.png` is typical)
- **Processing**
  - Resize to a fixed target size (default: `512x512`)
  - Optional denoise
  - Convert to grayscale
  - Optional edge detection
- **Output**
  - Writes `layout_processed.png` either:
    - **inside each variation folder** (default), or
    - into a separate `--output-dir` (preserving subfolder names)

If a variation folder has no suitable image, it is skipped.

## Setup

From the repo root:

```bash
pip install -r training_pipeline/requirements.txt
```

## Run

### Process a dataset (write `layout_processed.png` into each sample folder)

```bash
python training_pipeline/run_preprocess.py dataset_generation/Data/D01_120_variations
```

### Process only the first N folders

```bash
python training_pipeline/run_preprocess.py dataset_generation/Data/D01_120_variations --limit 5
```

### Write processed images to a separate directory

```bash
python training_pipeline/run_preprocess.py dataset_generation/Data/D01_120_variations \
  --output-dir training_pipeline/processed
```

### Change the target size and disable edge detection

```bash
python training_pipeline/run_preprocess.py dataset_generation/Data/D01_120_variations \
  --target-size 768 768 \
  --no-edges
```

## Output summary

At the end you’ll see something like:

```text
Done: <processed> processed, <skipped> skipped (no layout image).
```
