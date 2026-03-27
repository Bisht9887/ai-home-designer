# Inference (fal.ai FLUX)

This folder contains a small Python inference pipeline that can run:

- **Base inference** (original FLUX model)
- **LoRA inference** (base model + a LoRA adapter)

The CLI reads `FAL_KEY` from the project root `.env`.

---

## 1) Setup

### Install requirements

From the repo root:

```bash
pip install -r inference/requirements.txt
```

### Set your fal API key

In the project root `.env`:

```env
FAL_KEY=your_fal_key_here
```

---

## 2) Quick commands

### Text-to-image (base)

```bash
python -m inference.cli t2i \
  --prompt "a modern bauhaus living room, top-down" \
  --output-dir inference/outputs
```

### Image-to-image (base)

```bash
python -m inference.cli i2i \
  --image path/to/input.png \
  --prompt "Furnish this floorplan" \
  --output-dir inference/outputs
```

### Image edit with LoRA (FLUX.2 + LoRA adapter)

`--lora` can be a **local path** to a `.safetensors` file or an **https URL**.

```bash
python -m inference.cli i2i-lora \
  --image path/to/input.png \
  --prompt "Apply the learned edit" \
  --lora finetuning/runs/<REQUEST_ID>/diffusers_lora.safetensors \
  --lora-scale 1.0 \
  --output-dir inference/outputs
```

---

## 3) Dataset inference from `inference/data/`

The `dataset` command batches over `inference/data/`.

### Dataset file naming convention

For each item `ROOT`, you need:

- `ROOT_start.png` (or `.jpg/.jpeg/.webp`)
- `ROOT.txt` (prompt)

Example:

- `D11_R01_0001_start.png`
- `D11_R01_0001.txt`

### Run base inference on the dataset

```bash
python -m inference.cli dataset
```

Outputs go to:

- `inference/outputs/<ROOT>/base_1.png`

### Run only the first N items

```bash
python -m inference.cli dataset --limit 5
```

### Run LoRA inference on the dataset

```bash
python -m inference.cli dataset \
  --lora finetuning/runs/<REQUEST_ID>/diffusers_lora.safetensors \
  --lora-scale 1.0
```


####
```bash
python -m inference.cli i2i-lora \
  --image inference/vali_1/D01_R01_0008_start.png \
  --prompt "$(cat inference/vali_1/D01_R01_0008.txt)" \
  --lora finetuning/runs/a8e0c8fa-d39a-431a-9c1a-4597dfbe90d5/diffusers_lora.safetensors \
  --output-dir inference/outputs/vali_1
```

#### Inference for whole directory
The directory must have files named <ROOT>_start.png + <ROOT>.txt (default suffix is _start). Your vali_1 folder already matches this pattern.

With LoRA:

```bash
python -m inference.cli dataset \
  --data-dir inference/vali_1 \
  --lora finetuning/runs/a8e0c8fa-d39a-431a-9c1a-4597dfbe90d5/diffusers_lora.safetensors \
  --output-dir inference/outputs/vali_1_lora
```

Base model only (no --lora):

```bash
python -m inference.cli dataset \
  --data-dir inference/vali_1 \
  --output-dir inference/outputs/vali_1_base
```


Outputs go to:

- `inference/outputs/<ROOT>/lora_1.png`

---

## 4) Notes

- The base dataset mode uses `fal-ai/flux/dev/image-to-image`.
- The LoRA dataset mode uses `fal-ai/flux-2/lora/edit` and uploads both the input image and LoRA weights to fal storage.
- If downloads feel slow, run commands with unbuffered output:

```bash
python -u -m inference.cli dataset
```
