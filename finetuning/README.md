# Finetuning (FLUX.2 Edit LoRA on fal.ai)

This folder contains a small Python pipeline to fine-tune a **LoRA adapter** for image editing using:

- `fal-ai/flux-2-trainer/edit` (training)

The output is **not** a full model. You get a **LoRA adapter** (`diffusers_lora.safetensors`) which you apply on top of the base model during inference.

---

## 1) Setup

### Install requirements

From the repo root:

```bash
pip install -r finetuning/requirements.txt
```

### Set your fal API key

Add your key to the project root `.env`:

```env
FAL_KEY=your_fal_key_here
```

Notes:

- `.env` is ignored by git (do not commit keys)
- The CLI automatically loads `.env`

---

## 2) Training dataset format (zip)

fal expects a zip file containing **pairs** of images named like:

- `ROOT_start.png`
- `ROOT_end.png`

Optional:

- `ROOT.txt` (edit instruction for this pair)

If **no** `ROOT.txt` exists, you must pass `--default-caption` when training.

---

## 3) Train a LoRA

### Option A: Build a training zip from `dataset_generation/generated_interiors/`

This expects each sample folder to contain:

- `empty_plan.png` (start)
- `2D_plan.png` (end)

Command:

```bash
python -m finetuning.cli prepare \
  --generated-interiors dataset_generation/generated_interiors \
  --output-zip finetuning/artifacts/train.zip \
  --caption "Describe the edit to learn"
```


### Option B: Use an existing zip file

Example:

```bash
python -m finetuning.cli train finetuning/data/train.zip \
  --steps 500 \
  --learning-rate 5e-5 \
  --default-caption "Describe the edit to learn"
```

This prints a `request_id`.

---

#### Dataset Train_1
```bash
python -m finetuning.cli train finetuning/data/train_1.zip \
  --steps 9 \
  --learning-rate 5e-5 \
  --default-caption "Describe the edit to learn"
```


#### Dataset t  rain_A
```bash
python -m finetuning.cli train finetuning/data/train_A.zip \
  --steps 500 \
  --learning-rate 5e-5 \
  --default-caption "Describe the edit to learn"
```

#### Dataset train_A_RGB.zip
```bash
python -m finetuning.cli train finetuning/data/train_A_RGB.zip --steps 500 --learning-rate 5e-5
```


#### Incremental training for Dataset train_B_RGB.zip
```bash
python -m finetuning.cli train finetuning/data/train_A_RGB.zip \
  --steps 500 \
  --learning-rate 5e-5 \
  --network-weights finetuning/runs/a8e0c8fa-d39a-431a-9c1a-4597dfbe90d5/diffusers_lora.safetensors
```



## 3b) Incremental training (continue from an existing LoRA)

If you already trained a LoRA and want to continue training with a new dataset, pass the previous weights via `--network-weights`.

You can pass either:

- a **local path** to a `.safetensors` file (it will be uploaded), or
- an **https URL** pointing to the weights

Example (continue from a previously downloaded LoRA):

```bash
python -m finetuning.cli train finetuning/data/train_round2.zip \
  --steps 1000 \
  --learning-rate 5e-5 \
  --default-caption "Describe the edit to learn" \
  --network-weights finetuning/runs/<PREVIOUS_REQUEST_ID>/diffusers_lora.safetensors
```

---

## 4) Monitor training

### Check status (non-blocking)

```bash
python -m finetuning.cli status <REQUEST_ID>
```

### Wait for completion + download artifacts

`wait` downloads the LoRA weights and config to the output directory.

Recommended (unbuffered output):

```bash
python -u -m finetuning.cli wait <REQUEST_ID> \
  --output-dir finetuning/runs/<REQUEST_ID>
```

```bash
python -u -m finetuning.cli wait 8ab8b86d-fd15-4fd6-92a6-f0e7d0cb1873 \
  --output-dir finetuning/runs/8ab8b86d-fd15-4fd6-92a6-f0e7d0cb1873
```

```bash
python -u -m finetuning.cli wait 3be61b37-d510-47d7-bd30-be8fa2ea788e --output-dir finetuning/runs/3be61b37-d510-47d7-bd30-be8fa2ea788e
```



Downloads:

- `finetuning/runs/<REQUEST_ID>/diffusers_lora.safetensors`
- `finetuning/runs/<REQUEST_ID>/config.json`

---

## 5) What you get

- `diffusers_lora.safetensors`:
  - The trained **LoRA adapter weights** (apply at inference time)
- `config.json`:
  - Configuration/metadata for the trained adapter

---

## 6) Next step: inference

See `inference/` for:

- base inference (no LoRA)
- inference with your LoRA adapter
